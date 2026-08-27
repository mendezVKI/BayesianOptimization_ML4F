"""
Multi-fidelity Bayesian Optimization core: config containers, autoregressive
(AR1 / Kennedy & O'Hagan) two-fidelity GP, cost-aware acquisition over both a
location x and a fidelity level, and the main MFBO driver.

Within the research of the Machine Learning for Fluid Systems group (ML4F)

link: https://www.mendezma.com/

Ported from an exploratory script (src/mfbo/MFGP_BO_functions_legacy.py) that
mixed the multi-fidelity GP/acquisition math together with unrelated
hardware-control code. Only the mathematics (joint AR1 covariance, EI, and
cost-normalized fidelity selection) was kept; everything else was rebuilt
from scratch to match the structure of sbo (src/sbo/core.py). mfbo is
intentionally independent of sbo -- see the package-level docstring in
__init__.py.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .persistence import TraceLog, GPSnapshot

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
import copy
import os
import time

from .saving import setup_experiment_folder, setup_logger, log_update

Array = np.ndarray
Bounds = List[Tuple[float, float]]  # [(l1,u1), ..., (ld,ud)]

# The only two fidelity levels currently supported. Extending the AR1 model
# to more than two levels requires a recursive formulation (Le Gratiet, 2013)
# and is left for a future revision.
Level = str  # "L" or "H"


#%% Settings / Config containers

@dataclass
class GPConfig:
    """Hyperparameters of the two-fidelity autoregressive (AR1) GP:

        f_H(x) = rho * f_L(x) + delta(x)

    f_L (low fidelity) and delta (the high-fidelity discrepancy) are each
    modeled by an independent amplitude-parameterized RBF kernel; rho is the
    scalar autoregressive scaling from Kennedy & O'Hagan (2000).
    """

    # if optimize_hyperparams=False, these values are used and kept fixed
    l_lf: float = 0.2          # low-fidelity length-scale
    sigma_lf: float = 1.0      # low-fidelity kernel amplitude std
    l_delta: float = 0.2       # discrepancy length-scale
    sigma_delta: float = 0.5   # discrepancy kernel amplitude std
    rho: float = 1.0           # autoregressive scaling f_H ~ rho*f_L + delta
    sigma_L: float = 0.1       # low-fidelity observation noise std
    sigma_H: float = 0.05      # high-fidelity observation noise std
    jitter: float = 1e-10      # numerical stabilizer on diagonal

    # hyperparameter optimization (HPO) options
    optimize_hyperparams: bool = False
    hpo_every: int = 1        # optimize every k BO iterations (1 = every iteration)

    # Optional initial guess and bounds for the 7 free parameters, in the
    # order [rho, log(l_lf), log(sigma_lf), log(l_delta), log(sigma_delta),
    # log(sigma_L), log(sigma_H)]. rho itself is left un-transformed (it can
    # be negative), the remaining six positive scales are optimized in
    # log-space.
    theta0: Optional[Array] = None
    theta_bounds: Optional[List[Tuple[float, float]]] = None


@dataclass
class FidelityConfig:
    """Evaluation cost of each fidelity level, used by the acquisition to
    trade off information gain against cost."""

    cost_low: float = 1.0
    cost_high: float = 10.0

    def cost(self, level: Level) -> float:
        if level == "L":
            return self.cost_low
        if level == "H":
            return self.cost_high
        raise ValueError(f"Unknown fidelity level '{level}'. Use 'L' or 'H'.")


@dataclass
class AcqConfig:
    kind: str = "EI"  # only "EI" is currently implemented
    xi: float = 0.01
    maximize: bool = False  # typical BO for minimization -> False


@dataclass
class OptimConfig:
    method: str = "random"
    global_method: str = "random"

    optimizer: str = "L-BFGS-B"
    n_raw_samples: int = 2000
    grid_n_per_dim: int = 20
    n_restarts: int = 10
    local_maxiter: int = 200


@dataclass
class MFBOConfig:
    # --- Pure opti related parameters
    n_init_L: int = 10   # size of the initial low-fidelity data set
    n_init_H: int = 3    # size of the initial high-fidelity data set (must be >= 1)
    n_iter: int = 20     # number of BO iterations - in the high fidelity
    init_sampling: str = "lhs"  # "uniform" or "lhs"
    random_state: Optional[int] = None


@dataclass
class SaveConfig:
    # --- Saving related parameters
    out_path: Optional[str] = None
    create_timestamp: bool = True
    log_enabled: bool = True
    log_filename: str = "log.log"

    # --- Tier 1: lightweight per-iteration trace (numpy .npz)
    # Same contract as sbo, plus the fidelity level and cost of each
    # evaluation and the running cumulative cost (see persistence.TraceLog).
    trace_enabled: bool = True
    trace_filename: str = "trace.npz"
    trace_flush_every: Optional[int] = None

    # --- Tier 2: full posterior-reconstruction snapshots (opt-in, heavier)
    snapshot_enabled: bool = False
    snapshot_every: int = 1
    snapshot_dirname: str = "snapshots"
    snapshot_flush_every: Optional[int] = None

    # --- Plotting related parameters
    plt_state_enabled: bool = False
    plt_conv_enabled: bool = False
    plt_hist_enabled: bool = False
    plt_MLE_conv_enbable: bool = False
    plt_all: bool = False
    plot_every: int = 1
    dpi: int = 250


# ----------------------------
# GP model container
# ----------------------------

@dataclass
class NormalizationHelper:
    """Normalizes X only. y is intentionally left in physical units: the AR1
    kernel amplitudes (sigma_lf, sigma_delta) and the rho scaling already
    absorb the relative scale between the two fidelities, and normalizing y
    per-fidelity would fight with rho's meaning (rho relates the *physical*
    low- and high-fidelity outputs)."""

    x_lo: Array
    x_hi: Array
    eps: float = 1e-12

    def normalize_X(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        denom = np.maximum(self.x_hi - self.x_lo, self.eps)
        return (X - self.x_lo) / denom

    def denormalize_X(self, Xn: Array) -> Array:
        Xn = np.asarray(Xn, dtype=float)
        return self.x_lo + Xn * (self.x_hi - self.x_lo)


@dataclass
class MFGPModel:
    l_lf: float
    sigma_lf: float
    l_delta: float
    sigma_delta: float
    rho: float
    sigma_L: float
    sigma_H: float
    jitter: float

    X_L: Optional[Array] = None
    y_L: Optional[Array] = None
    X_H: Optional[Array] = None
    y_H: Optional[Array] = None
    alpha: Optional[Array] = None
    L: Optional[Array] = None

    theta: Optional[Array] = None

    # Normalized (X only) training data
    X_L_norm: Optional[Array] = None
    X_H_norm: Optional[Array] = None
    normalizer: Optional[NormalizationHelper] = None


# ----------------------------
# State / Results containers
# ----------------------------

@dataclass
class MFBOState:
    it: int
    X_L: Array
    y_L: Array
    X_H: Array
    y_H: Array
    gp: MFGPModel
    acq_res: "MFAcqOptimizationResult"
    x_next: Array
    level_next: Level
    y_next: float
    y_best: float          # incumbent, defined at the high fidelity only
    x_best: Array
    cost_next: float
    cumulative_cost: float


@dataclass
class MFBOResult:
    X_L: Array
    y_L: Array
    X_H: Array
    y_H: Array
    best_x: Array
    best_y: float           # best high-fidelity observation (the incumbent)
    history: List[Dict[str, Any]]
    gp: MFGPModel
    states: List[MFBOState]

    trace: Optional["TraceLog"] = None
    snapshots: Optional[List["GPSnapshot"]] = None


@dataclass
class MFAcqOptimizationResult:
    x_next: Array           # (d,)
    level_next: Level
    a_best: float
    Xcand: Array             # (N, d)
    a_L: Array                # (N,) acquisition value of every candidate, at low fidelity
    a_H: Array                 # (N,) acquisition value of every candidate, at high fidelity
    run_time: float


#%% Helper utilities

def _rng(random_state: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(random_state)


def sample_initial_points(bounds: Bounds, n_init: int, rng: np.random.Generator, method: str = "uniform") -> Array:
    d = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    if method == "uniform":
        X_unit = rng.random((n_init, d))
    elif method == "lhs":
        sampler = LatinHypercube(d=d, seed=rng)
        X_unit = sampler.random(n=n_init)
    else:
        raise ValueError(f"Unknown sampling method '{method}'. Use 'uniform' or 'lhs'.")

    return lo + X_unit * (hi - lo)


def evaluate_objective(f: Callable[[Array], Union[float, Array]], X: Array) -> Array:
    y_list = []
    for i in range(X.shape[0]):
        val = f(X[i])
        val = float(np.asarray(val).reshape(-1)[0])
        y_list.append(val)
    return np.asarray(y_list, dtype=float)


def _check_in_bounds(X: Array, bounds: Bounds) -> None:
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    if np.any(X < lo) or np.any(X > hi):
        raise ValueError("Some initial points lie outside the prescribed bounds.")


def init_dataset(
    f_low: Callable[[Array], Union[float, Array]],
    f_high: Callable[[Array], Union[float, Array]],
    bounds: Bounds,
    mfbo_cfg: MFBOConfig,
    rng: np.random.Generator,
    X_L_init: Optional[Array] = None,
    y_L_init: Optional[Array] = None,
    X_H_init: Optional[Array] = None,
    y_H_init: Optional[Array] = None,
) -> Tuple[Array, Array, Array, Array]:
    """Initialize the low- and high-fidelity datasets, either from
    user-provided evaluations or by sampling mfbo_cfg.n_init_L /
    mfbo_cfg.n_init_H points per fidelity."""

    if mfbo_cfg.n_init_H < 1:
        raise ValueError(
            "mfbo_cfg.n_init_H must be >= 1: the incumbent is defined at the "
            "highest fidelity only, so at least one high-fidelity evaluation "
            "is required before the loop starts."
        )

    d_bounds = len(bounds)

    if X_L_init is None:
        X_L = sample_initial_points(bounds, mfbo_cfg.n_init_L, rng, method=mfbo_cfg.init_sampling)
        y_L = evaluate_objective(f_low, X_L)
    else:
        X_L = np.asarray(X_L_init, dtype=float)
        if X_L.ndim == 1:
            X_L = X_L.reshape(1, -1)
        if X_L.shape[1] != d_bounds:
            raise ValueError(f"X_L_init has shape {X_L.shape}, expected dimension {d_bounds}.")
        _check_in_bounds(X_L, bounds)
        y_L = evaluate_objective(f_low, X_L) if y_L_init is None else np.asarray(y_L_init, dtype=float).reshape(-1)

    if X_H_init is None:
        X_H = sample_initial_points(bounds, mfbo_cfg.n_init_H, rng, method=mfbo_cfg.init_sampling)
        y_H = evaluate_objective(f_high, X_H)
    else:
        X_H = np.asarray(X_H_init, dtype=float)
        if X_H.ndim == 1:
            X_H = X_H.reshape(1, -1)
        if X_H.shape[1] != d_bounds:
            raise ValueError(f"X_H_init has shape {X_H.shape}, expected dimension {d_bounds}.")
        _check_in_bounds(X_H, bounds)
        y_H = evaluate_objective(f_high, X_H) if y_H_init is None else np.asarray(y_H_init, dtype=float).reshape(-1)

    if X_H.shape[0] < 1:
        raise ValueError("At least one high-fidelity initial point is required.")

    return X_L, y_L, X_H, y_H


# ----------------------------
# AR1 (Kennedy & O'Hagan) joint GP: kernel + fit / prediction
# ----------------------------

def rbf_kernel(X1, X2, gamma) -> Array:
    sqdist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
    return np.exp(-gamma * sqdist)


def rbf_kernel_amp(X1: Array, X2: Array, l_c: float, sigma_f: float) -> Array:
    """k(x,x') = sigma_f^2 * exp(-||x-x'||^2 / (2 l_c^2))"""
    gamma = 0.5 / (l_c ** 2)
    return (sigma_f ** 2) * rbf_kernel(X1, X2, gamma=gamma)


def _cross_cov(Xa: Array, level_a: Level, Xb: Array, level_b: Level, gp: MFGPModel) -> Array:
    """Cov(f_level_a(Xa), f_level_b(Xb)) under the AR1 model
    f_H = rho*f_L + delta, with f_L ~ GP(0, k_lf) and delta ~ GP(0, k_delta)
    independent of f_L. Implements the four blocks used throughout
    build_joint_covariance / mf_gp_predict / predictive_variance_reduction,
    so those all stay consistent by construction."""

    if level_a == "L" and level_b == "L":
        return rbf_kernel_amp(Xa, Xb, gp.l_lf, gp.sigma_lf)
    if level_a == "H" and level_b == "H":
        return (gp.rho ** 2) * rbf_kernel_amp(Xa, Xb, gp.l_lf, gp.sigma_lf) + rbf_kernel_amp(Xa, Xb, gp.l_delta, gp.sigma_delta)
    if {level_a, level_b} == {"L", "H"}:
        return gp.rho * rbf_kernel_amp(Xa, Xb, gp.l_lf, gp.sigma_lf)
    raise ValueError(f"Unknown fidelity pair ({level_a!r}, {level_b!r}).")


def _prior_var_H(gp: MFGPModel) -> float:
    """Prior (pre-data) variance of f_H at a single point."""
    return (gp.rho ** 2) * (gp.sigma_lf ** 2) + gp.sigma_delta ** 2


def _prior_var_L(gp: MFGPModel) -> float:
    return gp.sigma_lf ** 2


def _prior_cross_H_L(gp: MFGPModel) -> float:
    """Cov(f_H(x), f_L(x)) at the same x, before conditioning on any data."""
    return gp.rho * (gp.sigma_lf ** 2)


def build_joint_covariance(gp: MFGPModel, X_L: Array, X_H: Array) -> Array:
    """Assemble the joint (n_L+n_H, n_L+n_H) covariance of [f_L(X_L); f_H(X_H)]."""

    n_L = X_L.shape[0]
    n_H = X_H.shape[0]

    K_LL = _cross_cov(X_L, "L", X_L, "L", gp) + (gp.sigma_L ** 2 + gp.jitter) * np.eye(n_L)
    K_LH = _cross_cov(X_L, "L", X_H, "H", gp)
    K_HH = _cross_cov(X_H, "H", X_H, "H", gp) + (gp.sigma_H ** 2 + gp.jitter) * np.eye(n_H)

    top = np.hstack([K_LL, K_LH])
    bottom = np.hstack([K_LH.T, K_HH])
    return np.vstack([top, bottom])


def mf_gp_fit(gp: MFGPModel, X_L: Array, y_L: Array, X_H: Array, y_H: Array) -> Tuple[Array, Array]:
    """Compute the joint Cholesky factor and alpha. X_L, X_H are expected to
    already be normalized (see fit_gp)."""

    K = build_joint_covariance(gp, X_L, X_H)
    L = cholesky(K, lower=True)
    y = np.concatenate([np.asarray(y_L, dtype=float).reshape(-1), np.asarray(y_H, dtype=float).reshape(-1)])
    alpha = cho_solve((L, True), y)
    return alpha, L


def mf_gp_predict(
    Xtest: Array,
    gp: MFGPModel,
    X_L: Array,
    X_H: Array,
    alpha: Array,
    L: Array,
    return_cov: bool = False,
):
    """Predict the HIGH-fidelity (target) function at Xtest, using the
    fitted joint GP. If gp.normalizer is set, Xtest is assumed physical and
    X_L/X_H must already be normalized (as stored on a fitted gp)."""

    Xtest = np.asarray(Xtest, dtype=float)
    if gp.normalizer is not None:
        Xtest_n = gp.normalizer.normalize_X(Xtest)
    else:
        Xtest_n = Xtest

    K_s = np.hstack([
        _cross_cov(Xtest_n, "H", X_L, "L", gp),
        _cross_cov(Xtest_n, "H", X_H, "H", gp),
    ])  # (ntest, n_L+n_H)

    mu = K_s @ alpha
    v = solve_triangular(L, K_s.T, lower=True)  # (n, ntest)

    if return_cov:
        K_ss = _cross_cov(Xtest_n, "H", Xtest_n, "H", gp)
        cov = K_ss - v.T @ v
        return mu, cov

    var = _prior_var_H(gp) * np.ones(Xtest_n.shape[0]) - np.sum(v ** 2, axis=0)
    var = np.maximum(var, 0.0)
    return mu, var


def predictive_variance_reduction(Xcand: Array, level: Level, gp: MFGPModel, X_L: Array, X_H: Array, L: Array) -> Array:
    """Expected reduction in the HIGH-fidelity posterior variance at each
    candidate x, if a (noisy) observation at fidelity `level` were taken at
    that same x. Uses the standard rank-1 (Schur complement) GP update
    formula, without needing the observed value: for a hypothetical new
    point with prior self-variance s and cross-covariance u to the target,
    conditioned on the existing training set,

        var_after(x) = var_before(x) - u(x)^2 / s(x)

    Vectorized over all candidates via batched triangular solves. Returns a
    non-negative array of the same length as Xcand.
    """

    Xcand = np.atleast_2d(np.asarray(Xcand, dtype=float))

    K_s_H = np.hstack([
        _cross_cov(Xcand, "H", X_L, "L", gp),
        _cross_cov(Xcand, "H", X_H, "H", gp),
    ])
    v = solve_triangular(L, K_s_H.T, lower=True)  # (n, ncand)
    var_before = np.maximum(_prior_var_H(gp) - np.sum(v ** 2, axis=0), 0.0)

    if level == "H":
        w = v
        u = var_before  # Cov(f_H(x),f_H(x)) - w.v == prior_H - v.v == var_before, by construction
        prior_self = _prior_var_H(gp)
        noise = gp.sigma_H ** 2
    elif level == "L":
        K_s_L = np.hstack([
            _cross_cov(Xcand, "L", X_L, "L", gp),
            _cross_cov(Xcand, "L", X_H, "H", gp),
        ])
        w = solve_triangular(L, K_s_L.T, lower=True)
        u = _prior_cross_H_L(gp) - np.sum(w * v, axis=0)
        prior_self = _prior_var_L(gp)
        noise = gp.sigma_L ** 2
    else:
        raise ValueError(f"Unknown fidelity level '{level}'. Use 'L' or 'H'.")

    s = np.maximum(prior_self + noise + gp.jitter - np.sum(w ** 2, axis=0), 1e-12)
    var_after = var_before - (u ** 2) / s
    var_after = np.clip(var_after, 0.0, var_before)
    return var_before - var_after


# ----------------------------
# Hyperparameter optimization by log marginal likelihood
# ----------------------------

def negative_log_marginal_likelihood(
    theta: Array,
    X_L: Array,
    y_L: Array,
    X_H: Array,
    y_H: Array,
    jitter: float = 1e-10,
    MLE_hist: Optional[list] = None,
    param_hist: Optional[list] = None,
) -> float:
    """theta = [rho, log(l_lf), log(sigma_lf), log(l_delta), log(sigma_delta),
    log(sigma_L), log(sigma_H)]. Returns NLL = -log p(y_L, y_H | X_L, X_H, theta)."""

    rho = float(theta[0])
    l_lf, sigma_lf, l_delta, sigma_delta, sigma_L, sigma_H = np.exp(theta[1:])

    gp_tmp = MFGPModel(
        l_lf=l_lf, sigma_lf=sigma_lf, l_delta=l_delta, sigma_delta=sigma_delta,
        rho=rho, sigma_L=sigma_L, sigma_H=sigma_H, jitter=jitter,
    )

    try:
        K = build_joint_covariance(gp_tmp, X_L, X_H)
        L = cholesky(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e12

    y = np.concatenate([np.asarray(y_L, dtype=float).reshape(-1), np.asarray(y_H, dtype=float).reshape(-1)])
    alpha = cho_solve((L, True), y)
    n = y.shape[0]
    ll = -0.5 * (y @ alpha) - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2.0 * np.pi)

    if MLE_hist is not None:
        MLE_hist.append(float(ll))
    if param_hist is not None:
        param_hist.append([rho, float(l_lf), float(sigma_lf), float(l_delta), float(sigma_delta), float(sigma_L), float(sigma_H)])

    return -float(ll)


def optimize_gp_hyperparams(
    X_L: Array,
    y_L: Array,
    X_H: Array,
    y_H: Array,
    gp: MFGPModel,
    gp_cfg: GPConfig,
    MLE_hist: Optional[list] = None,
    param_hist: Optional[list] = None,
) -> MFGPModel:
    """Update gp.(rho, l_lf, sigma_lf, l_delta, sigma_delta, sigma_L,
    sigma_H) by minimizing the negative joint log marginal likelihood."""

    if gp_cfg.theta0 is not None:
        x0 = np.asarray(gp_cfg.theta0, dtype=float).reshape(-1)
    else:
        x0 = np.concatenate([
            [gp.rho],
            np.log([gp.l_lf, gp.sigma_lf, gp.l_delta, gp.sigma_delta, gp.sigma_L, gp.sigma_H]),
        ])

    res = minimize(
        negative_log_marginal_likelihood,
        x0=x0,
        args=(X_L, y_L, X_H, y_H, gp.jitter, MLE_hist, param_hist),
        method="L-BFGS-B",
        bounds=gp_cfg.theta_bounds,
    )

    theta_opt = res.x
    gp.rho = float(theta_opt[0])
    gp.l_lf, gp.sigma_lf, gp.l_delta, gp.sigma_delta, gp.sigma_L, gp.sigma_H = (float(v) for v in np.exp(theta_opt[1:]))
    gp.theta = theta_opt.copy()

    return gp


# ----------------------------
# GP wrapper used by MFBO
# ----------------------------

def build_gp_model(gp_cfg: GPConfig, bounds: Bounds) -> MFGPModel:
    """Create an unfitted MFGPModel container."""
    x_lo = np.array([b[0] for b in bounds], dtype=float)
    x_hi = np.array([b[1] for b in bounds], dtype=float)
    normalizer = NormalizationHelper(x_lo=x_lo, x_hi=x_hi)

    return MFGPModel(
        l_lf=gp_cfg.l_lf, sigma_lf=gp_cfg.sigma_lf,
        l_delta=gp_cfg.l_delta, sigma_delta=gp_cfg.sigma_delta,
        rho=gp_cfg.rho, sigma_L=gp_cfg.sigma_L, sigma_H=gp_cfg.sigma_H,
        jitter=gp_cfg.jitter, normalizer=normalizer,
    )


def fit_gp(gp: MFGPModel, X_L: Array, y_L: Array, X_H: Array, y_H: Array, gp_cfg: GPConfig, it: int) -> MFGPModel:
    X_L = np.asarray(X_L, dtype=float)
    y_L = np.asarray(y_L, dtype=float).reshape(-1)
    X_H = np.asarray(X_H, dtype=float)
    y_H = np.asarray(y_H, dtype=float).reshape(-1)

    if gp.normalizer is None:
        raise ValueError("GP normalizer is not initialized.")

    gp.X_L, gp.y_L, gp.X_H, gp.y_H = X_L, y_L, X_H, y_H
    gp.X_L_norm = gp.normalizer.normalize_X(X_L)
    gp.X_H_norm = gp.normalizer.normalize_X(X_H)

    do_hpo = (
        gp_cfg.optimize_hyperparams
        and gp_cfg.hpo_every > 0
        and (it % gp_cfg.hpo_every == 0)
    )
    if do_hpo:
        gp = optimize_gp_hyperparams(gp.X_L_norm, y_L, gp.X_H_norm, y_H, gp, gp_cfg)

    gp.alpha, gp.L = mf_gp_fit(gp, gp.X_L_norm, y_L, gp.X_H_norm, y_H)
    return gp


# ----------------------------
# Acquisition + optimizer
# ----------------------------

def make_acquisition(acq_cfg: AcqConfig, fidelity_cfg: FidelityConfig, gp: MFGPModel, y_best: float):
    """
    Build the cost-aware acquisition value(Xcand, level).

    Convention: value(.) is MAXIMIZED, and combines three ingredients:
      - EI(x): standard expected improvement of the HIGH-fidelity posterior
        at x against the (high-fidelity-only) incumbent y_best.
      - a variance-reduction weight in [0,1]: the fraction of the current
        predictive uncertainty on f_H(x) that would be resolved by sampling
        `level` at x (see predictive_variance_reduction) -- this is what
        lets a cheap low-fidelity sample compete with a high-fidelity one
        when it is still highly informative about the target.
      - division by fidelity_cfg.cost(level).

    This generalizes the legacy script's cost_aware_EI = EI / cost: without
    the variance-reduction weight, cost-normalized EI degenerates to always
    picking the cheapest fidelity (since EI(x) itself does not depend on
    which fidelity would be sampled).
    """

    if acq_cfg.kind.upper() != "EI":
        raise ValueError(f"Unknown acquisition kind '{acq_cfg.kind}'. Only 'EI' is implemented.")

    def value(Xcand: Array, level: Level) -> Array:
        Xcand = np.atleast_2d(np.asarray(Xcand, dtype=float))

        mu, var = mf_gp_predict(Xcand, gp, gp.X_L_norm, gp.X_H_norm, gp.alpha, gp.L, return_cov=False)
        var = np.maximum(var, 0.0)
        sigma = np.sqrt(var)

        eps = 1e-12
        sigma_safe = np.maximum(sigma, eps)

        if not acq_cfg.maximize:
            improvement = y_best - mu - acq_cfg.xi
        else:
            improvement = mu - y_best - acq_cfg.xi

        Z = improvement / sigma_safe
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = np.where(sigma > eps, ei, 0.0)

        Xcand_n = gp.normalizer.normalize_X(Xcand) if gp.normalizer is not None else Xcand
        vr = predictive_variance_reduction(Xcand_n, level, gp, gp.X_L_norm, gp.X_H_norm, gp.L)

        weight = np.zeros_like(vr)
        nonzero = var > eps
        weight[nonzero] = np.clip(vr[nonzero] / var[nonzero], 0.0, 1.0)

        return (ei * weight) / fidelity_cfg.cost(level)

    return value


def _make_random_candidates(bounds: Bounds, N: int, rng: np.random.Generator) -> Array:
    d = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo + (hi - lo) * rng.random((N, d))


def _make_grid_candidates(bounds: Bounds, n_per_dim: int) -> Array:
    grids_1d = [np.linspace(b[0], b[1], n_per_dim) for b in bounds]
    mesh = np.meshgrid(*grids_1d, indexing="xy")
    X = np.stack([m.ravel() for m in mesh], axis=1)
    return X


def optimize_acquisition(
    value: Callable[[Array, Level], Array],
    bounds: Bounds,
    optim_cfg: OptimConfig,
    rng: np.random.Generator,
) -> MFAcqOptimizationResult:
    """
    Jointly optimize the acquisition over location x AND fidelity level.

    Supported modes (mirrors sbo.core.optimize_acquisition):
      - method="random": global random candidates only
      - method="grid":   global grid candidates only
      - method="refined": global (random or grid) + local L-BFGS-B
        refinement of x, for the fidelity level that won the global scan.
    """

    method = optim_cfg.method.lower()
    if method not in ["random", "grid", "refined"]:
        raise ValueError("optim_cfg.method must be 'random', 'grid', or 'refined'.")

    start_time = time.perf_counter()

    if method == "random":
        Xcand = _make_random_candidates(bounds, optim_cfg.n_raw_samples, rng)
    elif method == "grid":
        Xcand = _make_grid_candidates(bounds, optim_cfg.grid_n_per_dim)
    else:
        gmethod = optim_cfg.global_method.lower()
        if gmethod == "random":
            Xcand = _make_random_candidates(bounds, optim_cfg.n_raw_samples, rng)
        elif gmethod == "grid":
            Xcand = _make_grid_candidates(bounds, optim_cfg.grid_n_per_dim)
        else:
            raise ValueError("optim_cfg.global_method must be 'random' or 'grid' when method='refined'.")

    a_L = np.asarray(value(Xcand, "L"), dtype=float).reshape(-1)
    a_H = np.asarray(value(Xcand, "H"), dtype=float).reshape(-1)
    a_L[~np.isfinite(a_L)] = -np.inf
    a_H[~np.isfinite(a_H)] = -np.inf

    idx_L = int(np.argmax(a_L))
    idx_H = int(np.argmax(a_H))

    if a_L[idx_L] >= a_H[idx_H]:
        level_best, idx_best, a_best = "L", idx_L, float(a_L[idx_L])
    else:
        level_best, idx_best, a_best = "H", idx_H, float(a_H[idx_H])
    x_best = Xcand[idx_best].copy()

    if method in ["random", "grid"]:
        run_time = time.perf_counter() - start_time
        return MFAcqOptimizationResult(
            x_next=x_best, level_next=level_best, a_best=a_best,
            Xcand=Xcand, a_L=a_L, a_H=a_H, run_time=run_time,
        )

    # ---- refined: local improvement of x, at the fidelity level that won
    a_level = a_L if level_best == "L" else a_H
    n_starts = min(optim_cfg.n_restarts, Xcand.shape[0])
    top_idx = np.argpartition(-a_level, n_starts - 1)[:n_starts]
    Xstarts = Xcand[top_idx]

    scipy_bounds = [(float(b[0]), float(b[1])) for b in bounds]

    def obj(x: Array) -> float:
        x2 = np.asarray(x, dtype=float).reshape(1, -1)
        val = float(np.asarray(value(x2, level_best)).reshape(-1)[0])
        if not np.isfinite(val):
            return 1e30
        return -val

    for x0 in Xstarts:
        res = minimize(
            obj, x0=x0, method=optim_cfg.optimizer, bounds=scipy_bounds,
            options=dict(maxiter=optim_cfg.local_maxiter),
        )
        if res.success and res.x is not None:
            a_try = -float(res.fun)
            if np.isfinite(a_try) and a_try > a_best:
                a_best = a_try
                x_best = np.asarray(res.x, dtype=float).copy()

    run_time = time.perf_counter() - start_time
    return MFAcqOptimizationResult(
        x_next=x_best, level_next=level_best, a_best=a_best,
        Xcand=Xcand, a_L=a_L, a_H=a_H, run_time=run_time,
    )


#%% Main MFBO function

def multi_fidelity_bayesian_optimization(
    f_low: Callable[[Array], Union[float, Array]],
    f_high: Callable[[Array], Union[float, Array]],
    bounds: Bounds,
    mfbo_cfg: MFBOConfig,
    gp_cfg: GPConfig,
    acq_cfg: AcqConfig,
    optim_cfg: OptimConfig,
    fidelity_cfg: FidelityConfig,
    save_cfg: SaveConfig,
    X_L_init: Optional[Array] = None,
    y_L_init: Optional[Array] = None,
    X_H_init: Optional[Array] = None,
    y_H_init: Optional[Array] = None,
    states: Optional[List[MFBOState]] = None,
    f_low_true: Optional[Callable[[Array], Array]] = None,
    f_high_true: Optional[Callable[[Array], Array]] = None,
) -> MFBOResult:
    """
    Multi-fidelity Bayesian Optimization driver (minimization by default).

    Each iteration jointly proposes a location x and a fidelity level
    ("L" or "H") by maximizing the cost-aware acquisition (see
    make_acquisition), evaluates the corresponding objective (f_low or
    f_high) at x, and appends the result to that fidelity's dataset. The
    incumbent (best_x, best_y) is always taken from the high-fidelity data
    only.
    """
    from .plotting import setup_plotting_toggle, plt_state, plt_hist, plt_conv
    from . import persistence

    rng = _rng(mfbo_cfg.random_state)

    if save_cfg.trace_flush_every is not None and save_cfg.out_path is None:
        raise ValueError("save_cfg.trace_flush_every requires save_cfg.out_path to be set.")
    if save_cfg.snapshot_flush_every is not None and save_cfg.out_path is None:
        raise ValueError("save_cfg.snapshot_flush_every requires save_cfg.out_path to be set.")

    if states is None:
        states = []

    experiment_path = setup_experiment_folder(save_cfg)
    setup_plotting_toggle(save_cfg)

    logger = None
    if save_cfg.log_enabled and experiment_path is not None:
        log_path = os.path.join(experiment_path, save_cfg.log_filename)
        logger = setup_logger(log_path)
        logger.info(f"Bounds: {bounds}")
        logger.info(f"MFBOConfig: {mfbo_cfg}")
        logger.info(f"GPConfig: {gp_cfg}")
        logger.info(f"AcqConfig: {acq_cfg}")
        logger.info(f"OptimConfig: {optim_cfg}")
        logger.info(f"FidelityConfig: {fidelity_cfg}")
        logger.info(f"SaveConfig: {save_cfg}")
        logger.info(f"Random seed: {mfbo_cfg.random_state}")
        logger.info(
            "it  | lvl |    x_next     |      y_next     |      y_best     |   l_lf    |  l_delta  |    rho    |    cost   | cum_cost"
        )
        logger.info("-" * 120)

    history: List[Dict[str, Any]] = []

    trace = persistence.TraceLog() if save_cfg.trace_enabled else None
    trace_path = (
        os.path.join(experiment_path, save_cfg.trace_filename)
        if (experiment_path is not None and trace is not None)
        else None
    )

    snapshot_buffer: Optional[List["persistence.GPSnapshot"]] = (
        [] if save_cfg.snapshot_enabled else None
    )
    snapshot_dir = (
        os.path.join(experiment_path, save_cfg.snapshot_dirname)
        if (experiment_path is not None and snapshot_buffer is not None)
        else None
    )

    if experiment_path is not None:
        persistence.write_meta(
            experiment_path,
            bounds=bounds,
            random_state=mfbo_cfg.random_state,
            mfbo_cfg=mfbo_cfg,
            gp_cfg=gp_cfg,
            acq_cfg=acq_cfg,
            optim_cfg=optim_cfg,
            fidelity_cfg=fidelity_cfg,
            save_cfg=save_cfg,
        )

    # ---- 1) initial datasets
    X_L, y_L, X_H, y_H = init_dataset(
        f_low=f_low, f_high=f_high, bounds=bounds, mfbo_cfg=mfbo_cfg, rng=rng,
        X_L_init=X_L_init, y_L_init=y_L_init, X_H_init=X_H_init, y_H_init=y_H_init,
    )

    # ---- 2) build GP object
    gp = build_gp_model(gp_cfg, bounds)

    cumulative_cost = 0.0

    # ---- 3) MFBO loop
    for it in range(mfbo_cfg.n_iter):
        iter_start_time = time.perf_counter()

        gp = fit_gp(gp, X_L, y_L, X_H, y_H, gp_cfg, it)

        # Incumbent: high fidelity only
        best_idx = int(np.argmin(y_H))
        y_best = float(y_H[best_idx])
        x_best = X_H[best_idx].copy()

        acq_value = make_acquisition(acq_cfg, fidelity_cfg, gp, y_best)
        acq_res = optimize_acquisition(acq_value, bounds, optim_cfg, rng)

        x_next = acq_res.x_next
        level_next = acq_res.level_next
        f_eval = f_low if level_next == "L" else f_high
        y_next = float(np.asarray(f_eval(x_next)).reshape(-1)[0])
        cost_next = fidelity_cfg.cost(level_next)
        cumulative_cost += cost_next

        if level_next == "L":
            X_L = np.vstack([X_L, x_next[None, :]])
            y_L = np.concatenate([y_L, [y_next]])
        else:
            X_H = np.vstack([X_H, x_next[None, :]])
            y_H = np.concatenate([y_H, [y_next]])

        if save_cfg.plt_state_enabled and len(bounds) == 1 and (it % save_cfg.plot_every == 0):
            plt_state(
                gp=gp, acq_res=acq_res, bounds=bounds, it=it,
                x_best=x_best, y_best=y_best, save_cfg=save_cfg,
                acq_cfg=acq_cfg, mfbo_cfg=mfbo_cfg, value=acq_value,
                f_low_true=f_low_true, f_high_true=f_high_true,
            )

        state = MFBOState(
            it=it, X_L=X_L.copy(), y_L=y_L.copy(), X_H=X_H.copy(), y_H=y_H.copy(),
            gp=copy.deepcopy(gp), acq_res=acq_res,
            x_next=x_next, level_next=level_next, y_next=y_next,
            y_best=y_best, x_best=x_best,
            cost_next=cost_next, cumulative_cost=cumulative_cost,
        )
        states.append(state)

        iter_wall_time = time.perf_counter() - iter_start_time

        if trace is not None:
            trace.append(
                it=it, x_next=x_next, level_next=level_next, y_next=y_next,
                x_best=x_best, y_best=y_best, wall_time=iter_wall_time,
                acq_value=acq_res.a_best, cost_next=cost_next,
                cumulative_cost=cumulative_cost,
            )
            if (
                save_cfg.trace_flush_every is not None
                and (it + 1) % save_cfg.trace_flush_every == 0
                and trace_path is not None
            ):
                trace.save(trace_path)

        if snapshot_buffer is not None and (it % save_cfg.snapshot_every == 0):
            snapshot_buffer.append(persistence.build_gp_snapshot(it, gp))
            if (
                save_cfg.snapshot_flush_every is not None
                and len(snapshot_buffer) >= save_cfg.snapshot_flush_every
                and snapshot_dir is not None
            ):
                persistence.flush_snapshots(snapshot_dir, snapshot_buffer)
                snapshot_buffer.clear()

        history.append(
            dict(
                it=it, x_next=x_next.copy(), level_next=level_next, y_next=y_next,
                best_x=x_best.copy(), best_y=y_best,
                cost_next=cost_next, cumulative_cost=cumulative_cost,
                acq_time=acq_res.run_time, n_L=len(y_L), n_H=len(y_H),
                l_lf=gp.l_lf, sigma_lf=gp.sigma_lf,
                l_delta=gp.l_delta, sigma_delta=gp.sigma_delta, rho=gp.rho,
            )
        )

        if logger is not None:
            log_update(logger, state)

    if trace is not None and trace_path is not None:
        trace.save(trace_path)

    if snapshot_buffer is not None and snapshot_dir is not None and len(snapshot_buffer) > 0:
        persistence.flush_snapshots(snapshot_dir, snapshot_buffer)

    if save_cfg.plt_hist_enabled:
        plt_hist(history, save_cfg)

    if save_cfg.plt_conv_enabled:
        plt_conv(history, save_cfg)

    best_idx = int(np.argmin(y_H))
    best_x = X_H[best_idx].copy()
    best_y = float(y_H[best_idx])

    return MFBOResult(
        X_L=X_L, y_L=y_L, X_H=X_H, y_H=y_H,
        best_x=best_x, best_y=best_y,
        history=history, gp=gp, states=states,
        trace=trace, snapshots=snapshot_buffer,
    )
