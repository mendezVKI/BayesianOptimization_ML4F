"""
Bayesian Optimization core: config containers, GP model, acquisition
functions, gradient-based local refinement, and the main BO driver.

Within the research of the Machine Learning for Fluid Systems group (ML4F)

link: https://www.mendezma.com/

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

Array = np.ndarray  # short cut for numpy arrays
# short cut for domain bounds as a list
Bounds = List[Tuple[float, float]]  # [(l1,u1), ..., (ld,ud)]


#%%  Settings / Config containers

# these are dataclasses that makes it simple to access all the settings.

# general settigns of the GP (hyperparameters etc)
@dataclass
class GPConfig:
    # if optimize_hyperparams=False, these values are used and kept fixed
    l_c: float = 0.2          # length-scale ell
    sigma_f: float = 1.0      # kernel amplitude std (so variance is sigma_f^2)
    sigma_y: float = 0.1     # observation noise std
    jitter: float = 1e-10     # numerical stabilizer on diagonal

    # hyperparameter optimization (HPO) options
    optimize_hyperparams: bool = False
    hpo_every: int = 1        # optimize every k BO iterations (1 = every iteration)

    # Optional initial guess and bounds in log-space
    # theta_log = log([l_c, sigma_f, sigma_y])
    theta0_log: Optional[Array] = None
    theta_bounds_log: Optional[List[Tuple[float, float]]] = None

    # Rank-1 update to reduced computation complexity
    rank_one: bool = True
    rank_one_threshold: int=1000  # threshold size of the matrix to inverse when to use the rank-1 update

# general settigns of the expeted improvement
@dataclass
class AcqConfig:
    kind: str = "EI"  # "EI", "UCB", "PI"
    xi: float = 0.01
    kappa: float = 2.0
    maximize: bool = False  # typical BO for minimization -> False

# general setting for the Acquisition function optimization
@dataclass
class OptimConfig:
    method: str = "random"
    global_method: str = "random"

    optimizer: str = "L-BFGS-B"
    n_raw_samples: int = 2000
    grid_n_per_dim: int = 20
    n_restarts: int = 10
    local_maxiter: int = 200

    # Diverse batch acquisition: propose n_candidates points per iteration by
    # greedily penalizing candidates close to already-selected points.
    n_candidates: int = 1
    batch_distance_scale: float = 0.05  # relative to normalized search box

# Optional local refinement of each acquisition proposal using the true
# objective gradient (e.g. adjoint/AD gradients), via projected ADAM.
@dataclass
class GradientRefinementConfig:
    enabled: bool = False
    n_steps: int = 50
    learning_rate: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    distance_threshold: float = 1e-3  # normalized parameter-space distance
    close_pair_policy: str = "final"  # "final" or "midpoint"

# general settigns for the BO optimizer (optimization etc)
@dataclass
class BOConfig:
    # --- Pure opti related parameters
    n_init: int = 5    # size of the inital data set
    n_iter: int = 20   # number of iteration
    # How we sample the initial points
    init_sampling: str = "lhs"  # "uniform" or "lhs"
    random_state: Optional[int] = None


@dataclass
class SaveConfig:
    # --- Saving related parameters
    out_path: Optional[str] = None # path to save everything
    create_timestamp: bool = True
    # Logging (text)
    log_enabled: bool = True
    log_filename: str = "log.log"

    # --- Tier 1: lightweight per-iteration trace (numpy .npz)
    # Always cheap: scalars/small (d,) vectors only, no model objects.
    # Recorded in memory regardless of `out_path`; written to disk whenever
    # `out_path` is set (so default SaveConfig() = Tier 1 only, no I/O).
    trace_enabled: bool = True
    trace_filename: str = "trace.npz"
    trace_flush_every: Optional[int] = None  # periodically re-write trace.npz to disk (crash safety); None = write once at the end

    # --- Tier 2: full posterior-reconstruction snapshots (opt-in, heavier)
    # Stores (X, y, hyperparameters, kernel id, normalization state) needed to
    # refit the GP exactly -- never dense covariance/Cholesky factors.
    snapshot_enabled: bool = False
    snapshot_every: int = 1          # record a snapshot every k BO iterations
    snapshot_dirname: str = "snapshots"
    snapshot_flush_every: Optional[int] = None  # write buffered snapshots to disk and clear them from memory every k snapshots; None = keep all in memory, write once at the end

    # --- Plotting related parameters
    # enable plot of state: depends on the dimension (1D or 2D)
    plt_state_enabled: bool = False
    # enable plt of the conv of the best values
    plt_conv_enabled: bool = False
    plt_hist_enabled: bool = False         # enable plt of the BO exploration
    plt_MLE_conv_enbable: bool = False     # enable hyperparameter conv plt
    plt_all: bool = False                  # master switch
    plot_every: int = 1                    # plot every k iterations
    dpi: int = 250


# ----------------------------
# GP model container
# ----------------------------
# this stores exactly the quantities you derive in the lecture:
# (training points + Cholesky factor + alpha + hyperparameters)

@dataclass
class NormalizationHelper:
    x_lo: Array
    x_hi: Array
    y_mean: float = 0.0
    y_std: float = 1.0
    eps: float = 1e-12

    def normalize_X(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        denom = np.maximum(self.x_hi - self.x_lo, self.eps)
        return (X - self.x_lo) / denom

    def denormalize_X(self, Xn: Array) -> Array:
        Xn = np.asarray(Xn, dtype=float)
        return self.x_lo + Xn * (self.x_hi - self.x_lo)

    def fit_y(self, y: Array) -> None:
        y = np.asarray(y, dtype=float).reshape(-1)
        self.y_mean = float(np.mean(y))
        y_std = float(np.std(y))
        self.y_std = y_std if y_std > self.eps else 1.0

    def normalize_y(self, y: Array) -> Array:
        y = np.asarray(y, dtype=float)
        return (y - self.y_mean) / self.y_std

    def denormalize_y(self, y_norm: Array) -> Array:
        y_norm = np.asarray(y_norm, dtype=float)
        return self.y_mean + self.y_std * y_norm

    def denormalize_var(self, var_norm: Array) -> Array:
        var_norm = np.asarray(var_norm, dtype=float)
        return (self.y_std ** 2) * var_norm

@dataclass
class GPModel:
    l_c: float
    sigma_f: float
    sigma_y: float
    jitter: float

    Xs: Optional[Array] = None   # Xs is for X_star -> training data
    ys: Optional[Array] = None
    alpha: Optional[Array] = None
    L: Optional[Array] = None

    theta_log: Optional[Array] = None

    # Normalized training data
    Xs_norm: Optional[Array] = None
    ys_norm: Optional[Array] = None
    normalizer: Optional[NormalizationHelper] = None


# ----------------------------
# State / Results containers
# ----------------------------
# collect the current state of the BO and use it to checkpoint data while running
@dataclass
class BOState:
    it: int
    X: Array
    y: Array
    gp: GPModel
    acq: Any
    acq_res: AcqOptimizationResult
    x_next: Optional[Array] = None
    y_next: Optional[float] = None
    y_best: Optional[float] = None

    # Batch acquisition (OptimConfig.n_candidates) + gradient refinement
    x_proposed: Optional[Array] = None              # (n_candidates, d) acquisition proposals
    y_proposed: Optional[Array] = None               # (n_candidates,) f evaluated at proposals
    x_refined: Optional[Array] = None                 # (n_candidates, d) ADAM-refined points (NaN if not refined/failed)
    y_refined: Optional[Array] = None                  # (n_candidates,) f evaluated at refined points
    refinement_displacement: Optional[Array] = None     # (n_candidates,) normalized |x_refined - x_proposed|
    n_added: Optional[int] = None                        # number of (X,y) rows actually appended this iteration

# collects all the info concerning the result of the BO
@dataclass
class BOResult:
    X: Array
    y: Array
    best_x: Array
    best_y: float
    history: List[Dict[str, Any]]
    gp: GPModel
    states: List[BOState]

    # Tier 1 persistence: always populated when save_cfg.trace_enabled (default).
    trace: Optional["TraceLog"] = None
    # Tier 2 persistence: only allocated when save_cfg.snapshot_enabled; holds
    # whatever snapshots have not yet been flushed+cleared (all of them, if
    # snapshot_flush_every was never triggered). None if Tier 2 was disabled.
    snapshots: Optional[List["GPSnapshot"]] = None


# collects all the information concerning the evaluation of
# the next point to sample
@dataclass
class AcqOptimizationResult:
    x_next: Array          # (d,)
    a_best: Array          # (d,)
    Xcand: Array           # (N, d)
    a: Array               # (N,)
    run_time: float



#%% Helper utilities (skeleton)

# this is for setting the seed (if one wants reproducible results)
def _rng(random_state: Optional[int]) -> np.random.Generator:
    """Return a numpy Generator for reproducibility."""
    return np.random.default_rng(random_state)


# this is to define the random initialization (initial points)
def sample_initial_points(bounds: Bounds, n_init: int, rng: np.random.Generator, method: str = "uniform") -> Array:
    """
    INPUTS:
      bounds: list of (lo, hi)
      n_init: number of initial points
      rng: numpy generator
      method: "uniform" or "lhs"
    OUTPUT:
      X0: (n_init, d) array within bounds
    """
    d = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    if method == "uniform":
        X_unit = rng.random((n_init, d))  # in [0,1]^d

    elif method == "lhs":
        sampler = LatinHypercube(d=d, seed=rng)
        X_unit = sampler.random(n=n_init)

    else:
        raise ValueError(f"Unknown sampling method '{method}'. Use 'uniform' or 'lhs'.")

    # affine map from [0,1]^d to the physical bounds
    X0 = lo + X_unit * (hi - lo)
    return X0


# this is to evaluate the objective function. That could
# be run in parallel if the function is expensive. For the moment only
# running it serial.
def evaluate_objective(f: Callable[[Array], Union[float, Array]], X: Array) -> Array:
    """
    Evaluate objective on a batch of points.

    INPUTS:
      f: callable mapping x (d,) -> scalar (or (1,))
      X: (n, d)
    OUTPUT:
      y: (n,)
    """
    y_list = []
    for i in range(X.shape[0]):
        val = f(X[i])
        val = float(np.asarray(val).reshape(-1)[0])
        y_list.append(val)
    return np.asarray(y_list, dtype=float)


# this is to initialize (X,y) either from user-provided points, or by sampling
def init_dataset(
    f: Callable[[Array], Union[float, Array]],
    bounds: Bounds,
    bo_cfg: BOConfig,
    rng: np.random.Generator,
    X_init: Optional[Array] = None,
    y_init: Optional[Array] = None,
    top_up_to_n_init: bool = True,
) -> Tuple[Array, Array]:
    """
    Initialize the BO dataset (X,y) from either:
      - user-provided initial evaluations, or
      - sampled initial points + evaluation.

    Cases
    -----
    1) X_init is None:
         sample bo_cfg.n_init points, evaluate f
    2) X_init provided, y_init is None:
         evaluate f on X_init
    3) X_init and y_init provided:
         use as-is (after shape checks)

    Optional:
      if top_up_to_n_init=True, and the user provides fewer than n_init points,
      we sample extra points to reach n_init.

    Returns
    -------
    X : (n0,d)
    y : (n0,)
    """
    # Case 1: nothing provided -> sample initial points
    if X_init is None:
        X = sample_initial_points(bounds, bo_cfg.n_init, rng, method=bo_cfg.init_sampling)
        y = evaluate_objective(f, X)
        return X, y

    # Case 2/3: user provided X_init
    X = np.asarray(X_init, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)  # allow passing a single point (d,)

    # dimension check: bounds define the dimension
    d_bounds = len(bounds)
    if X.shape[1] != d_bounds:
        raise ValueError(
            f"Dimension mismatch:\n"
            f"  bounds define dimension d = {d_bounds}\n"
            f"  X_init has shape {X.shape}\n"
        )

    # bounds check: all points must lie in the box
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    if np.any(X < lo) or np.any(X > hi):
        raise ValueError("Some initial points X_init lie outside the prescribed bounds.")

    # y_init handling
    if y_init is None:
        y = evaluate_objective(f, X)
    else:
        y = np.asarray(y_init, dtype=float).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"y_init length {y.shape[0]} must match X_init rows {X.shape[0]}")

    # optional top up with random points if not enough initial points
    if top_up_to_n_init:
        n_missing = max(0, bo_cfg.n_init - X.shape[0])
        if n_missing > 0:
            X_extra = sample_initial_points(bounds, n_missing, rng, method=bo_cfg.init_sampling)
            y_extra = evaluate_objective(f, X_extra)
            X = np.vstack([X, X_extra])
            y = np.concatenate([y, y_extra])

    return X, y


# ----------------------------
# GP kernel + fit / prediction (Cholesky reuse)
# ----------------------------

def rbf_kernel(X1, X2, gamma) -> Array:
    sqdist = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)
    return np.exp(- gamma * sqdist)

def rbf_kernel_amp(X1: Array, X2: Array, l_c: float, sigma_f: float) -> Array:
    """
    RBF kernel with amplitude:
      k(x,x') = sigma_f^2 * exp(-||x-x'||^2 / (2 l_c^2))
    """
    gamma = 0.5 / (l_c**2)
    return (sigma_f**2) * rbf_kernel(X1, X2, gamma=gamma)


def gp_fit(Xs, ys, l_c=0.3, sigma_f=1.0, sigma_y=0.1, jitter=1e-10):
    """
    Compute Cholesky factor and alpha for GP regression.
    """
    Xs = np.asarray(Xs, dtype=float)
    ys = np.asarray(ys, dtype=float).reshape(-1)

    Kss = rbf_kernel_amp(Xs, Xs, l_c=l_c, sigma_f=sigma_f)
    n = Xs.shape[0]
    Ky = Kss + (sigma_y**2 + jitter) * np.eye(n)

    L = cholesky(Ky, lower=True)
    alpha = cho_solve((L, True), ys)
    return alpha, L


def gp_predict(X, Xs, alpha, L, l_c=0.3, sigma_f=1.0, return_cov=False, gp: Optional[GPModel] = None):
    """
    GP prediction using precomputed Cholesky factor.

    If gp is provided and gp.normalizer is available:
      - input X is assumed in physical coordinates
      - prediction is returned in physical output scale
    """
    X = np.asarray(X, dtype=float)

    if gp is not None and gp.normalizer is not None:
        X_eval = gp.normalizer.normalize_X(X)
        X_train = gp.Xs_norm
    else:
        X_eval = X
        X_train = np.asarray(Xs, dtype=float)

    alpha = np.asarray(alpha, dtype=float).reshape(-1)

    Ks = rbf_kernel_amp(X_eval, X_train, l_c=l_c, sigma_f=sigma_f)
    mu_norm = Ks @ alpha

    V = cho_solve((L, True), Ks.T)

    if return_cov:
        Kxx = rbf_kernel_amp(X_eval, X_eval, l_c=l_c, sigma_f=sigma_f)
        cov_norm = Kxx - Ks @ V

        if gp is not None and gp.normalizer is not None:
            mu = gp.normalizer.denormalize_y(mu_norm)
            cov = (gp.normalizer.y_std ** 2) * cov_norm
            return mu, cov # unscaled output
        return mu_norm, cov_norm # scaled output

    else:
        Kxx_diag = (sigma_f**2) * np.ones(X_eval.shape[0])
        var_norm = Kxx_diag - np.sum(Ks * V.T, axis=1)
        var_norm = np.maximum(var_norm, 0.0)

        if gp is not None and gp.normalizer is not None:
            mu = gp.normalizer.denormalize_y(mu_norm)
            var = gp.normalizer.denormalize_var(var_norm)
            return mu, var

        return mu_norm, var_norm


# ----------------------------
# Hyperparameter optimization by log marginal likelihood
# ----------------------------

def negative_log_marginal_likelihood(
    theta_log: Array,
    X_train: Array,
    y_train: Array,
    jitter: float = 1e-10,
    MLE_hist: Optional[list] = None,
    param_hist: Optional[list] = None,
) -> float:
    """
    theta_log = log([l_c, sigma_f, sigma_y])
    returns NLL = -log p(y | X, theta)
    """
    l_c, sigma_f, sigma_y = np.exp(theta_log)

    K = rbf_kernel_amp(X_train, X_train, l_c=l_c, sigma_f=sigma_f)
    n = X_train.shape[0]
    Ky = K + (sigma_y**2 + jitter) * np.eye(n)

    try:
        L = cholesky(Ky, lower=True)
    except np.linalg.LinAlgError:
        return 1e12

    alpha = cho_solve((L, True), y_train)

    ll = -0.5 * (y_train @ alpha) - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2.0 * np.pi)

    if MLE_hist is not None:
        MLE_hist.append(float(ll))
    if param_hist is not None:
        param_hist.append([float(l_c), float(sigma_f), float(sigma_y)])

    return -float(ll)


def optimize_gp_hyperparams(
    X: Array,
    y: Array,
    gp: GPModel,
    gp_cfg: GPConfig,
    MLE_hist: Optional[list] = None,
    param_hist: Optional[list] = None,
) -> GPModel:
    """
    Update gp.(l_c, sigma_f, sigma_y) by minimizing negative log marginal likelihood.
    """
    y = np.asarray(y, dtype=float).reshape(-1)

    # initial guess in log-space
    if gp_cfg.theta0_log is not None:
        x0 = np.asarray(gp_cfg.theta0_log, dtype=float).reshape(-1)
    else:
        x0 = np.log([gp.l_c, gp.sigma_f, gp.sigma_y])

    bounds = gp_cfg.theta_bounds_log

    res = minimize(
        negative_log_marginal_likelihood,
        x0=x0,
        args=(X, y, gp.jitter, MLE_hist, param_hist),
        method="L-BFGS-B",
        bounds=bounds,
    )

    theta_opt_log = res.x
    l_c, sigma_f, sigma_y = np.exp(theta_opt_log)

    gp.l_c = float(l_c)
    gp.sigma_f = float(sigma_f)
    gp.sigma_y = float(sigma_y)
    gp.theta_log = theta_opt_log.copy()

    return gp


# ----------------------------
# GP wrapper used by BO
# ----------------------------

def build_gp_model(gp_cfg: GPConfig, bounds: Bounds) -> GPModel:
    """Create an unfitted GP container."""
    x_lo = np.array([b[0] for b in bounds], dtype=float)
    x_hi = np.array([b[1] for b in bounds], dtype=float)

    normalizer = NormalizationHelper(
        x_lo=x_lo,
        x_hi=x_hi,
    )

    return GPModel(
        l_c=gp_cfg.l_c,
        sigma_f=gp_cfg.sigma_f,
        sigma_y=gp_cfg.sigma_y,
        jitter=gp_cfg.jitter,
        normalizer=normalizer,
    )


def fit_gp(gp: GPModel, X: Array, y: Array, gp_cfg: GPConfig, it: int) -> GPModel:

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if gp.normalizer is None:
        raise ValueError("GP normalizer is not initialized.")

    # number of points already fitted before this call (needed below to know
    # how many NEW rows were appended, since batch acquisition / gradient
    # refinement can add more than one point per BO iteration)
    n_prev = 0 if gp.Xs is None else gp.Xs.shape[0]

    # store physical data
    gp.Xs = X
    gp.ys = y

    # normalize X and y
    gp.Xs_norm = gp.normalizer.normalize_X(X)
    gp.normalizer.fit_y(y)
    gp.ys_norm = gp.normalizer.normalize_y(y)

    # Decide whether to run hyperparameter optimization
    do_hpo = (
        gp_cfg.optimize_hyperparams
        and (gp_cfg.hpo_every > 0)
        and (it % gp_cfg.hpo_every == 0)
    )

    if do_hpo:
        gp = optimize_gp_hyperparams(gp.Xs_norm, gp.ys_norm, gp, gp_cfg)

    # Decide whether we can use rank-one update
    n_new = X.shape[0] - n_prev
    can_rank1 = (
        gp_cfg.rank_one
        and not do_hpo
        and gp.L is not None
        and n_prev >= 1
        and n_new >= 1
        and X.shape[0] > gp_cfg.rank_one_threshold
    )

    if can_rank1:
        # Extend the Cholesky factor one row at a time. A single BO round can
        # append more than one point (batch acquisition / gradient
        # refinement), so each new row must see only the rows fitted so far.
        for j in range(n_prev, X.shape[0]):
            gp = _rank_one_update(
                gp,
                x_new=gp.Xs_norm[j],
                y_new=gp.ys_norm[j],
                X_old=gp.Xs_norm[:j],
                y_old=gp.ys_norm[:j],
            )
    else:
        gp.alpha, gp.L = gp_fit(
            gp.Xs_norm,
            gp.ys_norm,
            gp.l_c,
            gp.sigma_f,
            gp.sigma_y,
            gp.jitter,
        )

    return gp


def _rank_one_update(
    gp: GPModel, x_new: Array, y_new: float, X_old: Array, y_old: Array
) -> GPModel:
    """Extend gp.L / gp.alpha by one row, given the exact training set (X_old,
    y_old) that gp.L was built on."""

    L_old = gp.L

    # compute cross kernel vector
    k = rbf_kernel_amp(
        X_old,
        x_new.reshape(1, -1),
        l_c=gp.l_c,
        sigma_f=gp.sigma_f
    ).reshape(-1)

    # diagonal term
    k_nn = rbf_kernel_amp(
        x_new.reshape(1, -1),
        x_new.reshape(1, -1),
        l_c=gp.l_c,
        sigma_f=gp.sigma_f
    )[0, 0] + gp.sigma_y**2 + gp.jitter

    v = solve_triangular(L_old, k, lower=True)

    val = k_nn - np.dot(v, v)
    val = max(val, 1e-14)
    diag_new = np.sqrt(val)

    n = L_old.shape[0]
    L_new = np.zeros((n + 1, n + 1))
    L_new[:n, :n] = L_old
    L_new[n, :n] = v
    L_new[n, n] = diag_new

    y_new_vec = np.concatenate([y_old, [y_new]])

    z = solve_triangular(L_new, y_new_vec, lower=True)
    alpha_new = solve_triangular(L_new.T, z, lower=False)

    gp.L = L_new
    gp.alpha = alpha_new

    return gp


# ----------------------------
# Acquisition + optimizer
# ----------------------------
def make_acquisition(acq_cfg: AcqConfig, gp: GPModel, y_best: float):
    """
    Build acquisition function a(x).

    Convention:
      - optimize_acquisition will MAXIMIZE a(x)
      - for minimization problems:
          EI(x) = E[max(0, y_best - f(x) - xi)]
          PI(x) = P(f(x) <= y_best - xi)
          UCB(x) uses LCB: -(mu - kappa*sigma) so that maximizing picks low values
    """

    def acq(Xcand: Array) -> Array:
        Xcand = np.asarray(Xcand, dtype=float)
        if Xcand.ndim == 1:
            Xcand = Xcand.reshape(1, -1)

        # GP predictive mean and variance (latent function)
        mu, var = gp_predict(
            Xcand, gp.Xs, gp.alpha, gp.L,
            l_c=gp.l_c, sigma_f=gp.sigma_f,
            return_cov=False,
            gp=gp
        )

        var = np.maximum(var, 0.0)          # numerical safety
        sigma = np.sqrt(var)                # (m,)

        # avoid divide-by-zero in EI/PI
        eps = 1e-12
        sigma_safe = np.maximum(sigma, eps)

        kind = acq_cfg.kind.upper()

        # ----- Minimization vs Maximization handling
        # We want to write "improvement" in a consistent way.
        if not acq_cfg.maximize:
            # Minimization: improvement happens when f is smaller than current best
            # EI/PI use (y_best - mu - xi)
            improvement = (y_best - mu - acq_cfg.xi)
        else:
            # Maximization: improvement when f is larger than current best
            # EI/PI use (mu - y_best - xi)
            improvement = (mu - y_best - acq_cfg.xi)

        if kind == "EI":
            Z = improvement / sigma_safe
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            # If sigma==0, EI should be 0
            ei = np.where(sigma > eps, ei, 0.0)
            return ei

        elif kind == "PI":
            Z = improvement / sigma_safe
            pi = norm.cdf(Z)
            return pi

        elif kind == "UCB":
            # UCB is usually for maximization: mu + kappa*sigma
            # For minimization, we often use LCB: mu - kappa*sigma and MINIMIZE it.
            # But our optimizer will MAXIMIZE acquisition, so we flip the sign for minimization.
            if not acq_cfg.maximize:
                # maximize -(mu - kappa*sigma)  <=> minimize (mu - kappa*sigma)
                return -(mu - acq_cfg.kappa * sigma)
            else:
                # maximize (mu + kappa*sigma)
                return (mu + acq_cfg.kappa * sigma)

        else:
            raise ValueError(f"Unknown acquisition kind '{acq_cfg.kind}'. Use 'EI', 'PI', or 'UCB'.")

    return acq


def _make_random_candidates(bounds: Bounds, N: int, rng: np.random.Generator) -> Array:
    d = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo + (hi - lo) * rng.random((N, d))


def _make_grid_candidates(bounds: Bounds, n_per_dim: int) -> Array:
    # WARNING: grid grows exponentially with d (n_per_dim^d).
    grids_1d = [np.linspace(b[0], b[1], n_per_dim) for b in bounds]
    mesh = np.meshgrid(*grids_1d, indexing="xy")
    X = np.stack([m.ravel() for m in mesh], axis=1)
    return X


def optimize_acquisition(
    acq: Callable[[Array], Array],
    bounds: Bounds,
    optim_cfg: OptimConfig,
    rng: np.random.Generator,
) -> Array:
    """
    Optimize acquisition to propose next x.

    Supported modes:
      - method="random": global random candidates only
      - method="grid":   global grid candidates only
      - method="refined": global (random or grid) + local L-BFGS-B refinement

    OUTPUT:
      x_next: (d,) proposed point
    """
    method = optim_cfg.method.lower()
    if method not in ["random", "grid", "refined"]:
        raise ValueError("optim_cfg.method must be 'random', 'grid', or 'refined'.")

    # ---- 1) build the candidate set for the GLOBAL step

    # start the time, only if required
    start_time = time.perf_counter()

    if method == "random":
        Xcand = _make_random_candidates(bounds, optim_cfg.n_raw_samples, rng)

    elif method == "grid":
        Xcand = _make_grid_candidates(bounds, optim_cfg.grid_n_per_dim)

    else:
        # refined: first choose the global method
        gmethod = optim_cfg.global_method.lower()
        if gmethod == "random":
            Xcand = _make_random_candidates(bounds, optim_cfg.n_raw_samples, rng)
        elif gmethod == "grid":
            Xcand = _make_grid_candidates(bounds, optim_cfg.grid_n_per_dim)
        else:
            raise ValueError("optim_cfg.global_method must be 'random' or 'grid' when method='refined'.")

    # ---- 2) evaluate acquisition on all candidates (VECTORISED)
    a = np.asarray(acq(Xcand), dtype=float).reshape(-1)
    a[~np.isfinite(a)] = -np.inf

    # best point from global scan
    best_idx = int(np.argmax(a))
    x_best = Xcand[best_idx].copy()
    a_best = float(a[best_idx])

    # If we are not refining, stop here
    if method in ["random", "grid"]:
        run_time = time.perf_counter() - start_time
        return AcqOptimizationResult(
        x_next=x_best,
        a_best=a_best,
        Xcand=Xcand,
        a=a,
        run_time = run_time
    )

    # ---- 3) refined: local improvement from the best few candidates
    n_starts = min(optim_cfg.n_restarts, Xcand.shape[0])

    # pick indices of the top n_starts values (fast without sorting everything)
    top_idx = np.argpartition(-a, n_starts - 1)[:n_starts]
    Xstarts = Xcand[top_idx]

    # SciPy bounds format
    scipy_bounds = [(float(b[0]), float(b[1])) for b in bounds]

    # Objective for minimizer: minimize -acq(x)
    def obj(x: Array) -> float:
        x2 = np.asarray(x, dtype=float).reshape(1, -1)
        val = float(np.asarray(acq(x2)).reshape(-1)[0])
        if not np.isfinite(val):
            return 1e30
        return -val

    for x0 in Xstarts:
        res = minimize(
            obj,
            x0=x0,
            method=optim_cfg.optimizer,
            bounds=scipy_bounds,
            options=dict(maxiter=optim_cfg.local_maxiter),
        )
        if res.success and res.x is not None:
            x_try = np.asarray(res.x, dtype=float)
            a_try = -float(res.fun)
            if np.isfinite(a_try) and a_try > a_best:
                a_best = a_try
                x_best = x_try.copy()

    # Stop the timer
    run_time = time.perf_counter() - start_time

    return AcqOptimizationResult(
        x_next=x_best,
        a_best=a_best,
        Xcand=Xcand,
        a=a,
        run_time=run_time
    )


def optimize_acquisition_batch(
    acq: Callable[[Array], Array],
    bounds: Bounds,
    optim_cfg: OptimConfig,
    rng: np.random.Generator,
) -> List[AcqOptimizationResult]:
    """
    Propose optim_cfg.n_candidates diverse points per iteration.

    Greedily runs optimize_acquisition n_candidates times; after each pick,
    already-selected points are penalized (Gaussian repulsion in normalized
    parameter space, scale = optim_cfg.batch_distance_scale) so that later
    picks favor other high-acquisition regions rather than clustering near
    previous ones.

    With n_candidates=1 this reduces exactly to [optimize_acquisition(acq, ...)].
    """
    n_candidates = int(optim_cfg.n_candidates)
    if n_candidates < 1:
        raise ValueError("optim_cfg.n_candidates must be at least one.")
    if optim_cfg.batch_distance_scale <= 0.0:
        raise ValueError("optim_cfg.batch_distance_scale must be positive.")

    lo = np.asarray([b[0] for b in bounds], dtype=float)
    width = np.asarray([b[1] - b[0] for b in bounds], dtype=float)
    selected: List[Array] = []
    results: List[AcqOptimizationResult] = []

    for _ in range(n_candidates):
        def penalized(Xcand: Array) -> Array:
            values = np.asarray(acq(Xcand), dtype=float).reshape(-1)
            if not selected:
                return values
            normalized = (np.atleast_2d(Xcand) - lo) / width
            selected_normalized = (np.asarray(selected) - lo) / width
            distance = np.min(
                np.linalg.norm(
                    normalized[:, None, :] - selected_normalized[None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            penalty = 1.0 - np.exp(
                -(distance / optim_cfg.batch_distance_scale) ** 2
            )
            return values * penalty

        result = optimize_acquisition(penalized, bounds, optim_cfg, rng)
        selected.append(np.asarray(result.x_next, dtype=float))
        results.append(result)

    return results


def adam_refine_candidate(
    x0: Array,
    gradient: Callable[[Array], Array],
    bounds: Bounds,
    config: GradientRefinementConfig,
) -> Array:
    """
    Locally refine a single candidate by projected ADAM gradient descent, in
    normalized [0,1]^d parameter space (so a single learning_rate is
    meaningful across dimensions with different physical scales).

    ``gradient`` must return the gradient of the (minimization) objective at
    a point in physical coordinates; it may also return a tuple whose last
    element is the gradient (e.g. (value, grad)), matching common
    autodiff/adjoint solver conventions.
    """
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    width = np.asarray([b[1] - b[0] for b in bounds], dtype=float)
    z = np.clip((np.asarray(x0, dtype=float) - lo) / width, 0.0, 1.0)
    first = np.zeros_like(z)
    second = np.zeros_like(z)

    for step in range(1, int(config.n_steps) + 1):
        raw = gradient(lo + width * z)
        if isinstance(raw, tuple):
            raw = raw[-1]
        grad_z = np.asarray(raw, dtype=float).reshape(z.shape) * width
        if not np.all(np.isfinite(grad_z)):
            break
        first = config.beta1 * first + (1.0 - config.beta1) * grad_z
        second = config.beta2 * second + (1.0 - config.beta2) * grad_z**2
        first_hat = first / (1.0 - config.beta1**step)
        second_hat = second / (1.0 - config.beta2**step)
        z = np.clip(
            z - config.learning_rate * first_hat / (np.sqrt(second_hat) + config.epsilon),
            0.0,
            1.0,
        )

    return lo + width * z


#%% Main BO function

def bayesian_optimization(
    f: Callable[[Array], Union[float, Array]],
    bounds: Bounds,
    bo_cfg: BOConfig,
    gp_cfg: GPConfig,
    acq_cfg: AcqConfig,
    optim_cfg: OptimConfig,
    save_cfg: SaveConfig,
    X_init: Optional[Array] = None,  # initial points (optional)
    y_init: Optional[Array] = None,  # initial values (optional)
    top_up_to_n_init: bool = True,   # if X_init has too few points, add more random ones
    states: Optional[List[BOState]] = None,
    f_true: Optional[Callable[[Array], Array]] = None, # in case its available
    gradient: Optional[Callable[[Array], Array]] = None,
    refinement_cfg: Optional[GradientRefinementConfig] = None,
) -> BOResult:
    """
    Bayesian Optimization driver (minimization by default).

    Each iteration proposes optim_cfg.n_candidates diverse points (batch
    acquisition; n_candidates=1 reproduces the original single-point
    behavior). If refinement_cfg.enabled, each proposal is additionally
    refined by local ADAM gradient descent using the user-supplied
    ``gradient`` of the true objective, and both the proposal and its
    refined endpoint are evaluated (merged into a single point when they
    land too close together, per refinement_cfg.close_pair_policy).
    """
    from .plotting import setup_plotting_toggle, plt_state, plt_hist, plt_conv
    from . import persistence

    rng = _rng(bo_cfg.random_state)

    if save_cfg.trace_flush_every is not None and save_cfg.out_path is None:
        raise ValueError("save_cfg.trace_flush_every requires save_cfg.out_path to be set.")
    if save_cfg.snapshot_flush_every is not None and save_cfg.out_path is None:
        raise ValueError("save_cfg.snapshot_flush_every requires save_cfg.out_path to be set.")

    refinement_cfg = refinement_cfg or GradientRefinementConfig()
    if refinement_cfg.enabled and gradient is None:
        raise ValueError("gradient must be provided when gradient refinement is enabled.")
    if refinement_cfg.close_pair_policy not in {"final", "midpoint"}:
        raise ValueError("refinement_cfg.close_pair_policy must be 'final' or 'midpoint'.")
    if refinement_cfg.distance_threshold < 0.0:
        raise ValueError("refinement_cfg.distance_threshold must be non-negative.")
    if refinement_cfg.n_steps < 0:
        raise ValueError("refinement_cfg.n_steps must be non-negative.")
    if refinement_cfg.learning_rate <= 0.0:
        raise ValueError("refinement_cfg.learning_rate must be positive.")
    if not (0.0 <= refinement_cfg.beta1 < 1.0):
        raise ValueError("refinement_cfg.beta1 must lie in [0, 1).")
    if not (0.0 <= refinement_cfg.beta2 < 1.0):
        raise ValueError("refinement_cfg.beta2 must lie in [0, 1).")

    # ---- 0) Initialization
    if states is None:
        states = []

    # Define the main output path given the time and hour
    experiment_path = setup_experiment_folder(save_cfg)

    # Define the main output path given the time and hour
    setup_plotting_toggle(save_cfg)

    # Initialize the logging
    logger = None
    if save_cfg.log_enabled and experiment_path is not None:
        log_path = os.path.join(experiment_path, save_cfg.log_filename)
        logger = setup_logger(log_path)
        logger.info(f"Bounds: {bounds}")
        logger.info(f"BOConfig: {bo_cfg}")
        logger.info(f"GPConfig: {gp_cfg}")
        logger.info(f"AcqConfig: {acq_cfg}")
        logger.info(f"OptimConfig: {optim_cfg}")
        logger.info(f"GradientRefinementConfig: {refinement_cfg}")
        logger.info(f"SaveConfig: {save_cfg}")
        logger.info(f"Random seed: {bo_cfg.random_state}")
        logger.info(
            "it  |    x_next     |      y_next     |      best_y     |    l_c    |  sigma_f  |  sigma_y  | batch"
        )
        logger.info("-" * 105)

    # History saving list
    history: List[Dict[str, Any]] = []

    # ---- Persistence setup (Tier 1 trace always on; Tier 2 snapshots opt-in)
    trace = persistence.TraceLog() if save_cfg.trace_enabled else None
    trace_path = (
        os.path.join(experiment_path, save_cfg.trace_filename)
        if (experiment_path is not None and trace is not None)
        else None
    )

    # snapshot_buffer stays exactly `None` (never a list) when Tier 2 is
    # disabled, so no per-iteration (X, y) copy is ever allocated.
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
            random_state=bo_cfg.random_state,
            bo_cfg=bo_cfg,
            gp_cfg=gp_cfg,
            acq_cfg=acq_cfg,
            optim_cfg=optim_cfg,
            save_cfg=save_cfg,
        )

    # ---- 1) initial points (either provided by user or sampled)
    X, y = init_dataset(
        f=f,
        bounds=bounds,
        bo_cfg=bo_cfg,
        rng=rng,
        X_init=X_init,
        y_init=y_init,
        top_up_to_n_init=top_up_to_n_init
    )

    # ---- 2) build GP object
    gp = build_gp_model(gp_cfg, bounds)

    width = np.asarray([b[1] - b[0] for b in bounds], dtype=float)

    # ---- 3) BO loop
    for it in range(bo_cfg.n_iter):
        iter_start_time = time.perf_counter()

        # Fit/update GP on current data
        # NOTE: we pass it so that fit_gp can decide if hyperparams are optimized now
        gp = fit_gp(gp, X, y, gp_cfg, it)

        # Decide current best (minimization)
        best_idx = int(np.argmin(y))
        y_best = float(y[best_idx])
        x_best = X[best_idx].copy()

        # Build acquisition
        acq = make_acquisition(acq_cfg, gp, y_best)

        # Propose a (possibly diverse) batch of candidates. n_candidates=1
        # reproduces the original single-EI-point behavior.
        acq_results = optimize_acquisition_batch(acq, bounds, optim_cfg, rng)
        acq_res = acq_results[0]  # used for plotting / logging (first proposal)

        x_proposed = np.asarray([r.x_next for r in acq_results], dtype=float)
        y_proposed = np.asarray(
            [float(np.asarray(f(x)).reshape(-1)[0]) for x in x_proposed],
            dtype=float,
        )

        # Plot the current state if required AND that we are in 1D/2D case
        if save_cfg.plt_state_enabled and len(bounds) < 3 and (it % save_cfg.plot_every == 0):
            plt_state(
                gp=gp,
                acq_res=acq_res,
                f=f,
                bounds=bounds,
                it=it,
                x_best=x_best,
                y_best=y_best,
                save_cfg=save_cfg,
                acq_cfg=acq_cfg,
                bo_cfg=bo_cfg,
                acq=acq,
                f_true=f_true
            )

        # Optionally refine each proposal locally via ADAM on the true
        # objective gradient, then decide (per candidate) whether to keep
        # both the proposal and its refined endpoint, or merge them.
        x_refined = np.full_like(x_proposed, np.nan)
        y_refined = np.full(len(x_proposed), np.nan, dtype=float)
        displacements = np.zeros(len(x_proposed), dtype=float)
        round_X: List[Array] = []
        round_y: List[float] = []

        for q, (x0, y0) in enumerate(zip(x_proposed, y_proposed)):
            if not refinement_cfg.enabled:
                round_X.append(x0)
                round_y.append(float(y0))
                continue

            local_gradient = gradient
            if acq_cfg.maximize:
                def local_gradient(x, grad=gradient):
                    value = grad(x)
                    raw_gradient = value[-1] if isinstance(value, tuple) else value
                    return -np.asarray(raw_gradient)

            xL = adam_refine_candidate(x0, local_gradient, bounds, refinement_cfg)
            yL = float(np.asarray(f(xL)).reshape(-1)[0])
            displacement = float(np.linalg.norm((xL - x0) / width))

            x_refined[q] = xL
            y_refined[q] = yL
            displacements[q] = displacement

            if displacement > refinement_cfg.distance_threshold:
                round_X.extend([x0, xL])
                round_y.extend([float(y0), yL])
            elif refinement_cfg.close_pair_policy == "midpoint":
                midpoint = 0.5 * (x0 + xL)
                midpoint_value = float(np.asarray(f(midpoint)).reshape(-1)[0])
                round_X.append(midpoint)
                round_y.append(midpoint_value)
            else:
                round_X.append(xL)
                round_y.append(yL)

        round_X_array = np.asarray(round_X, dtype=float)
        round_y_array = np.asarray(round_y, dtype=float)

        # x_next / y_next: kept for backward compatibility -- the primary
        # (first) proposal of this round.
        x_next = x_proposed[0]
        y_next = float(y_proposed[0])

        # Append all non-redundant proposed/refined observations
        X = np.vstack([X, round_X_array])
        y = np.concatenate([y, round_y_array])

        # Store all what was evaluated at this iteration in the state
        state = BOState(
            it=it,
            X=X.copy(),
            y=y.copy(),
            gp=copy.deepcopy(gp),  # gp is mutated in-place each iteration; freeze this state's copy
            acq=acq,
            acq_res=acq_res,
            x_next=x_next,
            y_next=y_next,
            y_best=y_best,
            x_proposed=x_proposed.copy(),
            y_proposed=y_proposed.copy(),
            x_refined=x_refined.copy(),
            y_refined=y_refined.copy(),
            refinement_displacement=displacements.copy(),
            n_added=len(round_y_array),
        )

        # Add the current state in the save list
        states.append(state)

        iter_wall_time = time.perf_counter() - iter_start_time

        # ---- Tier 1: always-on lightweight trace
        if trace is not None:
            trace.append(
                it=it,
                x_next=x_next,
                y_next=y_next,
                x_best=x_best,
                y_best=y_best,
                wall_time=iter_wall_time,
                acq_value=acq_res.a_best,
            )
            if (
                save_cfg.trace_flush_every is not None
                and (it + 1) % save_cfg.trace_flush_every == 0
                and trace_path is not None
            ):
                trace.save(trace_path)

        # ---- Tier 2: opt-in full-posterior snapshot (never built unless enabled)
        if snapshot_buffer is not None and (it % save_cfg.snapshot_every == 0):
            snapshot_buffer.append(persistence.build_gp_snapshot(it, gp))
            if (
                save_cfg.snapshot_flush_every is not None
                and len(snapshot_buffer) >= save_cfg.snapshot_flush_every
                and snapshot_dir is not None
            ):
                persistence.flush_snapshots(snapshot_dir, snapshot_buffer)
                snapshot_buffer.clear()

        # Record lightweight history
        history.append(
            dict(
                it=it,
                x_next=x_next.copy(),
                y_next=y_next,
                x_proposed=x_proposed.copy(),
                y_proposed=y_proposed.copy(),
                x_refined=x_refined.copy(),
                y_refined=y_refined.copy(),
                refinement_displacement=displacements.copy(),
                n_added=len(round_y_array),
                best_x=x_best.copy(),
                best_y=y_best,
                acq_time=acq_res.run_time,
                n=len(y),
                l_c=gp.l_c,
                sigma_f=gp.sigma_f,
                sigma_y=gp.sigma_y,
            )
        )

        # Optinal log saving
        if logger is not None:
            log_update(logger, state)

    # ---- Final persistence flush
    if trace is not None and trace_path is not None:
        trace.save(trace_path)

    if snapshot_buffer is not None and snapshot_dir is not None and len(snapshot_buffer) > 0:
        persistence.flush_snapshots(snapshot_dir, snapshot_buffer)

    if save_cfg.plt_hist_enabled:
        plt_hist(history, save_cfg)

    if save_cfg.plt_conv_enabled:
        plt_conv(history, save_cfg)

    # ---- 4) final best
    best_idx = int(np.argmin(y))
    best_x = X[best_idx].copy()
    best_y = float(y[best_idx])


    return BOResult(
        X=X,
        y=y,
        best_x=best_x,
        best_y=best_y,
        history=history,
        gp=gp,
        states=states,
        trace=trace,
        snapshots=snapshot_buffer,
    )
