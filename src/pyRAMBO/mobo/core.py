"""
Multi-objective Bayesian Optimization core: config containers, a multi-output
intrinsic-coregionalization (ICM) GP, a Monte-Carlo Expected Hypervolume
Improvement (EHVI) acquisition, and the main MOBO driver.

Within the research of the Machine Learning for Fluid Systems group (ML4F)

link: https://www.mendezma.com/

Ported from an exploratory script (legacy/MultiOutputICMBO_legacy.py) that
packed the ICM GP, the Pareto/hypervolume utilities and the BO loop into a
single class. Only the mathematics (Kronecker ICM covariance, Pareto
dominance, EHVI by Monte Carlo) was kept; everything else was rebuilt from
scratch to match the structure of sbo (src/pyRAMBO/sbo/core.py) and mfbo
(src/pyRAMBO/mfbo/core.py). mobo is intentionally independent of both -- see the
package-level docstring in __init__.py.

Three deliberate departures from the legacy script are documented at their
definitions: the corrected 2D hypervolume (``pareto_front`` / ``hypervolume``),
the vectorized and *deterministic* (common-random-numbers) EHVI
(``make_acquisition``), and the removal of the redundant kernel amplitude
(``rbf_kernel_ard``).

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Tuple, Union, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .persistence import TraceLog, GPSnapshot

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
import copy
import os
import time

from .saving import setup_experiment_folder, setup_logger, log_summary

Array = np.ndarray
Bounds = List[Tuple[float, float]]  # [(l1,u1), ..., (ld,ud)]

# The hypervolume / EHVI machinery below is exact and closed-form for TWO
# objectives only. The ICM GP itself is written for an arbitrary number of
# outputs M, so extending mobo to M > 2 only requires a general-dimension
# hypervolume (e.g. the WFG or HSO recursions); that is left for a future
# revision. Everything that is 2D-only raises explicitly.
N_OBJ_SUPPORTED = 2


#%% Settings / Config containers

@dataclass
class GPConfig:
    """Hyperparameters of the multi-output ICM GP:

        Cov(f_i(x), f_j(x')) = B_ij * k(x, x')
        Cov(y_i(x), y_j(x))  = Cov(f_i(x), f_j(x)) + delta_ij * sigma_n_i^2

    with k an ARD RBF kernel of UNIT amplitude and B = L_B L_B^T the (M, M)
    positive semi-definite output covariance ("coregionalization") matrix.
    B carries all the signal variance, so B_ii is the marginal prior
    variance of output i and B_ij its prior covariance with output j.
    """

    # Normalization (like sbo's normalize_X / normalize_y). With True
    # (default) the GP works on X scaled to [0,1]^d by the BO bounds and on
    # each objective standardized to zero mean / unit std (per output), so the
    # hyperparameters and default_theta_bounds() are in those units;
    # predictions are returned in physical units either way. With False the GP
    # works directly on the raw data: then the hyperparameters and
    # theta_bounds must be given in the data's own units.
    normalize_X: bool = True
    normalize_y: bool = True

    # if optimize_hyperparams=False, these values are used and kept fixed
    length_scales: Union[float, Array] = 0.2  # ARD; a scalar is broadcast over d
    L_B_diag: Union[float, Array] = 1.0       # diagonal of the L_B factor of B
    L_B_offdiag: float = 0.0                  # off-diagonal entries of L_B
    sigma_n: Union[float, Array] = 0.05       # observation noise std, per output
    jitter: float = 1e-10                     # numerical stabilizer on diagonal

    # hyperparameter optimization (HPO) options
    optimize_hyperparams: bool = True
    hpo_every: int = 1  # optimize every k BO iterations (1 = every iteration)

    # Optional initial guess and bounds for the free parameters, packed as
    # [log(length_scales) (d), L_B params (M(M+1)/2), log(sigma_n) (M)].
    # The L_B block is the row-major lower triangle of L_B with its diagonal
    # in log-space (so B stays positive semi-definite for any theta) and its
    # off-diagonal entries left un-transformed -- they may be negative, which
    # is what lets the GP model anti-correlated (i.e. genuinely competing)
    # objectives. See pack_theta / unpack_theta.
    # When theta_bounds is None the optimizer uses default_theta_bounds().
    theta0: Optional[Array] = None
    theta_bounds: Optional[List[Tuple[float, float]]] = None


@dataclass
class AcqConfig:
    kind: str = "EHVI"   # only "EHVI" is currently implemented
    n_mc: int = 512      # Monte-Carlo samples used to estimate the EHVI
    mc_chunk: int = 256  # candidates processed per batch (caps peak memory)

    # Direction of each objective. A single bool applies to all of them; a
    # sequence gives one flag per objective. Default False = minimize
    # everything, consistent with sbo and mfbo.
    maximize: Union[bool, Sequence[bool]] = False

    # Hypervolume reference point, in PHYSICAL objective units. If None it
    # is built once from the initial dataset (see _resolve_ref_point) and
    # then held FIXED for the whole run, so that the hypervolume recorded at
    # different iterations is directly comparable.
    ref_point: Optional[Array] = None
    ref_margin: float = 0.10


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
class MOBOConfig:
    # --- Pure opti related parameters
    n_init: int = 10   # size of the initial data set
    n_iter: int = 20   # number of BO iterations
    init_sampling: str = "lhs"  # "uniform" or "lhs"
    random_state: Optional[int] = None


@dataclass
class SaveConfig:
    # --- Saving related parameters
    # A run always lands in its own subfolder of out_path (never directly in
    # out_path, and never nesting into a previous run's folder): named by
    # timestamp if create_timestamp=True (default), else auto-incrementing
    # run_<n>. That subfolder holds log.log, plots/ and res/ (see
    # persistence.write_meta / persistence.load_run and plotting.py).
    out_path: Optional[str] = None
    create_timestamp: bool = True
    # Logging (text): run start/end only -- config is in res/meta.json,
    # per-iteration values are in the Tier 1 trace (res/trace.npz|csv).
    log_enabled: bool = True
    log_filename: str = "log.log"

    # --- Tier 1: lightweight per-iteration trace (numpy .npz)
    # Same contract as sbo/mfbo, except that a multi-objective run has no
    # scalar incumbent: the recorded progress measure is the dominated
    # HYPERVOLUME and the size of the current Pareto front (see
    # persistence.TraceLog).
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
    plt_pareto_enabled: bool = False
    plt_conv_enabled: bool = False
    plt_hist_enabled: bool = False
    plt_all: bool = False
    plot_every: int = 1
    dpi: int = 250


# ----------------------------
# GP model container
# ----------------------------

@dataclass
class NormalizationHelper:
    """Normalizes X to the unit cube through the bounds, and each objective
    independently to zero mean / unit variance.

    Y normalization is per-output (unlike sbo's scalar version) because the
    objectives of a multi-objective problem routinely live on completely
    different scales; without it the coregionalization matrix B has to
    absorb that scale difference and the hyperparameter optimization becomes
    badly conditioned."""

    x_lo: Array
    x_hi: Array
    y_mean: Optional[Array] = None  # (M,)
    y_std: Optional[Array] = None   # (M,)
    eps: float = 1e-12

    def normalize_X(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        denom = np.maximum(self.x_hi - self.x_lo, self.eps)
        return (X - self.x_lo) / denom

    def denormalize_X(self, Xn: Array) -> Array:
        Xn = np.asarray(Xn, dtype=float)
        return self.x_lo + Xn * (self.x_hi - self.x_lo)

    def fit_Y(self, Y: Array) -> None:
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        self.y_mean = np.mean(Y, axis=0)
        y_std = np.std(Y, axis=0)
        self.y_std = np.where(y_std > self.eps, y_std, 1.0)

    def normalize_Y(self, Y: Array) -> Array:
        Y = np.asarray(Y, dtype=float)
        return (Y - self.y_mean) / self.y_std

    def denormalize_Y(self, Yn: Array) -> Array:
        Yn = np.asarray(Yn, dtype=float)
        return self.y_mean + self.y_std * Yn

    def denormalize_cov(self, cov_n: Array) -> Array:
        """Map (..., M, M) normalized covariances back to physical units:
        Sigma = D Sigma_n D with D = diag(y_std)."""
        cov_n = np.asarray(cov_n, dtype=float)
        scale = self.y_std[:, None] * self.y_std[None, :]
        return cov_n * scale


@dataclass
class ICMGPModel:
    n_obj: int
    length_scales: Array      # (d,)
    B: Array                  # (M, M), positive semi-definite
    sigma_n: Array            # (M,)
    jitter: float

    X: Optional[Array] = None   # (n, d), physical
    Y: Optional[Array] = None   # (n, M), physical
    alpha: Optional[Array] = None
    L: Optional[Array] = None

    theta: Optional[Array] = None

    # Normalized training data
    X_norm: Optional[Array] = None
    Y_norm: Optional[Array] = None
    normalizer: Optional[NormalizationHelper] = None


# ----------------------------
# State / Results containers
# ----------------------------

@dataclass
class MOBOState:
    it: int
    X: Array
    Y: Array
    gp: ICMGPModel
    acq_res: "AcqOptimizationResult"
    x_next: Array
    y_next: Array           # (M,)
    pareto_X: Array         # (p, d) Pareto-optimal designs AFTER this iteration's evaluation
    pareto_Y: Array         # (p, M) matching objective vectors, physical units
    hypervolume: float      # dominated hypervolume of pareto_Y w.r.t. ref_point
    hypervolume_acq: Optional[float] = None  # hypervolume BEFORE it, i.e. of the front the acquisition used


@dataclass
class MOBOResult:
    X: Array
    Y: Array
    pareto_X: Array
    pareto_Y: Array
    hypervolume: float
    ref_point: Array
    history: List[Dict[str, Any]]
    gp: ICMGPModel
    states: List[MOBOState]

    trace: Optional["TraceLog"] = None
    snapshots: Optional[List["GPSnapshot"]] = None
    # Resolved run folder actually written to (None if save_cfg.out_path was
    # None): <out_path>/res/ holds meta.json + trace + snapshots,
    # <out_path>/plots/ holds figures, <out_path>/log.log the run log.
    # persistence.load_run(out_path) reads it back.
    out_path: Optional[str] = None


@dataclass
class AcqOptimizationResult:
    x_next: Array          # (d,)
    a_best: float
    Xcand: Array           # (N, d)
    a: Array               # (N,)
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


def evaluate_objective(f: Callable[[Array], Array], X: Array) -> Array:
    """Evaluate a vector-valued objective on every row of X.

    f(x) must return the M objective values at x; the results are stacked
    into an (n, M) array. All rows must have the same M."""

    rows = []
    for i in range(X.shape[0]):
        val = np.asarray(f(X[i]), dtype=float).reshape(-1)
        rows.append(val)

    widths = {r.shape[0] for r in rows}
    if len(widths) != 1:
        raise ValueError(f"The objective returned inconsistent numbers of outputs: {sorted(widths)}.")

    return np.asarray(rows, dtype=float)


def _check_in_bounds(X: Array, bounds: Bounds) -> None:
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    if np.any(X < lo) or np.any(X > hi):
        raise ValueError("Some initial points lie outside the prescribed bounds.")


def init_dataset(
    f: Callable[[Array], Array],
    bounds: Bounds,
    mobo_cfg: MOBOConfig,
    rng: np.random.Generator,
    X_init: Optional[Array] = None,
    Y_init: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Initialize the dataset, either from user-provided evaluations or by
    sampling mobo_cfg.n_init points."""

    d_bounds = len(bounds)

    if X_init is None:
        X = sample_initial_points(bounds, mobo_cfg.n_init, rng, method=mobo_cfg.init_sampling)
        Y = evaluate_objective(f, X)
    else:
        X = np.asarray(X_init, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != d_bounds:
            raise ValueError(f"X_init has shape {X.shape}, expected dimension {d_bounds}.")
        _check_in_bounds(X, bounds)
        Y = evaluate_objective(f, X) if Y_init is None else np.atleast_2d(np.asarray(Y_init, dtype=float))

    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but Y has {Y.shape[0]}.")
    if X.shape[0] < 2:
        raise ValueError(
            "At least two initial points are required: the per-objective "
            "normalization needs a spread to estimate, and so does the "
            "automatic reference point."
        )

    return X, Y


def _objective_signs(maximize: Union[bool, Sequence[bool]], n_obj: int) -> Array:
    """+1 for an objective that is maximized, -1 for one that is minimized.

    Multiplying the physical objectives by these signs maps any mix of
    minimization and maximization onto the pure-maximization convention used
    by every Pareto / hypervolume routine below (and by the legacy script).
    The map is an involution, so the same signs convert a reference point
    back and forth."""

    if isinstance(maximize, (bool, np.bool_)):
        flags = [bool(maximize)] * n_obj
    else:
        flags = [bool(m) for m in maximize]
        if len(flags) != n_obj:
            raise ValueError(f"acq_cfg.maximize has {len(flags)} entries but there are {n_obj} objectives.")

    return np.where(np.asarray(flags, dtype=bool), 1.0, -1.0)


#%% ICM (Kronecker) multi-output GP: kernel + fit / prediction

def rbf_kernel(X1: Array, X2: Array, gamma: float) -> Array:
    sqdist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
    return np.exp(-gamma * sqdist)


def rbf_kernel_ard(X1: Array, X2: Array, length_scales: Array) -> Array:
    """k(x,x') = exp(-0.5 * sum_k (x_k - x'_k)^2 / l_k^2), i.e. an ARD RBF of
    UNIT amplitude.

    The legacy script carried a separate kernel amplitude sigma_f alongside
    B; the two are exactly redundant (only sigma_f^2 * B is identifiable),
    which leaves a flat ridge in the marginal likelihood for the optimizer
    to wander along. Here the amplitude lives entirely in B."""

    length_scales = np.atleast_1d(np.asarray(length_scales, dtype=float))
    X1s = np.asarray(X1, dtype=float) / length_scales
    X2s = np.asarray(X2, dtype=float) / length_scales

    sqdist = (
        np.sum(X1s ** 2, axis=1)[:, None]
        + np.sum(X2s ** 2, axis=1)[None, :]
        - 2.0 * X1s @ X2s.T
    )
    return np.exp(-0.5 * np.maximum(sqdist, 0.0))


def build_B(L_B_params: Array, n_obj: int) -> Array:
    """B = L_B L_B^T from the row-major lower triangle of L_B, with the
    diagonal given in log-space. B is positive semi-definite by
    construction, for any value of the parameters."""

    L_B_params = np.asarray(L_B_params, dtype=float).reshape(-1)
    expected = n_obj * (n_obj + 1) // 2
    if L_B_params.shape[0] != expected:
        raise ValueError(f"Expected {expected} L_B parameters for {n_obj} outputs, got {L_B_params.shape[0]}.")

    L_B = np.zeros((n_obj, n_obj), dtype=float)
    k = 0
    for i in range(n_obj):
        for j in range(i + 1):
            L_B[i, j] = np.exp(L_B_params[k]) if i == j else L_B_params[k]
            k += 1

    return L_B @ L_B.T


def _L_B_params_from_B(B: Array) -> Array:
    """Inverse of build_B, used to seed the hyperparameter optimizer from
    the GP's current B."""

    B = np.asarray(B, dtype=float)
    n_obj = B.shape[0]
    L_B = np.linalg.cholesky(B + 1e-12 * np.eye(n_obj))

    params = []
    for i in range(n_obj):
        for j in range(i + 1):
            params.append(np.log(max(L_B[i, j], 1e-12)) if i == j else L_B[i, j])

    return np.asarray(params, dtype=float)


def pack_theta(length_scales: Array, B: Array, sigma_n: Array) -> Array:
    """theta = [log(length_scales), L_B params, log(sigma_n)]."""
    return np.concatenate([
        np.log(np.atleast_1d(np.asarray(length_scales, dtype=float))),
        _L_B_params_from_B(B),
        np.log(np.atleast_1d(np.asarray(sigma_n, dtype=float))),
    ])


def unpack_theta(theta: Array, d: int, n_obj: int) -> Tuple[Array, Array, Array]:
    """Inverse of pack_theta: returns (length_scales, B, sigma_n)."""

    theta = np.asarray(theta, dtype=float).reshape(-1)
    n_B = n_obj * (n_obj + 1) // 2
    if theta.shape[0] != d + n_B + n_obj:
        raise ValueError(f"theta has length {theta.shape[0]}, expected {d + n_B + n_obj}.")

    length_scales = np.exp(theta[:d])
    B = build_B(theta[d:d + n_B], n_obj)
    sigma_n = np.exp(theta[d + n_B:])
    return length_scales, B, sigma_n


def build_icm_covariance(gp: ICMGPModel, Xn: Array, add_noise: bool = True) -> Array:
    """Assemble the joint (M*n, M*n) ICM covariance of the packed output
    vector [y_1(X); ...; y_M(X)] (see pack_Y)."""

    n = Xn.shape[0]
    Kx = rbf_kernel_ard(Xn, Xn, gp.length_scales)
    K = np.kron(gp.B, Kx)

    if add_noise:
        K = K + np.kron(np.diag(np.asarray(gp.sigma_n, dtype=float) ** 2), np.eye(n))
        K = K + gp.jitter * np.eye(gp.n_obj * n)

    return K


def pack_Y(Y: Array) -> Array:
    """(n, M) -> (M*n,) ordered as [y_1(all samples), ..., y_M(all samples)],
    which is the ordering the Kronecker product np.kron(B, Kx) expects."""
    return np.asarray(Y, dtype=float).T.reshape(-1)


def unpack_Y(y_vec: Array, n: int) -> Array:
    """Inverse of pack_Y."""
    return np.asarray(y_vec, dtype=float).reshape(-1, n).T


def icm_gp_fit(gp: ICMGPModel, Xn: Array, Yn: Array) -> Tuple[Array, Array]:
    """Compute the Cholesky factor of the ICM covariance and alpha. Xn, Yn
    are expected to already be normalized (see fit_gp)."""

    K = build_icm_covariance(gp, Xn, add_noise=True)
    L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), pack_Y(Yn))
    return alpha, L


def icm_gp_predict(
    Xtest: Array,
    gp: ICMGPModel,
    return_cov: bool = False,
):
    """Posterior of all M objectives at Xtest, in PHYSICAL units.

    Returns
    -------
    mu : (ntest, M)
    var : (ntest, M)      if return_cov is False
    cov : (ntest, M, M)   if return_cov is True

    Only the per-point M x M blocks of the joint posterior covariance are
    formed. The full (M*ntest, M*ntest) covariance is never assembled: the
    acquisition only ever needs the marginal distribution at one candidate
    at a time, and the full matrix is quadratic in the number of candidates.
    """

    if gp.alpha is None or gp.L is None or gp.X_norm is None or gp.normalizer is None:
        raise RuntimeError("The GP has not been fitted yet; call fit_gp() first.")

    Xtest = np.atleast_2d(np.asarray(Xtest, dtype=float))
    Xtest_n = gp.normalizer.normalize_X(Xtest)

    n = gp.X_norm.shape[0]
    n_test = Xtest_n.shape[0]
    M = gp.n_obj

    Kx_star = rbf_kernel_ard(gp.X_norm, Xtest_n, gp.length_scales)  # (n, ntest)
    K_star = np.kron(gp.B, Kx_star)                                 # (M*n, M*ntest)

    mu_n = (K_star.T @ gp.alpha).reshape(M, n_test).T               # (ntest, M)
    mu = gp.normalizer.denormalize_Y(mu_n)

    v = solve_triangular(gp.L, K_star, lower=True)                  # (M*n, M*ntest)
    v = v.reshape(M * n, M, n_test)                                 # [row, output, test]

    if return_cov:
        # k(x,x) = 1 for the unit-amplitude ARD RBF, so the prior block is B
        cov_n = gp.B[None, :, :] - np.einsum("kit,kjt->tij", v, v)
        cov_n = 0.5 * (cov_n + np.swapaxes(cov_n, -1, -2))
        return mu, gp.normalizer.denormalize_cov(cov_n)

    var_n = np.diag(gp.B)[None, :] - np.sum(v ** 2, axis=0).T       # (ntest, M)
    var_n = np.maximum(var_n, 0.0)
    var = var_n * (np.asarray(gp.normalizer.y_std, dtype=float) ** 2)[None, :]
    return mu, var


#%% Pareto dominance and hypervolume (2 objectives)
#
# Every routine in this section works in the MAXIMIZATION convention: a
# point z dominates z' when z >= z' componentwise with at least one strict
# inequality. The driver converts physical objectives with _objective_signs
# before calling in, so a minimization problem never has to be restated by
# the user.

def nondominated_mask(Z: Array) -> Array:
    """Boolean mask of the nondominated rows of Z (maximization).

    Duplicated rows are all kept: identical points do not dominate each
    other under the strict-inequality rule."""

    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    n = Z.shape[0]
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not mask[i]:
            continue
        dominates_i = np.all(Z >= Z[i], axis=1) & np.any(Z > Z[i], axis=1)
        if np.any(dominates_i):
            mask[i] = False

    return mask


def pareto_front(Z: Array) -> Array:
    """Nondominated rows of Z (maximization), reduced to the canonical
    staircase: sorted by increasing first objective, which on a genuine 2D
    front means strictly DECREASING second objective. Duplicates and points
    tying on one objective collapse to a single representative.

    NOTE (correction w.r.t. the legacy script): the legacy pareto_front
    sorted by increasing f1 and then kept only the points whose f2 was
    *increasing*. On a maximization front f2 decreases along that sort, so
    the filter silently collapsed every front to its single first point --
    and, through the same reduction, made the legacy hypervolume too small.
    The scan below keeps the staircase instead."""

    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    if Z.shape[0] == 0:
        return Z

    P = Z[nondominated_mask(Z)]
    if P.shape[0] == 0:
        return P

    if P.shape[1] != N_OBJ_SUPPORTED:
        raise ValueError(
            f"pareto_front is implemented for {N_OBJ_SUPPORTED} objectives only, got {P.shape[1]}."
        )

    # Sort by f1 ascending, breaking ties with f2 descending so the scan
    # below keeps the best representative of each f1 level.
    order = np.lexsort((-P[:, 1], P[:, 0]))
    P = P[order]

    keep: List[Array] = []
    best_f2 = np.inf
    for i in range(P.shape[0]):
        if P[i, 1] < best_f2:
            keep.append(P[i])
            best_f2 = P[i, 1]

    return np.asarray(keep, dtype=float)


def _hv_step_function(pareto_Z: Array, ref_point: Array) -> Tuple[Array, Array]:
    """Piecewise-constant attainment surface of the region dominated by
    pareto_Z, clipped at the reference point.

    Returns (t, g) with ``t`` of length m+2 (breakpoints along objective 1,
    starting at ref[0] and ending at +inf) and ``g`` of length m+1, such
    that on the interval (t[j], t[j+1]] the dominated region reaches up to
    g[j] along objective 2. Both the hypervolume and the hypervolume
    improvement below are read off this single decomposition, which is what
    keeps them exactly consistent with each other."""

    ref_point = np.asarray(ref_point, dtype=float).reshape(-1)
    if ref_point.shape[0] != N_OBJ_SUPPORTED:
        raise ValueError(f"ref_point must have {N_OBJ_SUPPORTED} entries, got {ref_point.shape[0]}.")

    P = pareto_front(pareto_Z) if np.size(pareto_Z) else np.zeros((0, N_OBJ_SUPPORTED))

    if P.shape[0] == 0:
        return np.array([ref_point[0], np.inf]), np.array([ref_point[1]])

    t = np.concatenate([[ref_point[0]], np.maximum(P[:, 0], ref_point[0]), [np.inf]])
    g = np.concatenate([np.maximum(P[:, 1], ref_point[1]), [ref_point[1]]])
    return t, g


def hypervolume(Z: Array, ref_point: Array) -> float:
    """Hypervolume of the region dominated by Z and dominating ref_point
    (2 objectives, maximization).

    Computed by integrating the attainment staircase of _hv_step_function;
    this replaces the legacy hypervolume_2d_max, which shared the broken
    front reduction described in ``pareto_front`` and therefore under-counted
    every front holding more than one point."""

    t, g = _hv_step_function(Z, ref_point)
    ref_point = np.asarray(ref_point, dtype=float).reshape(-1)

    # The last interval runs to +inf but sits exactly at the reference
    # height, so it contributes nothing and is dropped.
    widths = t[1:-1] - t[:-2]
    heights = g[:-1] - ref_point[1]
    return float(np.sum(np.maximum(widths, 0.0) * np.maximum(heights, 0.0)))


def hypervolume_improvement(Z: Array, pareto_Z: Array, ref_point: Array) -> Array:
    """Hypervolume improvement of each row of Z over the front pareto_Z
    (2 objectives, maximization), vectorized over the rows of Z.

    Equal to hypervolume(pareto_Z + z) - hypervolume(pareto_Z) for each z,
    but evaluated in closed form against the attainment staircase, so the
    cost is O(len(Z) * len(front)) instead of one full front reduction per
    row -- which is what the legacy Monte-Carlo loop paid."""

    t, g = _hv_step_function(pareto_Z, ref_point)

    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    if Z.shape[1] != N_OBJ_SUPPORTED:
        raise ValueError(f"hypervolume_improvement expects {N_OBJ_SUPPORTED} objectives, got {Z.shape[1]}.")

    widths = np.minimum(Z[:, 0][:, None], t[None, 1:]) - t[None, :-1]
    heights = Z[:, 1][:, None] - g[None, :]
    return np.sum(np.maximum(widths, 0.0) * np.maximum(heights, 0.0), axis=1)


def _resolve_ref_point(Z: Array, acq_cfg: AcqConfig, signs: Array) -> Array:
    """Reference point in MAXIMIZATION space, either taken from the user
    (converted from physical units) or placed just below the worst observed
    value of each objective, as in the legacy script."""

    if acq_cfg.ref_point is not None:
        ref = np.asarray(acq_cfg.ref_point, dtype=float).reshape(-1)
        if ref.shape[0] != Z.shape[1]:
            raise ValueError(f"acq_cfg.ref_point has {ref.shape[0]} entries but there are {Z.shape[1]} objectives.")
        return signs * ref

    z_min = np.min(Z, axis=0)
    z_max = np.max(Z, axis=0)
    span = np.maximum(z_max - z_min, 1e-12)
    return z_min - acq_cfg.ref_margin * span


#%% Hyperparameter optimization by log marginal likelihood

def default_theta_bounds(d: int, n_obj: int) -> List[Tuple[float, float]]:
    """A generous but finite box for the packed hyperparameters.

    Used whenever GPConfig.theta_bounds is None. Bounds are not cosmetic
    here: the ICM likelihood has many more free parameters than the
    single-output one (d + M(M+1)/2 + M), and an unbounded L-BFGS-B run
    readily walks a log-scale off to +-1e3, where exp() overflows and the
    covariance fills with inf/nan. The numbers below are stated for the
    NORMALIZED problem the optimizer actually sees -- inputs in the unit
    cube, objectives standardized per output -- so they are meaningful
    independently of the physical units of any given application."""

    n_B = n_obj * (n_obj + 1) // 2
    return (
        [(np.log(1e-3), np.log(1e2))] * d        # ARD length-scales
        + [(-1e2, 1e2)] * n_B                    # L_B entries (log on the diagonal)
        + [(np.log(1e-6), np.log(1e1))] * n_obj  # observation noise
    )


def negative_log_marginal_likelihood(
    theta: Array,
    Xn: Array,
    Yn: Array,
    n_obj: int,
    jitter: float = 1e-10,
    MLE_hist: Optional[list] = None,
    param_hist: Optional[list] = None,
) -> float:
    """theta as packed by pack_theta. Returns NLL = -log p(Y | X, theta) for
    the joint (all outputs at once) ICM likelihood.

    Returns a large finite penalty rather than raising whenever theta leads
    to a covariance that is not positive definite or not even finite, so the
    optimizer can back out of such a region instead of aborting the run."""

    d = Xn.shape[1]

    with np.errstate(over="ignore", invalid="ignore"):
        length_scales, B, sigma_n = unpack_theta(theta, d, n_obj)

        if not (np.all(np.isfinite(length_scales)) and np.all(np.isfinite(B)) and np.all(np.isfinite(sigma_n))):
            return 1e12

        gp_tmp = ICMGPModel(
            n_obj=n_obj, length_scales=length_scales, B=B, sigma_n=sigma_n, jitter=jitter,
        )

        K = build_icm_covariance(gp_tmp, Xn, add_noise=True)

    if not np.all(np.isfinite(K)):
        return 1e12

    try:
        L = cholesky(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e12

    y = pack_Y(Yn)
    alpha = cho_solve((L, True), y)
    n = y.shape[0]
    ll = -0.5 * (y @ alpha) - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2.0 * np.pi)

    if not np.isfinite(ll):
        return 1e12

    if MLE_hist is not None:
        MLE_hist.append(float(ll))
    if param_hist is not None:
        param_hist.append(np.asarray(theta, dtype=float).copy())

    return -float(ll)


def optimize_gp_hyperparams(
    Xn: Array,
    Yn: Array,
    gp: ICMGPModel,
    gp_cfg: GPConfig,
    MLE_hist: Optional[list] = None,
    param_hist: Optional[list] = None,
) -> ICMGPModel:
    """Update gp.(length_scales, B, sigma_n) by minimizing the negative
    joint log marginal likelihood."""

    if gp_cfg.theta0 is not None:
        x0 = np.asarray(gp_cfg.theta0, dtype=float).reshape(-1)
    else:
        x0 = pack_theta(gp.length_scales, gp.B, gp.sigma_n)

    bounds = gp_cfg.theta_bounds
    if bounds is None:
        bounds = default_theta_bounds(Xn.shape[1], gp.n_obj)

    res = minimize(
        negative_log_marginal_likelihood,
        x0=np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds]),
        args=(Xn, Yn, gp.n_obj, gp.jitter, MLE_hist, param_hist),
        method="L-BFGS-B",
        bounds=bounds,
    )

    theta_opt = np.asarray(res.x, dtype=float)
    gp.length_scales, gp.B, gp.sigma_n = unpack_theta(theta_opt, Xn.shape[1], gp.n_obj)
    gp.theta = theta_opt.copy()

    return gp


#%% GP wrapper used by MOBO

def build_gp_model(gp_cfg: GPConfig, bounds: Bounds, n_obj: int) -> ICMGPModel:
    """Create an unfitted ICMGPModel container."""

    d = len(bounds)
    x_lo = np.array([b[0] for b in bounds], dtype=float)
    x_hi = np.array([b[1] for b in bounds], dtype=float)
    if not gp_cfg.normalize_X:
        # identity transform (x - 0) / (1 - 0): the GP sees the raw inputs
        x_lo, x_hi = np.zeros_like(x_lo), np.ones_like(x_hi)
    normalizer = NormalizationHelper(x_lo=x_lo, x_hi=x_hi)

    length_scales = np.broadcast_to(np.atleast_1d(np.asarray(gp_cfg.length_scales, dtype=float)), (d,)).copy()
    sigma_n = np.broadcast_to(np.atleast_1d(np.asarray(gp_cfg.sigma_n, dtype=float)), (n_obj,)).copy()
    L_B_diag = np.broadcast_to(np.atleast_1d(np.asarray(gp_cfg.L_B_diag, dtype=float)), (n_obj,)).copy()

    L_B_params = []
    for i in range(n_obj):
        for j in range(i + 1):
            L_B_params.append(np.log(L_B_diag[i]) if i == j else float(gp_cfg.L_B_offdiag))
    B = build_B(np.asarray(L_B_params, dtype=float), n_obj)

    return ICMGPModel(
        n_obj=n_obj, length_scales=length_scales, B=B, sigma_n=sigma_n,
        jitter=gp_cfg.jitter, normalizer=normalizer,
    )


def fit_gp(gp: ICMGPModel, X: Array, Y: Array, gp_cfg: GPConfig, it: int) -> ICMGPModel:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))

    if gp.normalizer is None:
        raise ValueError("GP normalizer is not initialized.")

    gp.X, gp.Y = X, Y
    gp.X_norm = gp.normalizer.normalize_X(X)
    if gp_cfg.normalize_y:
        gp.normalizer.fit_Y(Y)
    else:
        # identity, per output
        gp.normalizer.y_mean, gp.normalizer.y_std = np.zeros(Y.shape[1]), np.ones(Y.shape[1])
    gp.Y_norm = gp.normalizer.normalize_Y(Y)

    do_hpo = (
        gp_cfg.optimize_hyperparams
        and gp_cfg.hpo_every > 0
        and (it % gp_cfg.hpo_every == 0)
    )
    if do_hpo:
        gp = optimize_gp_hyperparams(gp.X_norm, gp.Y_norm, gp, gp_cfg)

    gp.alpha, gp.L = icm_gp_fit(gp, gp.X_norm, gp.Y_norm)
    return gp


#%% Acquisition + optimizer

def _psd_factor(cov: Array, eps: float = 1e-12) -> Array:
    """Batched square-root factor F of a stack of (M, M) covariances, with
    F F^T = cov.

    Uses a symmetric eigendecomposition with clipped eigenvalues rather than
    the legacy Cholesky-with-escalating-jitter retry loop: posterior
    covariances go numerically indefinite exactly where the GP is already
    fully informed, and clipping handles that in one shot, batched, without
    a Python-level try/except per candidate."""

    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
    w, V = np.linalg.eigh(cov)
    w = np.maximum(w, eps)
    return V * np.sqrt(w)[..., None, :]


def make_acquisition(
    acq_cfg: AcqConfig,
    gp: ICMGPModel,
    pareto_Z: Array,
    ref_z: Array,
    signs: Array,
    rng: np.random.Generator,
):
    """
    Build the Monte-Carlo EHVI acquisition acq(Xcand).

    Convention: acq(.) is MAXIMIZED. At each candidate x the ICM posterior
    gives a joint Gaussian over the M objectives; drawing from it and
    averaging the hypervolume improvement of each draw over the current
    front estimates

        EHVI(x) = E[ HVI(y) ],   y ~ N(mu(x), Sigma(x)).

    Two differences from the legacy acquisition_ehvi_mc, both of which
    matter once the acquisition is handed to a gradient-based optimizer:

      - COMMON RANDOM NUMBERS. The standard-normal draws are generated ONCE
        here and reused for every candidate and every optimizer iteration.
        The legacy version re-sampled inside each call, so the acquisition
        was a different (noisy) function on every evaluation and the
        L-BFGS-B refinement in propose_location_ehvi was differentiating
        Monte-Carlo noise. With fixed draws, acq is a deterministic function
        of x -- the sample-average approximation -- which is what makes the
        local refinement meaningful and the run reproducible from the seed.

      - VECTORIZATION. Candidates are processed in chunks and the
        hypervolume improvement is evaluated in closed form against the
        attainment staircase (see hypervolume_improvement), instead of one
        Python-level front reduction per Monte-Carlo sample.
    """

    if acq_cfg.kind.upper() != "EHVI":
        raise ValueError(f"Unknown acquisition kind '{acq_cfg.kind}'. Only 'EHVI' is implemented.")
    if gp.n_obj != N_OBJ_SUPPORTED:
        raise ValueError(
            f"The EHVI acquisition is implemented for {N_OBJ_SUPPORTED} objectives only, "
            f"but the GP has {gp.n_obj}."
        )

    chunk = max(int(acq_cfg.mc_chunk), 1)

    # Common random numbers: drawn once, reused for every evaluation, and
    # ANTITHETIC -- every draw z is paired with -z. Beyond the usual
    # variance reduction, this makes the estimator symmetric under a sign
    # flip of the objectives, so minimizing f and maximizing -f give the
    # same acquisition surface rather than two different Monte-Carlo
    # realizations of it. n_mc is rounded up to the next even number.
    n_half = (int(acq_cfg.n_mc) + 1) // 2
    Z_half = rng.standard_normal((n_half, gp.n_obj))
    Z_mc = np.vstack([Z_half, -Z_half])
    n_mc = Z_mc.shape[0]

    # The front and the reference point are fixed within an iteration, so
    # the attainment staircase is built once too.
    t_edges, g_vals = _hv_step_function(pareto_Z, ref_z)
    signs = np.asarray(signs, dtype=float).reshape(-1)

    def _hvi(Z: Array) -> Array:
        widths = np.minimum(Z[:, 0][:, None], t_edges[None, 1:]) - t_edges[None, :-1]
        heights = Z[:, 1][:, None] - g_vals[None, :]
        return np.sum(np.maximum(widths, 0.0) * np.maximum(heights, 0.0), axis=1)

    def acq(Xcand: Array) -> Array:
        Xcand = np.atleast_2d(np.asarray(Xcand, dtype=float))
        N = Xcand.shape[0]
        out = np.empty(N, dtype=float)

        for start in range(0, N, chunk):
            stop = min(start + chunk, N)
            mu, cov = icm_gp_predict(Xcand[start:stop], gp, return_cov=True)

            F = _psd_factor(cov)                                          # (nc, M, M)
            samples = mu[:, None, :] + np.einsum("nij,kj->nki", F, Z_mc)  # (nc, n_mc, M)

            Z_samples = signs[None, None, :] * samples
            hvi = _hvi(Z_samples.reshape(-1, gp.n_obj)).reshape(stop - start, n_mc)
            out[start:stop] = hvi.mean(axis=1)

        return out

    return acq


def _make_random_candidates(bounds: Bounds, N: int, rng: np.random.Generator) -> Array:
    d = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo + (hi - lo) * rng.random((N, d))


def _make_grid_candidates(bounds: Bounds, n_per_dim: int) -> Array:
    grids_1d = [np.linspace(b[0], b[1], n_per_dim) for b in bounds]
    mesh = np.meshgrid(*grids_1d, indexing="xy")
    return np.stack([m.ravel() for m in mesh], axis=1)


def optimize_acquisition(
    acq: Callable[[Array], Array],
    bounds: Bounds,
    optim_cfg: OptimConfig,
    rng: np.random.Generator,
) -> AcqOptimizationResult:
    """
    Optimize the acquisition to propose the next x.

    Supported modes (mirrors sbo.core.optimize_acquisition):
      - method="random": global random candidates only
      - method="grid":   global grid candidates only
      - method="refined": global (random or grid) + local L-BFGS-B refinement
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

    a = np.asarray(acq(Xcand), dtype=float).reshape(-1)
    a[~np.isfinite(a)] = -np.inf

    best_idx = int(np.argmax(a))
    x_best = Xcand[best_idx].copy()
    a_best = float(a[best_idx])

    if method in ["random", "grid"]:
        run_time = time.perf_counter() - start_time
        return AcqOptimizationResult(x_next=x_best, a_best=a_best, Xcand=Xcand, a=a, run_time=run_time)

    # ---- refined: local improvement from the best few candidates
    n_starts = min(optim_cfg.n_restarts, Xcand.shape[0])
    top_idx = np.argpartition(-a, n_starts - 1)[:n_starts]
    Xstarts = Xcand[top_idx]

    scipy_bounds = [(float(b[0]), float(b[1])) for b in bounds]

    def obj(x: Array) -> float:
        x2 = np.asarray(x, dtype=float).reshape(1, -1)
        val = float(np.asarray(acq(x2)).reshape(-1)[0])
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
    return AcqOptimizationResult(x_next=x_best, a_best=a_best, Xcand=Xcand, a=a, run_time=run_time)


#%% Main MOBO function

def multi_objective_bayesian_optimization(
    f: Callable[[Array], Array],
    bounds: Bounds,
    mobo_cfg: MOBOConfig,
    gp_cfg: GPConfig,
    acq_cfg: AcqConfig,
    optim_cfg: OptimConfig,
    save_cfg: SaveConfig,
    X_init: Optional[Array] = None,
    Y_init: Optional[Array] = None,
    states: Optional[List[MOBOState]] = None,
    f_true: Optional[Callable[[Array], Array]] = None,
) -> MOBOResult:
    """
    Multi-objective Bayesian Optimization driver (every objective minimized
    by default -- set acq_cfg.maximize, globally or per objective, otherwise).

    Each iteration fits ONE ICM GP jointly to all objectives, proposes the
    location x that maximizes the Monte-Carlo EHVI over the current Pareto
    front, evaluates f there and appends the result. There is no scalar
    incumbent: progress is measured by the dominated hypervolume, with
    respect to a reference point fixed once (from the initial data, or by
    the user) so that it stays comparable across iterations.
    """

    from .plotting import setup_plotting_toggle, plt_state, plt_pareto, plt_hist, plt_conv
    from . import persistence

    rng = _rng(mobo_cfg.random_state)

    if save_cfg.trace_flush_every is not None and save_cfg.out_path is None:
        raise ValueError("save_cfg.trace_flush_every requires save_cfg.out_path to be set.")
    if save_cfg.snapshot_flush_every is not None and save_cfg.out_path is None:
        raise ValueError("save_cfg.snapshot_flush_every requires save_cfg.out_path to be set.")

    if states is None:
        states = []

    # Resolve the run folder. This returns a resolved COPY of save_cfg
    # (out_path pointing at the run subfolder); rebinding the local name means
    # every downstream use (plotting, persistence) sees the resolved path,
    # while the SaveConfig the caller passed in is left untouched and can be
    # reused across several runs.
    experiment_path, save_cfg = setup_experiment_folder(save_cfg)
    setup_plotting_toggle(save_cfg)

    # ---- 1) initial dataset
    X, Y = init_dataset(f=f, bounds=bounds, mobo_cfg=mobo_cfg, rng=rng, X_init=X_init, Y_init=Y_init)

    n_obj = Y.shape[1]
    if n_obj != N_OBJ_SUPPORTED:
        raise ValueError(
            f"mobo currently supports {N_OBJ_SUPPORTED} objectives; the objective returned {n_obj}."
        )

    signs = _objective_signs(acq_cfg.maximize, n_obj)
    ref_z = _resolve_ref_point(signs * Y, acq_cfg, signs)
    ref_point = signs * ref_z  # back to physical units, for reporting

    # log.log is metadata-only: full configuration (incl. the reference point)
    # is in meta.json (persistence.write_meta), per-iteration values are in
    # the Tier 1 trace (trace.npz / trace.csv). This just marks the run start/end.
    logger = None
    if save_cfg.log_enabled and experiment_path is not None:
        log_path = os.path.join(experiment_path, save_cfg.log_filename)
        logger = setup_logger(log_path)
        logger.info(f"MOBO run started: {mobo_cfg.n_iter} iterations. Configuration in meta.json.")

    history: List[Dict[str, Any]] = []

    # Everything persisted (metadata, trace, snapshots) lives under
    # <run folder>/res/; plots under <run folder>/plots/ (plotting.py).
    res_path = os.path.join(experiment_path, "res") if experiment_path is not None else None

    trace = persistence.TraceLog() if save_cfg.trace_enabled else None
    trace_path = (
        os.path.join(res_path, save_cfg.trace_filename)
        if (res_path is not None and trace is not None)
        else None
    )

    snapshot_buffer: Optional[List["persistence.GPSnapshot"]] = (
        [] if save_cfg.snapshot_enabled else None
    )
    snapshot_dir = (
        os.path.join(res_path, save_cfg.snapshot_dirname)
        if (res_path is not None and snapshot_buffer is not None)
        else None
    )

    if res_path is not None:
        persistence.write_meta(
            res_path,
            bounds=bounds,
            random_state=mobo_cfg.random_state,
            ref_point=ref_point,
            mobo_cfg=mobo_cfg,
            gp_cfg=gp_cfg,
            acq_cfg=acq_cfg,
            optim_cfg=optim_cfg,
            save_cfg=save_cfg,
        )

    # ---- 2) build GP object
    gp = build_gp_model(gp_cfg, bounds, n_obj)

    # ---- 3) MOBO loop
    for it in range(mobo_cfg.n_iter):
        iter_start_time = time.perf_counter()

        gp = fit_gp(gp, X, Y, gp_cfg, it)

        # Front BEFORE this iteration's evaluation: this is what the
        # acquisition must use. What is recorded per iteration
        # (state/trace/history) is the post-update front, below.
        Z = signs * Y
        mask = nondominated_mask(Z)
        hv_acq = hypervolume(Z[mask], ref_z)

        acq = make_acquisition(acq_cfg, gp, Z[mask], ref_z, signs, rng)
        acq_res = optimize_acquisition(acq, bounds, optim_cfg, rng)

        x_next = acq_res.x_next
        y_next = np.asarray(f(x_next), dtype=float).reshape(-1)
        if y_next.shape[0] != n_obj:
            raise ValueError(f"The objective returned {y_next.shape[0]} values at x_next, expected {n_obj}.")

        X = np.vstack([X, x_next[None, :]])
        Y = np.vstack([Y, y_next[None, :]])

        # Pareto front / hypervolume INCLUDING the point just evaluated, as in
        # sbo's running best: row `it` of state/trace/history is consistent
        # with the evaluation at `it`.
        Z = signs * Y
        mask = nondominated_mask(Z)
        pareto_X = X[mask].copy()
        pareto_Y = Y[mask].copy()
        hv = hypervolume(Z[mask], ref_z)

        # Plot the current state. Done after the evaluation so the figure also
        # shows the objective values obtained at x_next and the post-update
        # Pareto set. gp, acq and acq_res are still the pre-update ones (gp is
        # only refit at the next iteration), i.e. what the acquisition decided from.
        if save_cfg.plt_state_enabled and (it % save_cfg.plot_every == 0):
            plt_state(
                gp=gp, acq_res=acq_res, bounds=bounds, it=it,
                pareto_X=pareto_X, pareto_Y=pareto_Y,
                save_cfg=save_cfg, acq_cfg=acq_cfg, mobo_cfg=mobo_cfg, acq=acq,
                f_true=f_true, y_new=y_next,
            )

        if save_cfg.plt_pareto_enabled and (it % save_cfg.plot_every == 0):
            plt_pareto(
                Y=Y, pareto_Y=pareto_Y, y_next=y_next, ref_point=ref_point,
                it=it, hv=hv, save_cfg=save_cfg, mobo_cfg=mobo_cfg,
            )

        state = MOBOState(
            it=it, X=X.copy(), Y=Y.copy(), gp=copy.deepcopy(gp), acq_res=acq_res,
            x_next=x_next.copy(), y_next=y_next.copy(),
            pareto_X=pareto_X, pareto_Y=pareto_Y, hypervolume=float(hv),
            hypervolume_acq=float(hv_acq),
        )
        states.append(state)

        iter_wall_time = time.perf_counter() - iter_start_time

        if trace is not None:
            trace.append(
                it=it, x_next=x_next, y_next=y_next,
                hypervolume=hv, n_pareto=pareto_Y.shape[0],
                wall_time=iter_wall_time, acq_value=acq_res.a_best,
                length_scales=gp.length_scales, B=gp.B, sigma_n=gp.sigma_n,
            )
            if (
                save_cfg.trace_flush_every is not None
                and (it + 1) % save_cfg.trace_flush_every == 0
                and trace_path is not None
            ):
                trace.save(trace_path)

        if snapshot_buffer is not None and (it % save_cfg.snapshot_every == 0):
            snapshot_buffer.append(persistence.build_gp_snapshot(it, gp, ref_point))
            if (
                save_cfg.snapshot_flush_every is not None
                and len(snapshot_buffer) >= save_cfg.snapshot_flush_every
                and snapshot_dir is not None
            ):
                persistence.flush_snapshots(snapshot_dir, snapshot_buffer)
                snapshot_buffer.clear()

        history.append(
            dict(
                it=it, x_next=x_next.copy(), y_next=y_next.copy(),
                pareto_Y=pareto_Y.copy(), hypervolume=float(hv),
                n_pareto=int(pareto_Y.shape[0]),
                acq_value=acq_res.a_best, acq_time=acq_res.run_time, n_obs=X.shape[0],
                length_scales=np.asarray(gp.length_scales, dtype=float).copy(),
                B=np.asarray(gp.B, dtype=float).copy(),
                sigma_n=np.asarray(gp.sigma_n, dtype=float).copy(),
            )
        )

    if trace is not None and trace_path is not None:
        trace.save(trace_path)

    if snapshot_buffer is not None and snapshot_dir is not None and len(snapshot_buffer) > 0:
        persistence.flush_snapshots(snapshot_dir, snapshot_buffer)

    if save_cfg.plt_hist_enabled:
        plt_hist(history, save_cfg)

    if save_cfg.plt_conv_enabled:
        plt_conv(history, save_cfg)

    # ---- 4) final front, now including the last proposal
    Z = signs * Y
    mask = nondominated_mask(Z)
    hv_final = hypervolume(Z[mask], ref_z)

    if logger is not None:
        log_summary(logger, hv_final, int(mask.sum()), n_evals=len(Y), ref_point=ref_point)

    return MOBOResult(
        X=X, Y=Y,
        pareto_X=X[mask].copy(), pareto_Y=Y[mask].copy(),
        hypervolume=hv_final,
        ref_point=ref_point,
        history=history, gp=gp, states=states,
        trace=trace, snapshots=snapshot_buffer,
        out_path=experiment_path,
    )
