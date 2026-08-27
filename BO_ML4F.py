# -*- coding: utf-8 -*-
"""
Updated on Wed Jan 28 10:43:30 2026

@author: mendez, lecomte
"""

from __future__ import annotations  # this is to annotate the functions (-->)
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Tuple, Union

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import logging

# Import the functions of the home-made BO
from BO_func_YL import rbf_kernel_

#Customization of the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=12)

Array = np.ndarray  # short cut for numpy arrays
# short cut for domain bounds as a list
Bounds = List[Tuple[float, float]]  # [(l1,u1), ..., (ld,ud)]


# ----------------------------
# Settings / Config containers
# ----------------------------
# these are dataclasses that makes it simple to access all the settings.

# general settigns of the GP (hyperparameters etc)
@dataclass
class GPConfig:
    # if optimize_hyperparams=False, these values are used and kept fixed
    #
    # l_c is EITHER a scalar (isotropic kernel -- the default and the historical
    # behaviour of this library) OR a length-d sequence, one length scale per
    # input coordinate.  A vector l_c with optimize_hyperparams=False is a
    # "frozen ARD" kernel: the anisotropy is declared up front and never fitted.
    # See the ARD section of README.md and docs/ARD_IMPLEMENTATION_NOTE.md.
    l_c: Union[float, Array] = 0.3   # length-scale ell: scalar, or (d,) for ARD
    sigma_f: float = 1.0      # kernel amplitude std (so variance is sigma_f^2)
    sigma_y: float = 0.1      # observation noise std
    jitter: float = 1e-10     # numerical stabilizer on diagonal

    # hyperparameter optimization (HPO) options
    optimize_hyperparams: bool = False
    hpo_every: int = 1        # optimize every k BO iterations (1 = every iteration)

    # optional initial guess and bounds in log-space.
    #
    # Layout of theta_log:
    #   isotropic (ard=False) : log([l_c, sigma_f, sigma_y])            -> 3
    #   ARD       (ard=True)  : log([l_1, ..., l_d, sigma_f, sigma_y])  -> d+2
    #
    # For convenience a 3-entry theta0_log / theta_bounds_log is accepted in
    # ARD mode as well: its length-scale entry is broadcast to all d
    # coordinates.  This is what lets an existing isotropic configuration be
    # switched to ARD without rewriting its HPO box.
    theta0_log: Optional[Array] = None
    theta_bounds_log: Optional[List[Tuple[float, float]]] = None

    # Rank-1 Cholesky update: switch from O(n^3) full refit to O(n^2) block
    # extension once the dataset reaches rank1_threshold points.  Set to 0 to
    # disable entirely.  Skipped automatically whenever HPO is triggered.
    # Example: rank1_threshold=500 with 300 initial points → kicks in after
    #          200 BO iterations; with 500 initial points → kicks in from iter 1.
    rank1_threshold: int = 0

    # Multi-restart HPO: number of random restarts added on top of the warm start.
    n_hpo_restarts: int = 3

    # Kernel selection.
    #   "rbf"      — squared-exponential, infinitely differentiable.  Good baseline.
    #   "matern52" — Matérn ν=5/2, twice differentiable.  More realistic for physical
    #                simulations whose outputs are not infinitely smooth.
    kernel: str = "rbf"

    # Output normalisation: subtract mean and divide by std of observed y before fitting.
    # Keeps the GP amplitude σ_f and noise σ_y on a well-conditioned scale regardless
    # of the absolute level of the objective.  Recommended when y values are large or
    # when HPO struggles to converge.
    normalize_y: bool = False

    # ------------------------------------------------------------------
    # Automatic Relevance Determination (ARD)
    # ------------------------------------------------------------------
    # ard=False (default) is the historical isotropic library, bit for bit.
    #
    # ard=True gives the kernel one length scale per input coordinate and HPO
    # then optimises d+2 hyperparameters instead of 3.  A short length scale
    # means the objective decorrelates quickly along that coordinate, i.e. the
    # coordinate is "relevant"; a length scale that runs to the upper bound
    # means no variation was detected over the search box at this sample size.
    #
    # COST: d-1 extra hyperparameters are fitted from the same n observations.
    # ARD is a clear win offline (large n, screening / diagnosis) and can lose
    # to the isotropic kernel in-campaign when n is small -- see
    # docs/ARD_IMPLEMENTATION_NOTE.md, section "When NOT to switch this on".
    #
    # This flag only decides whether a SCALAR l_c is expanded to d length
    # scales.  A vector l_c is anisotropic whatever this flag says, and HPO
    # then optimises d+2 hyperparameters rather than silently collapsing the
    # declared anisotropy back to one number.
    ard: bool = False

    # Seed for the random HPO restarts.  None (default) preserves the historical
    # behaviour of drawing them from an unseeded Generator, which makes runs
    # non-reproducible.  Set an integer to make HPO -- and hence a whole BO
    # campaign -- bit-reproducible.
    hpo_random_state: Optional[int] = None


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
    method: str = "random"        # "random", "grid", "refined"
    global_method: str = "random" # used only if method=="refined": "random" or "grid"

    optimizer: str = "L-BFGS-B"
    n_raw_samples: int = 2000     # for random
    grid_n_per_dim: int = 20      # for grid (roughly)
    n_restarts: int = 10          # number of best global points used as starts
    local_maxiter: int = 200
    n_candidates: int = 1         # diverse acquisition proposals per BO round
    batch_distance_scale: float = 0.05  # relative to normalized search box


@dataclass
class GradientRefinementConfig:
    """Optional local refinement of each acquisition proposal."""
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
    n_init: int = 5
    n_iter: int = 20
    random_state: Optional[int] = None
    export_every: Optional[int] = None  # e.g. export state every k iterations

    # NEW: how we sample the initial points
    init_sampling: str = "uniform"  # "uniform" or "lhs"


# ----------------------------
# GP model container
# ----------------------------
# this stores exactly the quantities you derive in the lecture:
# (training points + Cholesky factor + alpha + hyperparameters)

@dataclass
class GPModel:
    l_c: Union[float, Array]   # scalar (isotropic) or (d,) array (ARD)
    sigma_f: float
    sigma_y: float
    jitter: float

    Xs: Optional[Array] = None     # training inputs
    ys: Optional[Array] = None     # training outputs
    alpha: Optional[Array] = None  # (Kss + σ²I)^(-1) y
    L: Optional[Array] = None      # Cholesky factor

    # optional: store last optimized theta in log-space.
    # Layout: log([l_1, ..., l_n_ell, sigma_f, sigma_y]); n_ell is 1 when the
    # kernel is isotropic and d when ARD is active.
    theta_log: Optional[Array] = None

    # kernel kind (copied from GPConfig via build_gp_model)
    kernel: str = "rbf"

    # output normalisation statistics (updated by fit_gp when normalize_y=True)
    y_mean: float = 0.0   # subtracted from y before fitting
    y_std:  float = 1.0   # divisor; predictions are scaled back by this


# ----------------------------
# State / Results containers
# ----------------------------
# collect the furrent state of the BO
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
    x_proposed: Optional[Array] = None
    y_proposed: Optional[Array] = None
    x_refined: Optional[Array] = None
    y_refined: Optional[Array] = None
    refinement_displacement: Optional[Array] = None


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


# collects all the information concerning the evaluation of
# the next point to sample
@dataclass
class AcqOptimizationResult:
    x_next: Array          # (d,)
    a_best: Array          # (d,)
    Xcand: Array           # (N, d)
    a: Array               # (N,)


# Give information about possible plot and saving
@dataclass
class SaveConfig:
    save_path: Optional[str] = None    # directory for all outputs
    log_enabled: bool = True
    log_filename: str = "run.log"
    # --- Per-iteration state export (recommended for post-processing / animation)
    export_states: bool = False        # save .npz snapshot each iteration
    n_plot: int = 400                  # grid resolution for 1D GP predictions in exports
    # --- Inline plotting (interactive sessions only, 1D problems)
    plt_state_enabled: bool = False    # legacy switch — use export_states instead

# ----------------------------
# Helper utilities (skeleton)
# ----------------------------
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
# Length-scale helpers (isotropic / ARD)
# ----------------------------
# One scalar length scale is an isotropic kernel: the objective is assumed to
# decorrelate at the same rate along every coordinate.  ARD replaces that scalar
# by a vector, which is equivalent to rescaling the inputs by 1/l_i before an
# isotropic kernel sees them.  Everything below is written so that a scalar l_c
# takes exactly the code path it always did.


def as_length_scales(l_c: Union[float, Array], d: Optional[int] = None) -> Array:
    """
    Normalise a length-scale specification to a 1-D positive float array.

    INPUTS:
      l_c : scalar (isotropic) or sequence of length d (ARD)
      d   : optional input dimension used to validate a vector specification
    OUTPUT:
      ell : (1,) array for an isotropic kernel, or (d,) for ARD
    """
    ell = np.atleast_1d(np.asarray(l_c, dtype=float)).reshape(-1)
    if ell.size == 0:
        raise ValueError("l_c must contain at least one length scale.")
    if not np.all(np.isfinite(ell)) or np.any(ell <= 0.0):
        raise ValueError(f"length scales must be finite and strictly positive, got {ell}.")
    if d is not None and ell.size not in (1, d):
        raise ValueError(
            f"l_c has {ell.size} length scales but the inputs have d = {d} "
            f"coordinates; use a scalar (isotropic) or exactly d values (ARD)."
        )
    return ell


def is_ard(l_c: Union[float, Array]) -> bool:
    """True when l_c carries more than one length scale."""
    return np.atleast_1d(np.asarray(l_c, dtype=float)).size > 1


def format_length_scales(l_c: Union[float, Array], fmt: str = ".3e") -> str:
    """Log-friendly rendering; a scalar formats exactly as it always did."""
    ell = np.atleast_1d(np.asarray(l_c, dtype=float)).reshape(-1)
    if ell.size == 1:
        return format(float(ell[0]), fmt)
    return "[" + ", ".join(format(float(v), fmt) for v in ell) + "]"


def ard_whitening_weights(l_c: Union[float, Array], normalize: bool = True) -> Array:
    """
    Turn fitted ARD length scales into input-rescaling weights w_i = 1 / l_i.

    Rescaling the inputs by these weights and then using an ISOTROPIC kernel is
    algebraically identical to ARD with those length scales.  That is the route
    to take when the anisotropy must be declared and frozen before a campaign
    rather than fitted during one.

    With normalize=True the weights are scaled so that their geometric mean is
    one, which leaves the overall scale of the rescaled box -- and therefore the
    meaning of a shared isotropic length scale and of any trust-region radius
    expressed in it -- comparable to the unrescaled box.
    """
    ell = as_length_scales(l_c)
    w = 1.0 / ell
    if normalize:
        w = w / float(np.exp(np.mean(np.log(w))))
    return w


def _expand_theta_bounds(
    theta_bounds_log: Optional[List[Tuple[float, float]]], n_ell: int
) -> Optional[List[Tuple[float, float]]]:
    """
    Bring an HPO box to the (n_ell + 2) layout.

    A 3-entry box is read as (length-scale, sigma_f, sigma_y) and its
    length-scale entry is broadcast to all n_ell coordinates.  A box that
    already has n_ell + 2 entries is returned unchanged.
    """
    if theta_bounds_log is None:
        return None
    bounds = [(float(lo), float(hi)) for lo, hi in theta_bounds_log]
    if len(bounds) == n_ell + 2:
        return bounds
    if len(bounds) == 3:
        return [bounds[0]] * n_ell + [bounds[1], bounds[2]]
    raise ValueError(
        f"theta_bounds_log has {len(bounds)} entries; expected 3 (broadcast) "
        f"or {n_ell + 2} (one per length scale, then sigma_f, sigma_y)."
    )


def _expand_theta_vector(theta_log: Array, n_ell: int) -> Array:
    """Same broadcasting rule as _expand_theta_bounds, for a theta_log vector."""
    theta = np.asarray(theta_log, dtype=float).reshape(-1)
    if theta.size == n_ell + 2:
        return theta
    if theta.size == 3:
        return np.concatenate([np.full(n_ell, theta[0]), theta[1:]])
    raise ValueError(
        f"theta0_log has {theta.size} entries; expected 3 (broadcast) or "
        f"{n_ell + 2}."
    )


def _split_theta(theta_log: Array) -> Tuple[Array, float, float]:
    """theta_log -> (length scales, sigma_f, sigma_y) in natural units."""
    params = np.exp(np.asarray(theta_log, dtype=float).reshape(-1))
    if params.size < 3:
        raise ValueError("theta_log must hold at least [l_c, sigma_f, sigma_y].")
    return params[:-2], float(params[-2]), float(params[-1])


def _pack_length_scales(ell: Array) -> Union[float, Array]:
    """Store one length scale as a float, several as an array."""
    ell = np.asarray(ell, dtype=float).reshape(-1)
    return float(ell[0]) if ell.size == 1 else ell.copy()


# ----------------------------
# GP kernel + fit / prediction (Cholesky reuse)
# ----------------------------

def rbf_kernel_amp(X1: Array, X2: Array, l_c: Union[float, Array], sigma_f: float) -> Array:
    """
    Squared-exponential (RBF) kernel with amplitude.

    Isotropic (scalar l_c):
      k(x,x') = sigma_f^2 * exp(-||x-x'||^2 / (2 l_c^2))
    ARD (vector l_c of length d):
      k(x,x') = sigma_f^2 * exp(-0.5 * sum_i (x_i-x'_i)^2 / l_i^2)

    Assumes the latent function is infinitely differentiable.
    """
    ell = as_length_scales(l_c, np.shape(X1)[1] if np.ndim(X1) == 2 else None)
    if ell.size == 1:
        # Isotropic: the historical code path, untouched.
        gamma = 0.5 / (float(ell[0])**2)
        return (sigma_f**2) * rbf_kernel_(X1, X2, gamma=gamma)
    # ARD: rescale the inputs, then use the very same isotropic routine.
    Z1 = np.asarray(X1, dtype=float) / ell
    Z2 = np.asarray(X2, dtype=float) / ell
    return (sigma_f**2) * rbf_kernel_(Z1, Z2, gamma=0.5)


def matern52_kernel_amp(X1: Array, X2: Array, l_c: Union[float, Array],
                        sigma_f: float) -> Array:
    """
    Matérn ν=5/2 kernel with amplitude:
      k(x,x') = sigma_f^2 * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
    where r = ||x-x'||_2 for a scalar l, and r is the ARD-scaled distance
      r = sqrt( sum_i (x_i-x'_i)^2 / l_i^2 )   with l = 1
    when l_c is a vector.
    Assumes only twice differentiability — more realistic for physical simulations.
    """
    ell = as_length_scales(l_c, np.shape(X1)[1] if np.ndim(X1) == 2 else None)
    if ell.size == 1:
        # Isotropic: the historical code path, untouched.
        sqdist = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)
        r      = np.sqrt(np.maximum(sqdist, 0.0))   # numerical safety for r=0
        a      = np.sqrt(5.0) * r / float(ell[0])
    else:
        Z1 = np.asarray(X1, dtype=float) / ell
        Z2 = np.asarray(X2, dtype=float) / ell
        sqdist = np.sum((Z1[:, None, :] - Z2[None, :, :])**2, axis=2)
        a      = np.sqrt(5.0) * np.sqrt(np.maximum(sqdist, 0.0))
    return (sigma_f**2) * (1.0 + a + a**2 / 3.0) * np.exp(-a)


def kernel_amp(X1: Array, X2: Array, l_c: Union[float, Array], sigma_f: float,
               kind: str = "rbf") -> Array:
    """
    Dispatch to the selected kernel function.

    Parameters
    ----------
    kind : "rbf" or "matern52"
    """
    if kind == "rbf":
        return rbf_kernel_amp(X1, X2, l_c, sigma_f)
    elif kind == "matern52":
        return matern52_kernel_amp(X1, X2, l_c, sigma_f)
    else:
        raise ValueError(f"Unknown kernel '{kind}'. Choose 'rbf' or 'matern52'.")


def gp_fit(Xs, ys, l_c=0.3, sigma_f=1.0, sigma_y=0.1, jitter=1e-10, kernel="rbf"):
    """
    Compute Cholesky factor and alpha for GP regression.
    ys is assumed to be already normalised if normalize_y=True was set upstream.
    l_c is a scalar (isotropic) or a length-d array (ARD).
    """
    Xs = np.asarray(Xs, dtype=float)
    ys = np.asarray(ys, dtype=float).reshape(-1)

    Kss = kernel_amp(Xs, Xs, l_c=l_c, sigma_f=sigma_f, kind=kernel)
    n = Xs.shape[0]
    Ky = Kss + (sigma_y**2 + jitter) * np.eye(n)

    L = cholesky(Ky, lower=True)
    alpha = cho_solve((L, True), ys)
    return alpha, L


def gp_predict(X, Xs, alpha, L, l_c=0.3, sigma_f=1.0, return_cov=False, kernel="rbf"):
    """
    GP prediction using precomputed Cholesky factor.
    Returns mu and var/cov in the *normalised* space if normalize_y was used.
    Denormalisation is the responsibility of the caller (make_acquisition, plt_state).
    l_c must be the SAME scalar or (d,) length-scale specification used in gp_fit.
    """
    X = np.asarray(X, dtype=float)
    Xs = np.asarray(Xs, dtype=float)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)

    Ks = kernel_amp(X, Xs, l_c=l_c, sigma_f=sigma_f, kind=kernel)
    mu = Ks @ alpha

    V = cho_solve((L, True), Ks.T)

    if return_cov:
        Kxx = kernel_amp(X, X, l_c=l_c, sigma_f=sigma_f, kind=kernel)
        cov = Kxx - Ks @ V
        return mu, cov
    else:
        # For both RBF and Matérn kernels k(x,x) = sigma_f^2  (r=0 → factor=1)
        Kxx_diag = (sigma_f**2) * np.ones(X.shape[0])
        var = Kxx_diag - np.sum(Ks * V.T, axis=1)
        return mu, var


# ----------------------------
# Rank-1 Cholesky update (O(n^2) GP extension when one point is added)
# ----------------------------

def chol_rank1_update(L: Array, k_cross: Array, k_self: float,
                      sigma_y: float, jitter: float) -> Array:
    """
    Extend the n×n lower-triangular Cholesky factor L by one new training point.

    When a new observation (x_{n+1}, y_{n+1}) is added, the noisy kernel matrix
    grows from K_n to

        K_{n+1} = [[K_n,        k_cross    ],
                   [k_cross^T,  k_self + sigma_y^2 + jitter]]

    Its Cholesky factor has the block form

        L_{n+1} = [[L,   0  ],
                   [l^T, l_nn]]

    where
        l    = L^{-1} k_cross          (forward substitution, O(n^2))
        l_nn = sqrt(k_self + sigma_y^2 + jitter - l^T l)

    Overall cost: O(n^2)  vs  O(n^3) for a full recomputation.

    Parameters
    ----------
    L        : (n, n) lower-triangular Cholesky of the current noisy kernel
    k_cross  : (n,)  cross-covariance k(X_n, x_{n+1})
    k_self   : float noise-free self-covariance k(x_{n+1}, x_{n+1})
    sigma_y  : float observation noise std
    jitter   : float numerical jitter on the diagonal

    Returns
    -------
    L_new : (n+1, n+1) lower-triangular Cholesky of K_{n+1}
    """
    n = L.shape[0]
    # Forward substitution: solve L l = k_cross  (O(n^2))
    l = solve_triangular(L, k_cross, lower=True)
    # New diagonal entry; clip to jitter for numerical safety
    d = k_self + sigma_y**2 + jitter - float(np.dot(l, l))
    l_nn = np.sqrt(max(d, jitter))
    # Assemble extended factor
    L_new = np.zeros((n + 1, n + 1), dtype=float)
    L_new[:n, :n] = L
    L_new[n, :n]  = l
    L_new[n, n]   = l_nn
    return L_new


def alpha_rank1_update(L_old: Array, alpha_old: Array,
                       L_new: Array, y_new: float) -> Array:
    """
    Update the GP weight vector alpha = K_n^{-1} y_n after one new point is added.

    The augmented linear system K_{n+1} alpha_{n+1} = y_{n+1} is solved using
    the block structure of L_{n+1}:

      Forward solve  L_{n+1} v = y_{n+1}:
          v_1 = L^{-1} y_n  = L^T alpha_n          [O(n^2), no new solve needed]
          v_2 = (y_{n+1} - l^T v_1) / l_nn

      Backward solve  L_{n+1}^T alpha_{n+1} = v:
          alpha_{n+1}   = v_2 / l_nn
          alpha_{1:n}   = L^{-T}(v_1 - l * alpha_{n+1})   [O(n^2) triangular solve]

    Overall cost: O(n^2).

    Parameters
    ----------
    L_old     : (n, n) old lower-triangular Cholesky
    alpha_old : (n,)  old weight vector K_n^{-1} y_n
    L_new     : (n+1, n+1) new Cholesky (output of chol_rank1_update)
    y_new     : float new observation

    Returns
    -------
    alpha_new : (n+1,) updated weight vector K_{n+1}^{-1} y_{n+1}
    """
    n    = L_old.shape[0]
    l    = L_new[n, :n]   # new row (first n entries)
    l_nn = L_new[n, n]    # new diagonal entry

    # v_1 = L^T alpha_old  =  L^{-1} y_n  (O(n^2) mat-vec)
    v1 = L_old.T @ alpha_old

    # Scalar step for the new entry
    v2            = (y_new - float(np.dot(l, v1))) / l_nn
    alpha_new_last = v2 / l_nn

    # Backward solve for the first n entries:  L^T alpha_1 = v_1 - l * alpha_{n+1}
    rhs             = v1 - l * alpha_new_last
    alpha_new_first = solve_triangular(L_old, rhs, lower=True, trans='T')

    return np.concatenate([alpha_new_first, [alpha_new_last]])


def update_gp_rank1(gp: GPModel, x_new: Array, y_new: float) -> GPModel:
    """
    Extend a fitted GP by one observation using rank-1 Cholesky arithmetic.

    Replaces an O(n^3) full Cholesky recomputation with two O(n^2) triangular
    solves.  Requires that gp.L and gp.alpha are already populated (i.e. the GP
    has been fitted at least once via fit_gp) and that gp.Xs / gp.ys do NOT yet
    contain the new point (they are appended here).

    This update is valid only when the GP hyperparameters are unchanged.
    Whenever HPO modifies l_c, sigma_f, or sigma_y the kernel matrix changes
    globally and a full refit (fit_gp) is mandatory.

    Parameters
    ----------
    gp    : GPModel  current fitted GP (Xs, ys, L, alpha all set)
    x_new : (d,) or (1, d)  new input point
    y_new : float            new noisy observation

    Returns
    -------
    gp : GPModel  updated in place — Xs, ys, L, alpha extended by one row/entry
    """
    x_new   = np.asarray(x_new, dtype=float).reshape(1, -1)
    y_new_raw = float(y_new)

    # Normalise using the EXISTING statistics (do not recompute — that would
    # invalidate the Cholesky built on the previous normalisation).
    y_new_norm = (y_new_raw - gp.y_mean) / gp.y_std

    # Cross-covariance vector  k(X_n, x_{n+1}),  shape (n,)
    k_cross = kernel_amp(gp.Xs, x_new, gp.l_c, gp.sigma_f, kind=gp.kernel).reshape(-1)
    # Noise-free self-covariance  k(x_{n+1}, x_{n+1}),  scalar  (= sigma_f^2 for both kernels)
    k_self  = float(gp.sigma_f**2)

    # O(n^2) Cholesky extension
    L_new     = chol_rank1_update(gp.L, k_cross, k_self, gp.sigma_y, gp.jitter)
    # O(n^2) alpha update (operates in normalised space)
    alpha_new = alpha_rank1_update(gp.L, gp.alpha, L_new, y_new_norm)

    # Commit changes to the GP model
    gp.L     = L_new
    gp.alpha = alpha_new
    gp.Xs    = np.vstack([gp.Xs, x_new])
    gp.ys    = np.concatenate([gp.ys, [y_new_norm]])  # store normalised

    return gp


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
    kernel: str = "rbf",
) -> float:
    """
    theta_log = log([l_1, ..., l_n_ell, sigma_f, sigma_y])

    n_ell is inferred from the length of theta_log: 1 gives the isotropic
    kernel (the historical 3-parameter case), d gives ARD.  No extra argument
    is needed, so every existing caller keeps working unchanged.

    returns NLL = -log p(y | X, theta)
    y_train is assumed already normalised if normalize_y=True was used upstream.
    """
    ell, sigma_f, sigma_y = _split_theta(theta_log)
    l_c = _pack_length_scales(ell)

    K = kernel_amp(X_train, X_train, l_c=l_c, sigma_f=sigma_f, kind=kernel)
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
        # [l_1, ..., l_n_ell, sigma_f, sigma_y] -- for n_ell = 1 this is exactly
        # the historical [l_c, sigma_f, sigma_y] record.
        param_hist.append(
            [float(v) for v in ell] + [float(sigma_f), float(sigma_y)]
        )

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
    Update gp.(l_c, sigma_f, sigma_y) by minimising the negative log marginal
    likelihood, using a multi-restart L-BFGS-B strategy.

    Starting points (tried in this order):
      1. Warm start — gp.theta_log, i.e. the solution found at the previous
         BO iteration.  This exploits the fact that adding one point rarely
         shifts the optimum much, so the previous solution is a good initial
         guess and convergence is fast.
      2. User-specified gp_cfg.theta0_log, or the current hyperparameters
         converted to log-space if theta0_log is None.
      3. gp_cfg.n_hpo_restarts additional random starting points drawn from
         N(0, 1) in log-space, providing global coverage to escape local optima.

    The run with the lowest NLL is kept.  If MLE_hist / param_hist are provided,
    a final single-restart run from the winner is executed so that the history
    arrays are populated (these are used only for diagnostics / plotting).

    ARD
    ---
    With gp_cfg.ard=True the search runs over d+2 hyperparameters instead of 3.
    Starting points, bounds and the warm start are all widened accordingly; a
    3-entry theta0_log / theta_bounds_log is broadcast across the coordinates.
    A stale warm start of the wrong length (e.g. carried over from an isotropic
    fit) is discarded rather than reshaped.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    d = X.shape[1]
    # A vector l_c implies ARD for the purposes of HPO even when gp_cfg.ard is
    # False: refitting a single scalar over an anisotropic seed would silently
    # discard the anisotropy the caller declared.
    n_ell = d if (gp_cfg.ard or is_ard(gp.l_c)) else 1
    n_theta = n_ell + 2
    bounds = _expand_theta_bounds(gp_cfg.theta_bounds_log, n_ell)

    # ------------------------------------------------------------------
    # Build candidate starting points in log-space
    # ------------------------------------------------------------------
    x0_list: List[Array] = []

    # 1) warm start: last optimised theta (best single candidate).
    #    Only usable if it has the layout we are optimising now.
    if gp.theta_log is not None and np.size(gp.theta_log) == n_theta:
        x0_list.append(np.asarray(gp.theta_log, dtype=float).reshape(-1).copy())

    # 2) user-provided or current hyperparameters
    if gp_cfg.theta0_log is not None:
        x0_list.append(_expand_theta_vector(gp_cfg.theta0_log, n_ell))
    else:
        ell_now = as_length_scales(gp.l_c, d)
        if ell_now.size != n_ell:
            # scalar seed, ARD layout: broadcast it across the coordinates
            ell_now = np.full(n_ell, float(ell_now[0]))
        x0_list.append(np.log(np.concatenate([ell_now, [gp.sigma_f, gp.sigma_y]])))

    # 3) random restarts.  hpo_random_state=None reproduces the historical
    #    unseeded Generator; an integer makes the restarts reproducible.
    rng_hpo = np.random.default_rng(gp_cfg.hpo_random_state)
    for _ in range(max(0, gp_cfg.n_hpo_restarts)):
        x0 = rng_hpo.standard_normal(n_theta)
        if bounds is not None:
            # L-BFGS-B clips an out-of-box x0 internally; doing it here as well
            # is behaviourally identical and makes the restart meaningful when
            # the box is narrow (as it is for a bounded l_c floor).
            lo = np.array([b[0] for b in bounds], dtype=float)
            hi = np.array([b[1] for b in bounds], dtype=float)
            x0 = np.clip(x0, lo, hi)
        x0_list.append(x0)

    # ------------------------------------------------------------------
    # Run L-BFGS-B from every starting point; keep best NLL
    # ------------------------------------------------------------------
    best_nll: float = np.inf
    best_theta: Optional[Array] = None

    for x0 in x0_list:
        res = minimize(
            negative_log_marginal_likelihood,
            x0=x0,
            args=(X, y, gp.jitter, None, None, gp.kernel),  # no history for restart runs
            method="L-BFGS-B",
            bounds=bounds,
        )
        if np.isfinite(res.fun) and res.fun < best_nll:
            best_nll   = res.fun
            best_theta = res.x.copy()

    # Fallback (should never trigger, but guards against all-nan outcomes)
    if best_theta is None:
        best_theta = x0_list[0]

    # ------------------------------------------------------------------
    # Optionally re-run from the winner to populate diagnostic histories
    # ------------------------------------------------------------------
    if MLE_hist is not None or param_hist is not None:
        minimize(
            negative_log_marginal_likelihood,
            x0=best_theta,
            args=(X, y, gp.jitter, MLE_hist, param_hist, gp.kernel),
            method="L-BFGS-B",
            bounds=bounds,
        )

    # ------------------------------------------------------------------
    # Commit optimised hyperparameters and warm-start seed for next call
    # ------------------------------------------------------------------
    ell, sigma_f, sigma_y = _split_theta(best_theta)
    gp.l_c     = _pack_length_scales(ell)
    gp.sigma_f = float(sigma_f)
    gp.sigma_y = float(sigma_y)
    gp.theta_log = np.asarray(best_theta, dtype=float).reshape(-1).copy()

    return gp


def estimate_ard_length_scales(
    X: Array,
    y: Array,
    kernel: str = "matern52",
    theta_bounds_log: Optional[List[Tuple[float, float]]] = None,
    n_restarts: int = 8,
    normalize_y: bool = True,
    jitter: float = 1e-10,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fit an ARD kernel ONCE to an existing dataset and report what it found.

    This is the offline / screening entry point.  It is deliberately separate
    from the BO driver: the intended use is to measure the anisotropy of an
    objective on an archive of evaluations, and then either

      (a) declare the resulting length scales as a FROZEN kernel --
          GPConfig(l_c=result["length_scales"], optimize_hyperparams=False), or
      (b) rescale the inputs by result["whitening_weights"] and keep an
          isotropic kernel, which is algebraically the same model,

    without ever fitting d+2 hyperparameters inside the campaign itself.

    INPUTS:
      X, y             : (n, d) inputs and (n,) observations
      kernel           : "rbf" or "matern52"
      theta_bounds_log : HPO box; 3 entries are broadcast over the coordinates
      n_restarts       : random restarts on top of the default start
      normalize_y      : centre/scale y before fitting (recommended)
      random_state     : seed for the restarts; set it for a reproducible fit

    OUTPUT: dict with
      length_scales     : (d,) fitted length scales
      sigma_f, sigma_y  : amplitude and noise std (normalised units if
                          normalize_y, else the units of y)
      nll               : negative log marginal likelihood at the optimum
      theta_log         : the raw optimised log-vector
      spread            : max(l) / min(l), the headline anisotropy number
      relevance_order   : coordinate indices sorted from shortest to longest
                          length scale, i.e. most to least relevant
      whitening_weights : 1 / l normalised to unit geometric mean
      y_mean, y_std     : normalisation statistics actually applied
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size or y.size == 0:
        raise ValueError("X must be (n, d) and y must be (n,), with n > 0.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must be finite.")

    gp_cfg = GPConfig(
        l_c=1.0,
        sigma_f=1.0,
        sigma_y=0.1,
        jitter=jitter,
        optimize_hyperparams=True,
        hpo_every=1,
        n_hpo_restarts=int(n_restarts),
        kernel=kernel,
        normalize_y=bool(normalize_y),
        theta_bounds_log=theta_bounds_log,
        ard=True,
        hpo_random_state=random_state,
    )
    gp = fit_gp(build_gp_model(gp_cfg), X, y, gp_cfg, it=0)

    ell = as_length_scales(gp.l_c, X.shape[1])
    nll = negative_log_marginal_likelihood(
        gp.theta_log, gp.Xs, gp.ys, jitter=jitter, kernel=kernel
    )
    return dict(
        length_scales=ell.copy(),
        sigma_f=float(gp.sigma_f),
        sigma_y=float(gp.sigma_y),
        nll=float(nll),
        theta_log=np.asarray(gp.theta_log, dtype=float).copy(),
        spread=float(np.max(ell) / np.min(ell)),
        relevance_order=np.argsort(ell),
        whitening_weights=ard_whitening_weights(ell),
        y_mean=float(gp.y_mean),
        y_std=float(gp.y_std),
    )


# ----------------------------
# GP wrapper used by BO
# ----------------------------

def build_gp_model(gp_cfg: GPConfig) -> GPModel:
    """
    Create an unfitted GP container.

    A vector l_c is copied rather than referenced, so mutating the model never
    reaches back into the configuration object.
    """
    return GPModel(
        l_c=_pack_length_scales(as_length_scales(gp_cfg.l_c)),
        sigma_f=gp_cfg.sigma_f,
        sigma_y=gp_cfg.sigma_y,
        jitter=gp_cfg.jitter,
        kernel=gp_cfg.kernel,
    )


def fit_gp(gp: GPModel, X: Array, y: Array, gp_cfg: GPConfig, it: int) -> GPModel:
    """
    Fit GP and store Cholesky + alpha in the container.

    If gp_cfg.normalize_y=True, y is centred and scaled before fitting; the
    normalisation statistics are stored in gp.y_mean / gp.y_std and used by
    make_acquisition and plt_state to recover predictions in original units.

    If gp_cfg.optimize_hyperparams=True, HPO runs every gp_cfg.hpo_every
    iterations on the (already normalised) gp.ys.

    ARD bookkeeping: the input dimension is only known here, so this is where a
    scalar l_c is expanded to d length scales when gp_cfg.ard is set, and where
    a vector l_c is checked against the data.
    """
    gp.Xs = np.asarray(X, dtype=float)
    y_raw = np.asarray(y, dtype=float).reshape(-1)

    # ------------------------------------------------------------------
    # Resolve the length-scale layout against the data
    # ------------------------------------------------------------------
    d = gp.Xs.shape[1]
    ell = as_length_scales(gp.l_c, d)          # raises on a d-mismatch
    if gp_cfg.ard and ell.size == 1:
        ell = np.full(d, float(ell[0]))        # scalar seed -> one scale per coord
    gp.l_c = _pack_length_scales(ell)

    # ------------------------------------------------------------------
    # Output normalisation (optional)
    # ------------------------------------------------------------------
    if gp_cfg.normalize_y:
        y_mean = float(np.mean(y_raw))
        y_std  = float(np.std(y_raw))
        if y_std < 1e-12:
            y_std = 1.0          # guard against constant outputs
        gp.y_mean = y_mean
        gp.y_std  = y_std
        gp.ys = (y_raw - y_mean) / y_std
    else:
        gp.y_mean = 0.0
        gp.y_std  = 1.0
        gp.ys = y_raw

    # ------------------------------------------------------------------
    # Hyperparameter optimisation (on normalised gp.ys)
    # ------------------------------------------------------------------
    do_hpo = gp_cfg.optimize_hyperparams and (gp_cfg.hpo_every > 0) and (it % gp_cfg.hpo_every == 0)
    if do_hpo:
        gp = optimize_gp_hyperparams(gp.Xs, gp.ys, gp, gp_cfg)

    # ------------------------------------------------------------------
    # Cholesky fit with current hyperparameters (fixed or optimised)
    # ------------------------------------------------------------------
    gp.alpha, gp.L = gp_fit(
        gp.Xs, gp.ys,
        gp.l_c, gp.sigma_f, gp.sigma_y, gp.jitter,
        kernel=gp.kernel,
    )
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

        # GP predictive mean and variance (latent function, normalised space)
        mu_norm, var_norm = gp_predict(
            Xcand, gp.Xs, gp.alpha, gp.L,
            l_c=gp.l_c, sigma_f=gp.sigma_f,
            return_cov=False, kernel=gp.kernel,
        )
        # Denormalise to original y scale
        mu  = mu_norm  * gp.y_std + gp.y_mean
        var = var_norm * gp.y_std**2
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
        return AcqOptimizationResult(
        x_next=x_best,
        a_best=a_best,
        Xcand=Xcand,
        a=a,
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

    return AcqOptimizationResult(
        x_next=x_best,
        a_best=a_best,
        Xcand=Xcand,
        a=a,
    )


def optimize_acquisition_batch(
    acq: Callable[[Array], Array],
    bounds: Bounds,
    optim_cfg: OptimConfig,
    rng: np.random.Generator,
) -> List[AcqOptimizationResult]:
    """Greedily select a diverse batch by locally penalizing previous picks."""
    n_candidates = int(optim_cfg.n_candidates)
    if n_candidates < 1:
        raise ValueError("optim_cfg.n_candidates must be at least one.")
    if optim_cfg.batch_distance_scale <= 0.0:
        raise ValueError("optim_cfg.batch_distance_scale must be positive.")

    lo = np.asarray([bound[0] for bound in bounds], dtype=float)
    width = np.asarray([bound[1] - bound[0] for bound in bounds], dtype=float)
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
    """Projected ADAM in normalized parameter space."""
    lo = np.asarray([bound[0] for bound in bounds], dtype=float)
    width = np.asarray([bound[1] - bound[0] for bound in bounds], dtype=float)
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
            z
            - config.learning_rate
            * first_hat
            / (np.sqrt(second_hat) + config.epsilon),
            0.0,
            1.0,
        )
    return lo + width * z

def export_state(state: BOState, path_or_handler: Any) -> None:
    """Optional exporting."""
    return


def plt_state(
    gp: GPModel,
    acq_res,
    f: Callable[[Array], Array],
    bounds: Bounds,
    it: int,
    save_cfg: SaveConfig,
    acq_cfg: AcqConfig,
    n_plot: int = 400,
):
    """
    Plot current BO state (1D only).

    Top:
      - noisy true function (single realization)
      - observations
      - GP posterior mean + 95% CI

    Bottom:
      - acquisition values on candidate set
    """
    # Safety checks
    if gp.Xs is None or gp.ys is None:
        return

    if gp.Xs.shape[1] != 1:
        # plotting only supported in 1D
        return

    # Build dense grid
    x_min, x_max = bounds[0]
    Xplot = np.linspace(x_min, x_max, n_plot).reshape(-1, 1)

    # True (noisy) function
    y_true = np.asarray(f(Xplot, noise_level=0)).reshape(-1)

    # GP posterior (normalised space → denormalise before plotting)
    mu_norm, var_norm = gp_predict(
        Xplot, gp.Xs, gp.alpha, gp.L,
        l_c=gp.l_c, sigma_f=gp.sigma_f,
        return_cov=False, kernel=gp.kernel,
    )
    mu  = (mu_norm  * gp.y_std + gp.y_mean).reshape(-1)
    std = (np.sqrt(np.maximum(var_norm, 0.0)) * gp.y_std).reshape(-1)

    # Observations — stored normalised; recover original scale for the scatter
    Xs = gp.Xs.reshape(-1)
    ys = (gp.ys * gp.y_std + gp.y_mean).reshape(-1)

    # Acquisition values
    Xcand = acq_res.Xcand.reshape(-1)
    a = acq_res.a.reshape(-1)

    idx = np.argsort(Xcand)
    Xcand = Xcand[idx]
    a = a[idx]

    # Plot
    fig, axs = plt.subplots(
        2, 1, figsize=(6, 5),
        constrained_layout=True,
        sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1])
    )

    # ---- Top: function + GP
    axs[0].plot(Xplot[:, 0], y_true, "k--", lw=1.0, label="True (unknown)")
    axs[0].plot(Xplot[:, 0], mu, "C0", lw=2, label="$\\mu_{\\mathcal{GP}}$")
    axs[0].fill_between(
        Xplot[:, 0],
        mu - 2 * std,
        mu + 2 * std,
        color="C0",
        alpha=0.25,
        label="$\\mu_{\\mathcal{GP}} \\pm 2 \\sigma$",
    )
    axs[0].scatter(Xs, ys, c="k", s=20, zorder=10, label="Observations")

    axs[0].set_ylabel("f(x)")
    axs[0].set_title(f"Iteration {it}")
    axs[0].legend(fontsize=8, loc='upper right')

    # ---- Bottom: acquisition
    axs[1].plot(Xcand, a, "C1", lw=1.5, label=acq_cfg.kind)
    axs[1].fill_between(
        x= Xcand,
        y1= a,
        color= "C1",
        alpha= 0.2
    )
    axs[1].scatter(acq_res.x_next, acq_res.a_best, c="C1", label='Next query point')
    axs[1].legend(fontsize=8, loc='lower right')

    axs[1].set_ylabel("acq(x)")
    axs[1].set_xlabel("x")

    if save_cfg.save_path:
        if not os.path.exists(save_cfg.save_path):
            os.makedirs(save_cfg.save_path)
        figname = os.path.join(save_cfg.save_path, f"it_{it:03d}.png")
        plt.savefig(figname, dpi=250)
        plt.close(fig)
    else:
        plt.show()


def setup_logger(log_path: str, filename: str = "run.log") -> logging.Logger:
    os.makedirs(log_path, exist_ok=True)

    logger = logging.getLogger("BO")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(
        os.path.join(log_path, filename),
        mode="w"
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

# ----------------------------
# Per-iteration state export
# ----------------------------
def _export_iteration_data(
    gp: GPModel,
    acq_res: AcqOptimizationResult,
    X: Array,
    y: Array,
    x_next: Array,
    y_next: float,
    bounds: Bounds,
    it: int,
    save_path: str,
    n_plot: int = 400,
) -> None:
    """
    Save a compressed .npz snapshot of the current BO state.

    Always saved:
      it, X, y, x_next, y_next, Xcand, a, a_best, l_c, sigma_f, sigma_y,
      y_mean, y_std

    l_c is always stored as a 1-D array: length 1 for an isotropic kernel and
    length d under ARD.

    Additionally for 1D problems:
      Xplot, mu, std  (dense GP posterior on a regular grid — useful for animation)

    Files are written to  <save_path>/states/state_NNN.npz.
    Use make_animation_1D.py to build a GIF from these snapshots.
    """
    states_dir = os.path.join(save_path, "states")
    os.makedirs(states_dir, exist_ok=True)

    data: Dict[str, Any] = dict(
        it      = np.array([it]),
        X       = X,
        y       = y,
        x_next  = np.asarray(x_next).reshape(-1),
        y_next  = np.array([y_next]),
        Xcand   = acq_res.Xcand,
        a       = acq_res.a,
        a_best  = np.array([acq_res.a_best]),
        # (1,) for an isotropic kernel, (d,) under ARD
        l_c     = np.atleast_1d(np.asarray(gp.l_c, dtype=float)),
        sigma_f = np.array([gp.sigma_f]),
        sigma_y = np.array([gp.sigma_y]),
        y_mean  = np.array([gp.y_mean]),
        y_std   = np.array([gp.y_std]),
    )

    # 1D-only: export dense GP posterior for animation
    if gp.Xs is not None and gp.Xs.shape[1] == 1:
        x_lo, x_hi = bounds[0]
        Xplot = np.linspace(x_lo, x_hi, n_plot).reshape(-1, 1)
        mu_norm, var_norm = gp_predict(
            Xplot, gp.Xs, gp.alpha, gp.L,
            l_c=gp.l_c, sigma_f=gp.sigma_f,
            return_cov=False, kernel=gp.kernel,
        )
        data['Xplot'] = Xplot.reshape(-1)
        data['mu']    = (mu_norm  * gp.y_std + gp.y_mean).reshape(-1)
        data['std']   = (np.sqrt(np.maximum(var_norm, 0.0)) * gp.y_std).reshape(-1)

    fname = os.path.join(states_dir, f"state_{it:03d}.npz")
    np.savez_compressed(fname, **data)


# ----------------------------
# Main BO function (skeleton)
# ----------------------------
def bayesian_optimization(
    f: Callable[[Array], Union[float, Array]],
    bounds: Bounds,
    bo_cfg: BOConfig,
    gp_cfg: GPConfig,
    acq_cfg: AcqConfig,
    optim_cfg: OptimConfig,
    save_cfg: SaveConfig,
    callbacks: Optional[List[Callable[[BOState], None]]] = None,
    exporter: Optional[Any] = None,
    X_init: Optional[Array] = None,  # initial points (optional)
    y_init: Optional[Array] = None,  # initial values (optional)
    top_up_to_n_init: bool = True,   # if X_init has too few points, add more random ones
    states: Optional[List[BOState]] = None,
    gradient: Optional[Callable[[Array], Array]] = None,
    refinement_cfg: Optional[GradientRefinementConfig] = None,
) -> BOResult:
    """
    Bayesian Optimization driver (minimization by default).

    ``gradient`` and ``refinement_cfg`` are optional. The legacy behavior is
    unchanged unless ``refinement_cfg.enabled`` is true. Batch acquisition is
    controlled independently by ``optim_cfg.n_candidates``.
    """
    rng = _rng(bo_cfg.random_state)
    callbacks = callbacks or []
    states = [] if states is None else states
    refinement_cfg = refinement_cfg or GradientRefinementConfig()
    if refinement_cfg.enabled and gradient is None:
        raise ValueError(
            "gradient must be provided when gradient refinement is enabled."
        )
    if refinement_cfg.close_pair_policy not in {"final", "midpoint"}:
        raise ValueError("close_pair_policy must be 'final' or 'midpoint'.")
    if refinement_cfg.distance_threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative.")
    if refinement_cfg.n_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    if refinement_cfg.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if not (0.0 <= refinement_cfg.beta1 < 1.0):
        raise ValueError("beta1 must lie in [0, 1).")
    if not (0.0 <= refinement_cfg.beta2 < 1.0):
        raise ValueError("beta2 must lie in [0, 1).")

    logger = None
    if save_cfg.log_enabled and save_cfg.save_path:
        logger = setup_logger(
            save_cfg.save_path,
            save_cfg.log_filename
        )

        logger.info("Starting Bayesian Optimization")
        logger.info(f"Bounds: {bounds}")
        logger.info(f"BOConfig: {bo_cfg}")
        logger.info(f"GPConfig: {gp_cfg}")
        logger.info(f"AcqConfig: {acq_cfg}")
        logger.info(f"OptimConfig: {optim_cfg}")
        logger.info(f"GradientRefinementConfig: {refinement_cfg}")
        logger.info(f"Random seed: {bo_cfg.random_state}")



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
    gp = build_gp_model(gp_cfg)

    history: List[Dict[str, Any]] = []

    # ---- 3) BO loop
    for it in range(bo_cfg.n_iter):

        # Decide whether to run HPO this iteration
        do_hpo = (
            gp_cfg.optimize_hyperparams
            and (gp_cfg.hpo_every > 0)
            and (it % gp_cfg.hpo_every == 0)
        )

        # Fit / update the GP on current data.
        #
        # Three paths:
        #   (a) it == 0 or HPO triggered  →  full O(n^3) Cholesky refit.
        #       HPO changes the kernel hyperparameters, so the whole K matrix
        #       changes; a rank-1 update would be invalid.
        #   (b) dataset size >= rank1_threshold and no HPO  →  O(n^2) extension
        #       appending only the single point added at iteration it-1.
        #       gp.Xs / gp.ys still hold the OLD dataset (n points); X / y
        #       already have n+1 points, so X[-1] / y[-1] is the new point.
        #   (c) rank-1 not yet triggered or disabled  →  full refit without HPO.
        n_pending = 0 if gp.Xs is None else len(X) - len(gp.Xs)
        use_rank1 = (
            gp_cfg.rank1_threshold > 0
            and len(y) >= gp_cfg.rank1_threshold
            and gp.L is not None
            and gp.Xs is not None
            and n_pending >= 1
        )
        if it == 0 or do_hpo:
            # Path (a): mandatory full refit
            gp = fit_gp(gp, X, y, gp_cfg, it)
        elif use_rank1:
            # Path (b): cheap O(n^2) rank-1 Cholesky extensions. Hybrid
            # refinement can append both the proposal and its ADAM endpoint.
            # Extend sequentially so that this remains valid for either one or
            # several observations added by the preceding BO round.
            for pending_index in range(len(X) - n_pending, len(X)):
                gp = update_gp_rank1(
                    gp, X[pending_index : pending_index + 1], float(y[pending_index])
                )
        else:
            # Path (c): full refit, HPO skipped by fit_gp based on iteration parity
            gp = fit_gp(gp, X, y, gp_cfg, it)

        # Decide current best (minimization)
        best_idx = int(np.argmin(y))
        y_best = float(y[best_idx])
        x_best = X[best_idx].copy()

        # Build acquisition
        acq = make_acquisition(acq_cfg, gp, y_best)

        # Propose a diverse acquisition batch. n_candidates=1 reproduces the
        # historical single-EI-point behavior.
        acq_results = optimize_acquisition_batch(acq, bounds, optim_cfg, rng)
        acq_res = acq_results[0]
        x_proposed = np.asarray(
            [result.x_next for result in acq_results], dtype=float
        )
        y_proposed = np.asarray(
            [float(np.asarray(f(x)).reshape(-1)[0]) for x in x_proposed],
            dtype=float,
        )

        x_refined = np.full_like(x_proposed, np.nan)
        y_refined = np.full(len(x_proposed), np.nan, dtype=float)
        displacements = np.zeros(len(x_proposed), dtype=float)
        round_X: List[Array] = []
        round_y: List[float] = []
        lo = np.asarray([bound[0] for bound in bounds], dtype=float)
        width = np.asarray([bound[1] - bound[0] for bound in bounds], dtype=float)

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
            xL = adam_refine_candidate(
                x0, local_gradient, bounds, refinement_cfg
            )
            yL = float(np.asarray(f(xL)).reshape(-1)[0])
            displacement = float(
                np.linalg.norm((xL - x0) / width)
            )
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
        x_next = x_proposed[0]
        y_next = float(y_proposed[0])

        # Export per-iteration snapshot for post-processing / animation
        if save_cfg.export_states and save_cfg.save_path:
            _export_iteration_data(
                gp=gp, acq_res=acq_res,
                X=X, y=y,
                x_next=x_next, y_next=y_next,
                bounds=bounds, it=it,
                save_path=save_cfg.save_path,
                n_plot=save_cfg.n_plot,
            )

        # Inline plot (interactive / legacy — use export_states for scripts)
        if save_cfg.plt_state_enabled:
            plt_state(
                gp=gp,
                acq_res=acq_res,
                f=f,
                bounds=bounds,
                it=it,
                save_cfg=save_cfg,
                acq_cfg=acq_cfg
            )

        # Append all non-redundant proposed/refined observations.
        X = np.vstack([X, round_X_array])
        y = np.concatenate([y, round_y_array])

        # Store all what was evaluated at this iteration in the state
        state = BOState(
            it=it,
            X=X.copy(),
            y=y.copy(),
            gp=gp,
            acq=acq,
            acq_res=acq_res,
            x_next=x_next,
            y_next=y_next,
            x_proposed=x_proposed.copy(),
            y_proposed=y_proposed.copy(),
            x_refined=x_refined.copy(),
            y_refined=y_refined.copy(),
            refinement_displacement=displacements.copy(),
        )
        # Add the current state in the save list
        states.append(state)

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
                n=len(y),
                l_c=(np.array(gp.l_c, dtype=float, copy=True)
                     if np.ndim(gp.l_c) > 0 else gp.l_c),
                sigma_f=gp.sigma_f,
                sigma_y=gp.sigma_y,
            )
        )

        # Callbacks
        for cb in callbacks:
            cb(state)

        # Optional exporting
        if exporter is not None and bo_cfg.export_every is not None:
            if (it + 1) % bo_cfg.export_every == 0:
                export_state(state, exporter)

        if save_cfg.log_enabled and save_cfg.save_path:
            logger.info(
                f"it={it:03d} | "
                f"x_next={x_next} | "
                f"y_next={y_next:.6e} | "
                f"n_proposed={len(x_proposed)} | "
                f"n_added={len(round_y_array)} | "
                f"best_y={y_best:.6e} | "
                f"l_c={format_length_scales(gp.l_c)} | "
                f"sigma_f={gp.sigma_f:.3e} | "
                f"sigma_y={gp.sigma_y:.3e}"
            )

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
        states=states
    )
