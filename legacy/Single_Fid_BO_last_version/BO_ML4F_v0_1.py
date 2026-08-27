"""
Bayesian Optimization framework.

Within the research of the Machine Learning for Fluid Systems group (ML4F)

link: https://www.mendezma.com/

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations 
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Tuple, Union

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import logging
from scipy.linalg import solve_triangular
import time
import h5py
import json
from dataclasses import asdict
from datetime import datetime

#Customization of the plot 
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=12)

Array = np.ndarray  # short cut for numpy arrays
# short cut for domain bounds as a list
Bounds = List[Tuple[float, float]]  # [(l1,u1), ..., (ld,ud)]


#%%  Settings / Config containers

# these are dataclasses that makes it simple to access all the settings.

# general settigns of the GP (hyperparameters etc)
@dataclass
class GPConfig:
    # if optimize_hyperparams=False, these values are used and kept fixed
    l_c: float = 0.3          # length-scale ell
    sigma_f: float = 1.0      # kernel amplitude std (so variance is sigma_f^2)
    sigma_y: float = 0.1      # observation noise std
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
    # Logging (text)
    log_enabled: bool = True
    log_filename: str = "log.log"
    # Reproducibility checkpointing (HDF5)
    export_enabled: bool = False
    export_filename: str = "experiment.h5"
    export_every: int = 1   # save each iteration

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
class GPModel:
    l_c: float
    sigma_f: float
    sigma_y: float
    jitter: float

    Xs: Optional[Array] = None     # training inputs
    ys: Optional[Array] = None     # training outputs
    alpha: Optional[Array] = None  # (Kss + σ²I)^(-1) y
    L: Optional[Array] = None      # Cholesky factor

    # optional: store last optimized theta in log-space
    theta_log: Optional[Array] = None


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


def gp_predict(X, Xs, alpha, L, l_c=0.3, sigma_f=1.0, return_cov=False):
    """
    GP prediction using precomputed Cholesky factor.
    """
    X = np.asarray(X, dtype=float)
    Xs = np.asarray(Xs, dtype=float)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)

    Ks = rbf_kernel_amp(X, Xs, l_c=l_c, sigma_f=sigma_f)
    mu = Ks @ alpha

    V = cho_solve((L, True), Ks.T)

    if return_cov:
        Kxx = rbf_kernel_amp(X, X, l_c=l_c, sigma_f=sigma_f)
        cov = Kxx - Ks @ V
        return mu, cov
    else:
        # diag(Kxx) = sigma_f^2
        Kxx_diag = (sigma_f**2) * np.ones(X.shape[0])
        var = Kxx_diag - np.sum(Ks * V.T, axis=1)
        return mu, var


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

def build_gp_model(gp_cfg: GPConfig) -> GPModel:
    """Create an unfitted GP container."""
    return GPModel(
        l_c=gp_cfg.l_c,
        sigma_f=gp_cfg.sigma_f,
        sigma_y=gp_cfg.sigma_y,
        jitter=gp_cfg.jitter,
    )


# def fit_gp(gp: GPModel, X: Array, y: Array, gp_cfg: GPConfig, it: int) -> GPModel:
#     """
#     Fit GP and store Cholesky + alpha in the container.

#     If gp_cfg.optimize_hyperparams=True, we run HPO (log-marginal likelihood)
#     every gp_cfg.hpo_every iterations.
#     """
#     gp.Xs = np.asarray(X, dtype=float)
#     gp.ys = np.asarray(y, dtype=float).reshape(-1)

#     # decide whether to do HPO at this iteration
#     do_hpo = gp_cfg.optimize_hyperparams and (gp_cfg.hpo_every > 0) and (it % gp_cfg.hpo_every == 0)
#     if do_hpo:
#         gp = optimize_gp_hyperparams(gp.Xs, gp.ys, gp, gp_cfg)

#     # fit with current hyperparameters (fixed or optimized)
#     gp.alpha, gp.L = gp_fit(gp.Xs, gp.ys, gp.l_c, gp.sigma_f, gp.sigma_y, gp.jitter)
#     return gp


def fit_gp(gp: GPModel, X: Array, y: Array, gp_cfg: GPConfig, it: int) -> GPModel:

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Decide whether to run hyperparameter optimization
    do_hpo = (
        gp_cfg.optimize_hyperparams
        and (gp_cfg.hpo_every > 0)
        and (it % gp_cfg.hpo_every == 0)
    )

    if do_hpo:
        gp = optimize_gp_hyperparams(X, y, gp, gp_cfg)

    # Decide whether we can use rank-one update
    can_rank1 = (
        gp_cfg.rank_one
        and not do_hpo
        and gp.L is not None
        and gp.Xs is not None
        and len(X) == len(gp.Xs) + 1
        and X.shape[0]>gp_cfg.rank_one_threshold
    )
    
    if can_rank1:

        # Use rank-one update
        x_new = X[-1]
        y_new = y[-1]
        gp = _rank_one_update(gp, x_new, y_new)

        # update stored dataset
        gp.Xs = X
        gp.ys = y

    else:
        # Full recompute
        gp.Xs = X
        gp.ys = y
        gp.alpha, gp.L = gp_fit(
            gp.Xs,
            gp.ys,
            gp.l_c,
            gp.sigma_f,
            gp.sigma_y,
            gp.jitter,
        )

    return gp


def _rank_one_update(gp: GPModel, x_new: Array, y_new: float) -> GPModel:

    X_old = gp.Xs
    L_old = gp.L
    y_old = gp.ys

    # ---- compute cross kernel vector
    k = rbf_kernel_amp(
        X_old,
        x_new.reshape(1, -1),
        l_c=gp.l_c,
        sigma_f=gp.sigma_f
    ).reshape(-1)

    # ---- compute diagonal term
    k_nn = rbf_kernel_amp(
        x_new.reshape(1, -1),
        x_new.reshape(1, -1),
        l_c=gp.l_c,
        sigma_f=gp.sigma_f
    )[0, 0] + gp.sigma_y**2 + gp.jitter

    # ---- solve L v = k
    v = solve_triangular(L_old, k, lower=True)

    # ---- new diagonal element
    val = k_nn - np.dot(v, v)
    val = max(val, 1e-14)   # numerical safety
    diag_new = np.sqrt(val)

    # ---- build new L
    n = L_old.shape[0]
    L_new = np.zeros((n+1, n+1))
    L_new[:n, :n] = L_old
    L_new[n, :n] = v
    L_new[n, n] = diag_new

    # ---- update alpha
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
            return_cov=False
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


#%% Save and export data



def setup_experiment_folder(save_cfg: SaveConfig) -> Optional[str]:

    if save_cfg.out_path is None:
        return None

    os.makedirs(save_cfg.out_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_path = os.path.join(save_cfg.out_path, timestamp)
    os.makedirs(exp_path, exist_ok=False)

    # Erase the out_path with the exp one 
    save_cfg.out_path = exp_path
    return exp_path



def init_exporter(
        bo_cfg: BOConfig,
        gp_cfg: GPConfig,
        acq_cfg: AcqConfig,
        optim_cfg: OptimConfig,
        save_cfg: SaveConfig,
        rng: np.random.Generator
    ):

    os.makedirs(save_cfg.out_path, exist_ok=True)
    path = os.path.join(save_cfg.out_path, save_cfg.export_filename)

    f = h5py.File(path, "w")

    meta = f.create_group("metadata")
    meta.attrs["bo_config"] = json.dumps(bo_cfg.__dict__)
    meta.attrs["gp_config"] = json.dumps(gp_cfg.__dict__)
    meta.attrs["acq_config"] = json.dumps(acq_cfg.__dict__)
    meta.attrs["optim_config"] = json.dumps(optim_cfg.__dict__)
    meta.attrs["save_config"] = json.dumps(save_cfg.__dict__)
    meta.attrs["random_seed"] = bo_cfg.random_state

    f.create_group("iterations")

    return f

def export_state(h5file, state: BOState):

    g = h5file["iterations"].create_group(f"{state.it:03d}")

    g.create_dataset("X", data=state.X)
    g.create_dataset("y", data=state.y)
    g.create_dataset("x_next", data=state.x_next)
    g.attrs["y_next"] = state.y_next

    # GP params
    gp_g = g.create_group("gp")
    
    for k, v in asdict(state.gp).items():
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            gp_g.create_dataset(k, data=v)
        else:
            gp_g.attrs[k] = v
            
            
    # Acquisition results params
    acq_res_g = g.create_group("acq_res")
    
    for k, v in asdict(state.acq_res).items():
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            acq_res_g.create_dataset(k, data=v)
        else:
            acq_res_g.attrs[k] = v
    # g.attrs["rng_state"] = json.dumps(state.rng_state)


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



def log_update(logger, state: BOState) -> None:

    # clean the format of n_next
    x_scalar = float(state.x_next[0]) if np.ndim(state.x_next) > 0 else float(state.x_next)
    
    logger.info(
        f"{state.it:03d} | "
        f"{x_scalar:>14.6e} | "
        f"{state.y_next:>14.6e} | "
        f"{state.y_best:>14.6e} | "
        f"{state.gp.l_c:>12.3e} | "
        f"{state.gp.sigma_f:>12.3e} | "
        f"{state.gp.sigma_y:>12.3e}"
    )


#%% Plot functions



def setup_plotting_toggle(save_cfg:SaveConfig):
    
    if save_cfg.plt_all:
        save_cfg.plt_state_enabled     = True
        save_cfg.plt_conv_enabled      = True
        save_cfg.plt_hist_enabled      = True
        save_cfg.plt_MLE_conv_enbable  = True

    
def plt_state(
    gp: GPModel,
    acq_res,
    f: Callable[[Array], Array],
    bounds: Bounds,
    it: int,
    x_best: Array,
    y_best: float, 
    save_cfg: SaveConfig,
    acq_cfg: AcqConfig,
    bo_cfg: BOConfig,
    n_plot: int = 400,
    f_true: Optional[Callable[[Array], Array]] = None,
):
    """
    Dispatcher for plotting depending on input dimension.
    """

    # Safety checks
    if gp.Xs is None or gp.ys is None:
        return

    d = gp.Xs.shape[1]

    if d == 1:
        plt_state_1D(
            gp, acq_res, f, bounds, it,
            x_best, y_best, save_cfg,
            acq_cfg, bo_cfg, n_plot=n_plot,
            f_true=f_true
        )

    elif d == 2:
        plt_state_2D(
            gp, acq_res, f, bounds, it,
            x_best, y_best, save_cfg,
            acq_cfg, bo_cfg, n_plot=n_plot,
            f_true=f_true
        )

    else:
        # Graceful fallback
        print(f"[plt_state] Plot not supported for dimension d={d}. Skipping.")
        
        
        
def plt_state_1D(
    gp: GPModel,
    acq_res,
    f: Callable[[Array], Array],
    bounds: Bounds,
    it: int,
    x_best: Array,
    y_best: float,
    save_cfg: SaveConfig,
    acq_cfg: AcqConfig, 
    bo_cfg: BOConfig,
    n_plot: int = 400,
    f_true: Optional[Callable[[Array], Array]] = None
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
    if f_true is not None:
        y_true = np.asarray(f_true(Xplot)).reshape(-1)
    else:
        y_true = None
        
    # GP posterior
    mu, var = gp_predict(
        Xplot, gp.Xs, gp.alpha, gp.L,
        l_c=gp.l_c, sigma_f=gp.sigma_f,
        return_cov=False
    ) 
    mu = mu.reshape(-1)
    std = np.sqrt(var.reshape(-1))

    # Observations
    Xs = gp.Xs.reshape(-1)
    ys = gp.ys.reshape(-1)

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

    # Top: function + GP
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
    axs[0].scatter(x_best, y_best, c="*", s=20, zorder=10, label="Best observed")

    axs[0].set_ylabel("f(x)")
    axs[0].set_title(f"Iteration {it}/{bo_cfg.n_iter}")
    axs[0].legend(fontsize=8, loc='upper right')

    #  Bottom: acquisition
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

    if save_cfg.out_path:
        fig_path = os.path.join(save_cfg.out_path, "GIF")
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        figname = os.path.join(fig_path, f"it_{it:03d}.png")
        plt.savefig(figname, dpi=250, bbox_inches='tight')
    plt.show()
    plt.close()

    
    
    
def plt_state_2D(
    gp: GPModel,
    acq_res,
    f: Callable[[Array], Array],
    bounds: Bounds,
    it: int,
    x_best: Array,
    y_best: float,
    save_cfg: SaveConfig,
    acq_cfg: AcqConfig,
    bo_cfg: BOConfig,
    n_plot: int = 50,
    f_true: Optional[Callable[[Array], Array]] = None,
):
    """
    2D visualization:
    - GP mean (contour)
    - Observations
    - Next query point
    """

    # Grid
    x1 = np.linspace(bounds[0][0], bounds[0][1], n_plot)
    x2 = np.linspace(bounds[1][0], bounds[1][1], n_plot)
    X1, X2 = np.meshgrid(x1, x2)

    Xgrid = np.stack([X1.ravel(), X2.ravel()], axis=1)

    # GP prediction
    mu, var = gp_predict(
        Xgrid, gp.Xs, gp.alpha, gp.L,
        l_c=gp.l_c, sigma_f=gp.sigma_f
    )

    mu = mu.reshape(n_plot, n_plot)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))

    contour = ax.contourf(X1, X2, mu, levels=20)
    plt.colorbar(contour, ax=ax, label="GP mean")

    # Observations
    ax.scatter(gp.Xs[:, 0], gp.Xs[:, 1], c="k", s=30, label="Observations")

    # Next point
    ax.scatter(
        acq_res.x_next[0],
        acq_res.x_next[1],
        c="r",
        s=80,
        marker="*",
        label="Next point"
    )

    ax.scatter(
        x_best[0],
        x_best[1],
        c="red",
        s=80,
        marker="o",
        label="Best observed"
    )
    ax.set_title(f"Iteration {it}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()

    if save_cfg.out_path:
        fig_path = os.path.join(save_cfg.out_path, "GIF")
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(os.path.join(fig_path, f"it_{it:03d}.png"), dpi=250, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
    
def plt_conv(hist: List[Dict[str, Any]], save_cfg: SaveConfig):
    
    # Extract the values fromt eh list of dictionaries
    its      = [h["it"] for h in hist]
    best_y  = [h["best_y"] for h in hist]

    # Plot 
    plt.figure(figsize=(5,3))
    plt.plot(its, best_y, "b-o")
    plt.xlabel("Calls $n$")
    plt.ylabel("min $f(x)$ after $n$ calls")
    plt.grid(True)
    if save_cfg.out_path:
        # Ensure the saving path exists
        if not os.path.exists(save_cfg.out_path):
            os.makedirs(save_cfg.out_path)
            
        figname = os.path.join(save_cfg.out_path, "conv.png")
        plt.savefig(figname, dpi=save_cfg.dpi)
    plt.show()
    
    
def plt_hist(hist: List[Dict[str, Any]], save_cfg: SaveConfig):
    
    # Extract the values fromt eh list of dictionaries
    its      = [h["it"] for h in hist]
    y_next  = [h["y_next"] for h in hist]

    # Plot 
    plt.figure(figsize=(5,3))
    plt.plot(its, y_next, "b-o")
    plt.xlabel("Calls $n$")
    plt.ylabel("$f(x)$ at each call")
    plt.grid(True)
    if save_cfg.out_path:
        # Ensure the saving path exists
        if not os.path.exists(save_cfg.out_path):
            os.makedirs(save_cfg.out_path)
            
        figname = os.path.join(save_cfg.out_path, "hist.png")
        plt.savefig(figname, dpi=save_cfg.dpi)
    plt.show()


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
    f_true: Optional[Callable[[Array], Array]] = None # in case its available
) -> BOResult:
    """
    Bayesian Optimization driver (minimization by default).
    """
    rng = _rng(bo_cfg.random_state)

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
        logger.info(f"SaveConfig: {save_cfg}")
        logger.info(f"Random seed: {bo_cfg.random_state}")
        logger.info(
            "it  |    x_next     |      y_next     |      best_y     |    l_c    |  sigma_f  |  sigma_y"
        )
        logger.info("-" * 95)
    
    # History saving list
    history: List[Dict[str, Any]] = []
    
    # Initialize the export variable, only if stated 
    h5file = None
    if save_cfg.export_enabled and save_cfg.out_path is not None:
        h5file = init_exporter(
               bo_cfg,
               gp_cfg,
               acq_cfg,
               optim_cfg,
               save_cfg,
               rng
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
    gp = build_gp_model(gp_cfg)
    
    # ---- 3) BO loop
    for it in range(bo_cfg.n_iter):

        # Fit/update GP on current data
        # NOTE: we pass it so that fit_gp can decide if hyperparams are optimized now
        gp = fit_gp(gp, X, y, gp_cfg, it)

        # Decide current best (minimization)
        best_idx = int(np.argmin(y))
        y_best = float(y[best_idx])
        x_best = X[best_idx].copy()

        # Build acquisition
        acq = make_acquisition(acq_cfg, gp, y_best)

        # Optimize acquisition -> propose next x
        acq_res = optimize_acquisition(acq, bounds, optim_cfg, rng) # output a container
        x_next = acq_res.x_next
            
        # Plot the current state if required AND that we are in 1D case
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
                f_true=f_true
            )   
            
        # Evaluate objective at proposed point
        y_next = float(np.asarray(f(x_next)).reshape(-1)[0])     
        
        # Append data
        X = np.vstack([X, x_next.reshape(1, -1)])
        y = np.concatenate([y, np.array([y_next], dtype=float)])

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
            y_best=y_best
        )
        
        # Add the current state in the save list
        states.append(state)

        # Record lightweight history
        history.append(
            dict(
                it=it,
                x_next=x_next.copy(),
                y_next=y_next,
                best_x=x_best.copy(),
                best_y=y_best,
                acq_time=acq_res.run_time,  
                n=len(y),
                l_c=gp.l_c,
                sigma_f=gp.sigma_f,
                sigma_y=gp.sigma_y,
            )
        )
        
        # Optional exporting
        if h5file is not None and save_cfg.export_every is not None:
            if (it + 1) % save_cfg.export_every == 0:
                export_state(h5file, state)
                h5file.flush()
                
        # Optinal log saving
        if logger is not None:
            log_update(logger, state)
        
    # Close the export file      
    if h5file is not None:
        h5file.close()
    
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
        states=states
    )
