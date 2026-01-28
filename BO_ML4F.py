# -*- coding: utf-8 -*-
"""
Updated on Wed Jan 28 10:43:30 2026

@author: mendez, lecomte
"""

from __future__ import annotations  # this is to annotate the functions (-->)
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Tuple, Union

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
# Import the functions of the home-made BO
from BO_func_YL import rbf_kernel_

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
    l_c: float = 0.3          # length-scale ell
    sigma_f: float = 1.0      # kernel amplitude std (so variance is sigma_f^2)
    sigma_y: float = 0.1      # observation noise std
    jitter: float = 1e-10     # numerical stabilizer on diagonal

    # hyperparameter optimization (HPO) options
    optimize_hyperparams: bool = False
    hpo_every: int = 1        # optimize every k BO iterations (1 = every iteration)

    # optional initial guess and bounds in log-space
    # theta_log = log([l_c, sigma_f, sigma_y])
    theta0_log: Optional[Array] = None
    theta_bounds_log: Optional[List[Tuple[float, float]]] = None


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
# collect the furrent state of the BO
@dataclass
class BOState:
    it: int
    X: Array              # (n, d)
    y: Array              # (n,)
    gp: GPModel           # GP container (alpha, L, Xs, etc.)
    acq: Any              # acquisition callable/object (your design)
    x_next: Optional[Array] = None  # (d,)
    y_next: Optional[float] = None  # scalar


# collects all the info concerning the result of the BO
@dataclass
class BOResult:
    X: Array
    y: Array
    best_x: Array
    best_y: float
    history: List[Dict[str, Any]]
    gp: GPModel

# collects all the information concerning the evaluation of 
# the next point to sample
@dataclass
class AcqOptimizationResult:
    x_next: Array          # (d,)
    Xcand: Array           # (N, d)
    a: Array               # (N,)


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
# GP kernel + fit / prediction (Cholesky reuse)
# ----------------------------

def rbf_kernel_amp(X1: Array, X2: Array, l_c: float, sigma_f: float) -> Array:
    """
    RBF kernel with amplitude:
      k(x,x') = sigma_f^2 * exp(-||x-x'||^2 / (2 l_c^2))
    """
    gamma = 0.5 / (l_c**2)
    # return (sigma_f**2) * rbf_kernel(X1, X2, gamma=gamma)
    return (sigma_f**2) * rbf_kernel_(X1, X2, gamma=gamma)


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


def fit_gp(gp: GPModel, X: Array, y: Array, gp_cfg: GPConfig, it: int) -> GPModel:
    """
    Fit GP and store Cholesky + alpha in the container.

    If gp_cfg.optimize_hyperparams=True, we run HPO (log-marginal likelihood)
    every gp_cfg.hpo_every iterations.
    """
    gp.Xs = np.asarray(X, dtype=float)
    gp.ys = np.asarray(y, dtype=float).reshape(-1)

    # decide whether to do HPO at this iteration
    do_hpo = gp_cfg.optimize_hyperparams and (gp_cfg.hpo_every > 0) and (it % gp_cfg.hpo_every == 0)
    if do_hpo:
        gp = optimize_gp_hyperparams(gp.Xs, gp.ys, gp, gp_cfg)

    # fit with current hyperparameters (fixed or optimized)
    gp.alpha, gp.L = gp_fit(gp.Xs, gp.ys, gp.l_c, gp.sigma_f, gp.sigma_y, gp.jitter)
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
        Xcand=Xcand,
        a=a,
    )

def export_state(state: BOState, path_or_handler: Any) -> None:
    """Optional exporting."""
    return


def plt_current_state(gp: GPModel, ):
    
    
    
    
    Xs = gp.Xs
    ys = gp.ys
    
    
    return


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
    callbacks: Optional[List[Callable[[BOState], None]]] = None,
    exporter: Optional[Any] = None,
    X_init: Optional[Array] = None,  # initial points (optional)
    y_init: Optional[Array] = None,  # initial values (optional)
    top_up_to_n_init: bool = True,   # if X_init has too few points, add more random ones
) -> BOResult:
    """
    Bayesian Optimization driver (minimization by default).
    """
    rng = _rng(bo_cfg.random_state)
    callbacks = callbacks or []

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


        # Evaluate objective at proposed point
        y_next = float(np.asarray(f(x_next)).reshape(-1)[0])
        
        # Plot the current state
        plot_current_state(
            gp=gp,
            acq_res=acq_res,
            f_true=f,
            bounds=bounds,
            it=it,
        )        
        
        # Append data
        X = np.vstack([X, x_next.reshape(1, -1)])
        y = np.concatenate([y, np.array([y_next], dtype=float)])

        # Build state for callbacks/export/logging
        state = BOState(
            it=it,
            X=X,
            y=y,
            gp=gp,
            acq=acq,
            x_next=x_next,
            y_next=y_next,
        )

        # Record lightweight history
        history.append(
            dict(
                it=it,
                x_next=x_next.copy(),
                y_next=y_next,
                best_x=x_best.copy(),
                best_y=y_best,
                n=len(y),
                l_c=gp.l_c,
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
    )
