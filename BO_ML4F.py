# -*- coding: utf-8 -*-
"""
Updated on Wed Jan 28 10:43:30 2026

@author: mendez, lecompte
"""

from __future__ import annotations # this is to annotate the functions (-->)
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Tuple, Union


import numpy as np
from scipy.linalg import cholesky, cho_solve
from sklearn.metrics.pairwise import rbf_kernel


Array = np.ndarray # short cut for numpy arrays
# short cut for domain bounds as a list
Bounds = List[Tuple[float, float]]  # [(l1,u1), ..., (ld,ud)] 


# ----------------------------
# Settings / Config containers
# ----------------------------
# these are dataclasses that makes it simple to access all the settings.

# general settigns of the optimizer
@dataclass
class GPConfig:
    kernel: str = "RBF"
    noise: float = 1e-6
    normalize_y: bool = True
    optimize_hyperparams: bool = True
    # add: priors, ARD, etc.

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
    method: str = "multistart"   # "multistart", "random", "grid"
    optimizer: str = "L-BFGS-B"  # for local optimization
    n_restarts: int = 20
    n_raw_samples: int = 2000
    # add: constraints, batch, etc.

# general settigns for the BO optimizer (optimization etc)
@dataclass
class BOConfig:
    n_init: int = 5
    n_iter: int = 20
    random_state: Optional[int] = None
    export_every: Optional[int] = None  # e.g. export state every k iterations


# ----------------------------
# State / Results containers
# ----------------------------
# collect the furrent state of the BO
@dataclass
class BOState:
    it: int
    X: Array              # (n, d)
    y: Array              # (n,)
    gp: Any               # GP model object (your class)
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
    gp: Any


# ----------------------------
# Helper utilities (skeleton)
# ----------------------------
# this is for setting the seed (if one wants reproducible results)
def _rng(random_state: Optional[int]) -> np.random.Generator:
    """Return a numpy Generator for reproducibility."""
    return np.random.default_rng(random_state)

# this is to define the random initialization
def sample_initial_design(bounds: Bounds, n_init: int, rng: np.random.Generator) -> Array:
    """
    INPUTS:
      bounds: list of (lo, hi)
      n_init: number of initial points
      rng: numpy generator
    OUTPUT:
      X0: (n_init, d) array within bounds
    """
    d = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    # TODO: replace with LHS if you want
    X0 = lo + (hi - lo) * rng.random((n_init, d))
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


# ----------------------------
# Subfunctions you will fill
# ----------------------------
def gp_fit(Xs, ys, l_c=0.3, sigma_y=0.1, jitter=1e-10):
    """
    Fit GP with RBF kernel and Gaussian noise.

    Inputs
    ------
    Xs : (n,d) training inputs
    ys : (n,) or (n,1) training targets
    l_c : length scale (ell)
    sigma_y : observation noise std
    jitter : small diagonal added for numerical stability

    Returns
    -------
    alpha : (n,)   where alpha = (Kss + sigma_y^2 I)^(-1) y
    L     : (n,n)  Cholesky factor of (Kss + sigma_y^2 I)
    """
    Xs = np.asarray(Xs, dtype=float)
    ys = np.asarray(ys, dtype=float).reshape(-1)  # force (n,)

    gamma = 0.5 / (l_c**2)
    Kss = rbf_kernel(Xs, Xs, gamma=gamma)  # (n,n)

    n = Xs.shape[0]
    Ky = Kss + (sigma_y**2 + jitter) * np.eye(n)

    L = cholesky(Ky, lower=True)                 # Ky = L L^T
    alpha = cho_solve((L, True), ys)             # solve Ky alpha = y

    return alpha, L


def gp_predict(X, Xs, alpha, L, l_c=0.3, return_cov=False):
    """
    GP prediction reusing alpha and Cholesky factor L from gp_fit.

    Inputs
    ------
    X : (m,d) test inputs
    Xs : (n,d) training inputs
    alpha : (n,) from gp_fit
    L : (n,n) from gp_fit
    l_c : length scale
    return_cov : if True return full covariance (m,m),
                 else return only diagonal variances (m,)

    Returns
    -------
    mu : (m,) predictive mean
    cov_or_var : (m,m) if return_cov else (m,) latent predictive variance
    """
    X = np.asarray(X, dtype=float)
    Xs = np.asarray(Xs, dtype=float)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)

    gamma = 0.5 / (l_c**2)

    Ks = rbf_kernel(X, Xs, gamma=gamma)          # (m,n)
    mu = Ks @ alpha                               # (m,)

    # Solve Ky V = Ks^T  -> V = Ky^{-1} Ks^T
    V = cho_solve((L, True), Ks.T)               # (n,m)

    if return_cov:
        Kxx = rbf_kernel(X, X, gamma=gamma)      # (m,m)
        cov = Kxx - Ks @ V                       # (m,m)
        return mu, cov
    else:
        # diag(Kxx - Ks Ky^{-1} Ks^T)
        # for RBF, diag(Kxx) = 1 (since k(x,x)=1) if no amplitude factor
        # More generally compute diag explicitly:
        Kxx_diag = np.ones(X.shape[0])           # for plain RBF with unit amplitude
        var = Kxx_diag - np.sum(Ks * V.T, axis=1)  # (m,)
        return mu, var


def acquisition_f(acq_cfg: AcqConfig, gp: Any, y_best: float) -> Callable[[Array], Array]:
    """
    Build acquisition function a(x).

    INPUTS:
      acq_cfg: acquisition settings
      gp: fitted GP model
      y_best: current best objective value (min or max depending on convention)
    OUTPUT:
      acq: callable mapping Xcand (m, d) -> a (m,)
    """
    # TODO: implement EI/UCB/PI (likely minimization convention)
    def acq(Xcand: Array) -> Array:
        raise NotImplementedError
    return acq


def optimize_acquisition(
    acq: Callable[[Array], Array],
    bounds: Bounds,
    optim_cfg: OptimConfig,
    rng: np.random.Generator,
) -> Array:
    """
    Optimize acquisition to propose next x.

    INPUTS:
      acq: acquisition callable, Xcand (m,d) -> a(m,)
      bounds: list of (lo, hi)
      optim_cfg: settings for multistart / raw sampling / local optimizer
      rng: numpy generator
    OUTPUT:
      x_next: (d,) proposed point
    """
    # TODO:
    #  1) draw n_raw_samples random points
    #  2) pick best few as starts
    #  3) run L-BFGS-B (or other) per start
    #  4) return best found
    raise NotImplementedError


def export_state(state: BOState, path_or_handler: Any) -> None:
    """
    Optional exporting.

    INPUTS:
      state: BOState
      path_or_handler: could be a filepath or a function handle, up to you
    OUTPUT:
      None
    """
    # TODO: implement (pickle, np.savez, json, etc.)
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
) -> BOResult:
    """
    Bayesian Optimization driver (minimization by default).

    INPUTS:
      f: objective function, x(d,) -> scalar
      bounds: [(lo, hi), ...] (d dims)
      bo_cfg: BOConfig (n_init, n_iter, random_state, export_every)
      gp_cfg: GPConfig
      acq_cfg: AcqConfig
      optim_cfg: OptimConfig
      callbacks: list of callables cb(state) executed each iter
      exporter: optional handle used by export_state

    OUTPUT:
      BOResult with data, best, history, and final GP
    """
    rng = _rng(bo_cfg.random_state)
    callbacks = callbacks or []

    # ---- 1) initial design
    X = sample_initial_design(bounds, bo_cfg.n_init, rng)  # (n_init, d)
    y = evaluate_objective(f, X)                           # (n_init,)

    # ---- 2) build GP object
    gp = build_gp_model(gp_cfg)

    history: List[Dict[str, Any]] = []

    # ---- 3) BO loop
    for it in range(bo_cfg.n_iter):
        # Fit/update GP on current data
        gp = fit_gp(gp, X, y, gp_cfg)

        # Decide current best (minimization)
        best_idx = int(np.argmin(y))
        y_best = float(y[best_idx])
        x_best = X[best_idx].copy()

        # Build acquisition
        acq = make_acquisition(acq_cfg, gp, y_best)

        # Optimize acquisition -> propose next x
        x_next = optimize_acquisition(acq, bounds, optim_cfg, rng)  # (d,)

        # Evaluate objective at proposed point
        y_next = float(np.asarray(f(x_next)).reshape(-1)[0])

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

        # Record lightweight history (don’t store gp unless you want to)
        history.append(
            dict(
                it=it,
                x_next=x_next.copy(),
                y_next=y_next,
                best_x=x_best.copy(),
                best_y=y_best,
                n=len(y),
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
