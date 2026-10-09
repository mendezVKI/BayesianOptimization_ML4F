"""
The three optimizers compared on the Kolmogorov control case:

    sbo                sbo, GP refitted from scratch at every iteration
    sbo (rank-one)     sbo, Cholesky factor extended by rank-one updates
    scikit-optimize    skopt (GP + EI built on scikit-learn)

All use the same GP settings (defined once below): amplitude * RBF + noise
kernel, X normalized to [0,1]^d and y standardized, EI with xi = 0.01, and the
hyperparameters re-optimized every `hpo_every` iterations, at the same
iterations, within the same bounds, starting from the previous optimum.
Per run, all start from the SAME initial design, evaluated once outside the
timed region.

The rank-one update of sbo only acts above GPConfig.rank_one_threshold points
(1000 by default) and on the iterations without hyperparameter optimization:
below the threshold, "sbo" and "sbo (rank-one)" are the same algorithm.

Remaining differences are internal to the libraries: the local optimizers of
the acquisition and of the hyperparameters differ (sbo: finite-difference
gradients; scikit-optimize / scikit-learn: analytical gradients).

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
import time
import warnings

import numpy as np
from scipy.stats.qmc import LatinHypercube
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
# skopt's own kernel classes: its acquisition optimizer needs kernel.gradient_x
from skopt.learning.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from pyRAMBO import sbo

Array = np.ndarray

# ---- shared settings, defined once -----------------------------------------
# HPO bounds in log-space of (l_c, sigma_f, sigma_y), in normalized units.
THETA_BOUNDS_LOG = [(np.log(1e-2), np.log(1e1)),     # length-scale
                    (np.log(1e-2), np.log(1e1)),     # amplitude std
                    (np.log(1e-4), np.log(1e0))]     # noise std
N_RAW_SAMPLES = 1000     # random candidates for the acquisition search
LOCAL_MAXITER = 20       # iterations of each local L-BFGS-B search (hard-coded to 20 in skopt)
HPO_EVERY = 25           # hyperparameters re-optimized every 25 iterations
RANK_ONE_THRESHOLD = sbo.GPConfig().rank_one_threshold   # sbo's default (1000 points)

gp_cfg = sbo.GPConfig(normalize_X=True, normalize_y=True,
                      optimize_hyperparams=True, hpo_every=HPO_EVERY,
                      theta_bounds_log=THETA_BOUNDS_LOG, rank_one=False)
acq_cfg = sbo.AcqConfig(kind="EI", xi=0.01)

METHODS = ("sbo", "sbo (rank-one)", "scikit-optimize")


def initial_length_scale(d: int) -> float:
    """Length-scale the hyperparameter search starts from: sqrt(d / 6), the
    typical distance between two random points of the unit cube. With a fixed
    small value (sbo's default is 0.2) and tens of unknowns, every point is far
    from every other, the kernel matrix is the identity, the marginal likelihood
    is flat and its optimization does not move: the GP learns nothing."""
    return float(np.sqrt(d / 6))


@dataclass
class RunResult:
    name: str
    seed: int
    X: Array                    # (n_total, d) all evaluated points, init design first
    y: Array                    # (n_total,)
    n_init: int
    total_time: float           # wall time of the optimization loop (excludes the init design)
    objective_time: float       # part of total_time spent inside the objective
    iter_time: Array            # (n_iter,) optimizer time of each iteration (objective excluded)
    fit_time: Array             # (n_iter,) part of iter_time spent updating the GP (the rest: acquisition)
    best_so_far: Array = field(init=False)

    def __post_init__(self):
        self.best_so_far = np.minimum.accumulate(self.y)

    @property
    def overhead_time(self) -> float:
        """Time spent by the optimizer itself (GP fit, hyperparameters, acquisition)."""
        return self.total_time - self.objective_time

    @property
    def best_y(self) -> float:
        return float(self.best_so_far[-1])

    @property
    def best_x(self) -> Array:
        return self.X[int(np.argmin(self.y))]


def _with_progress(problem, name, n_iter, every=100):
    """problem.cost, printing one line every `every` evaluations."""
    t0, best = time.perf_counter(), [np.inf]

    def f(x):
        y = problem.cost(x)
        best[0] = min(best[0], y)
        if problem.n_evals % every == 0 or problem.n_evals == n_iter:
            print(f"    {name:<15} | {problem.n_evals:>5}/{n_iter} | best new = {best[0]:.4f} | "
                  f"{time.perf_counter() - t0:7.1f} s", flush=True)
        return y
    return f


# ---- sbo ---------------------------------------------------------------------
class _NoStates(list):
    """sbo appends a copy of the GP, with its n x n Cholesky factor, to `states`
    at every iteration: ~9 GB over a 1500-point run. Not used here: dropped."""

    def append(self, state):
        pass


def run_sbo(problem, X0, y0, n_iter, bounds, n_restarts, seed, hpo_every=HPO_EVERY,
            rank_one=False, rank_one_threshold=RANK_ONE_THRESHOLD) -> RunResult:
    name = "sbo (rank-one)" if rank_one else "sbo"
    cfg = replace(gp_cfg, l_c=initial_length_scale(len(bounds)), hpo_every=hpo_every,
                  rank_one=rank_one, rank_one_threshold=rank_one_threshold)
    # "refined" = random scan + local L-BFGS-B from the n_restarts best candidates
    # (with the default "random" method n_restarts would be ignored), as skopt does
    optim_cfg = sbo.OptimConfig(method="refined", global_method="random",
                                n_raw_samples=N_RAW_SAMPLES, n_restarts=n_restarts,
                                local_maxiter=LOCAL_MAXITER)
    bo_cfg = sbo.BOConfig(n_init=len(X0), n_iter=n_iter, random_state=seed)

    problem.reset_counters()
    t0 = time.perf_counter()
    res = sbo.bayesian_optimization(
        f=_with_progress(problem, name, n_iter), bounds=bounds,
        bo_cfg=bo_cfg, gp_cfg=cfg, acq_cfg=acq_cfg, optim_cfg=optim_cfg,
        save_cfg=sbo.SaveConfig(),            # in-memory only, no plots/logs
        X_init=X0, y_init=y0, states=_NoStates(),
    )
    total = time.perf_counter() - t0
    iter_time = np.asarray(res.trace.wall_time) - np.asarray(problem.eval_times)
    fit_time = iter_time - np.array([h["acq_time"] for h in res.history])
    return RunResult(name, seed, res.X, res.y, len(X0), total, problem.eval_time, iter_time, fit_time)


# ---- scikit-optimize ---------------------------------------------------------
class _TimedGP(GaussianProcessRegressor):
    """skopt's GP, recording the duration of each fit."""
    fit_times: list = []

    def fit(self, X, y):
        t0 = time.perf_counter()
        super().fit(X, y)
        _TimedGP.fit_times.append(time.perf_counter() - t0)
        return self


def _skopt_gp(theta, optimize, seed):
    """skopt GP with hyperparameters theta = (l_c, sigma_f, sigma_y), optimized
    within THETA_BOUNDS_LOG from these values if `optimize`, kept fixed otherwise.

    Same kernel as sbo. sbo uses (length, amplitude std, noise std);
    scikit-learn uses (length, amplitude VARIANCE, noise VARIANCE): hence the squares."""
    l_c, sigma_f, sigma_y = theta
    lc_b, sf_b, sy_b = np.exp(THETA_BOUNDS_LOG)
    bnd = (lambda b: tuple(b)) if optimize else (lambda b: "fixed")
    kernel = (ConstantKernel(sigma_f**2, bnd(sf_b**2)) * RBF(l_c, bnd(lc_b))
              + WhiteKernel(sigma_y**2, bnd(sy_b**2)))
    return _TimedGP(
        kernel=kernel, optimizer="fmin_l_bfgs_b" if optimize else None,
        normalize_y=gp_cfg.normalize_y,       # X is normalized to [0,1] by the Optimizer itself
        alpha=gp_cfg.jitter,
        noise="gaussian",                     # the WhiteKernel is the noise; excluded from predictions
        random_state=seed,
    )


def _fitted_theta(model):
    """(l_c, sigma_f, sigma_y) of a fitted skopt GP, clipped to the HPO bounds."""
    amp, rbf = model.kernel_.k1.k1, model.kernel_.k1.k2
    theta = np.array([rbf.length_scale, np.sqrt(amp.constant_value), np.sqrt(model.noise_)])
    lo, hi = np.exp(np.array(THETA_BOUNDS_LOG)).T
    return np.clip(theta, lo, hi)


def run_skopt(problem, X0, y0, n_iter, bounds, n_restarts, seed, hpo_every=HPO_EVERY) -> RunResult:
    name = "scikit-optimize"
    f = _with_progress(problem, name, n_iter)
    theta = (initial_length_scale(len(bounds)), gp_cfg.sigma_f, gp_cfg.sigma_y)
    # The loop of skopt.gp_minimize, written out to choose at which iterations
    # the hyperparameters are optimized (gp_minimize does it at every one).
    opt = Optimizer(
        bounds, base_estimator=_skopt_gp(theta, True, seed), n_initial_points=0,
        acq_func=acq_cfg.kind, acq_func_kwargs=dict(xi=acq_cfg.xi), acq_optimizer="lbfgs",
        acq_optimizer_kwargs=dict(n_points=N_RAW_SAMPLES, n_restarts_optimizer=n_restarts),
        model_queue_size=1,                   # keep the last GP only (default: all of them)
        random_state=seed,
    )
    X, y, iter_time = [list(x) for x in X0], list(y0), np.zeros(n_iter)

    problem.reset_counters()
    _TimedGP.fit_times.clear()
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # sklearn convergence warnings
        for it in range(n_iter):
            t_it = time.perf_counter()
            # tell = fit the GP on all the points so far + optimize the acquisition
            if it == 0:
                opt.tell(X, y)
            else:
                optimize = it % hpo_every == 0
                opt.base_estimator_ = _skopt_gp(_fitted_theta(opt.models[-1]), optimize, seed)
                opt.tell(X[-1], y[-1])
            x_next = opt.ask()
            y_next = f(x_next)
            X.append(x_next)
            y.append(y_next)
            iter_time[it] = time.perf_counter() - t_it - problem.eval_times[-1]
    total = time.perf_counter() - t0
    return RunResult(name, seed, np.asarray(X), np.asarray(y), len(X0), total,
                     problem.eval_time, iter_time, np.array(_TimedGP.fit_times))


# ---- study -------------------------------------------------------------------
def run_all(problem, bounds, n_init, n_iter, n_restarts, seeds, methods=METHODS,
            hpo_every=HPO_EVERY, rank_one_threshold=RANK_ONE_THRESHOLD,
            done=None, on_run=None) -> dict:
    """Repeat the comparison once per seed. Each run uses ONE seed for all the
    methods and for the shared initial design, so within a run the methods have
    the same configuration. `done` = {(run index, method): RunResult} holds runs
    already available (not repeated); on_run(run index, RunResult) is called
    after each new one. Returns {method: [RunResult per run]}."""
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    runners = {
        "sbo": lambda *a: run_sbo(*a, hpo_every, False, rank_one_threshold),
        "sbo (rank-one)": lambda *a: run_sbo(*a, hpo_every, True, rank_one_threshold),
        "scikit-optimize": lambda *a: run_skopt(*a, hpo_every),
    }
    done = dict(done or {})
    results = {m: [] for m in methods}

    for i, seed in enumerate(seeds):
        seed = int(seed)
        todo = [m for m in methods if (i, m) not in done]
        if todo:
            # shared initial design: Latin hypercube over the bounds, evaluated once
            X0 = lo + (hi - lo) * LatinHypercube(d=len(bounds), seed=seed).random(n_init)
            y0 = np.array([problem.cost(x) for x in X0])

        for m in methods:
            if (i, m) in done:
                r = done[(i, m)]
            else:
                r = runners[m](problem, X0, y0, n_iter, bounds, n_restarts, seed)
                if on_run is not None:
                    on_run(i, r)
            results[m].append(r)
            print(f"run {i + 1}/{len(seeds)} (seed {seed}) | {r.name:<15} | best = {r.best_y:>10.4f} | "
                  f"time = {r.total_time:7.1f} s (objective {r.objective_time:6.1f} s, "
                  f"optimizer {r.overhead_time:6.1f} s) | n = {len(r.y)}", flush=True)
    return results
