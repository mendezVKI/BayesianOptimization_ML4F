"""
The two optimizers compared on the Burgers control case: sbo and
scikit-optimize (skopt.gp_minimize, GP + EI built on scikit-learn).

Both use the same GP settings (defined once below): amplitude * RBF + noise
kernel, hyperparameters re-optimized at every iteration within the same
bounds, X normalized to [0,1]^d and y standardized, EI with xi = 0.01.
Per run, the inputs are the search-space bounds and the number of restarts of
the acquisition optimization. Both start from the SAME initial design,
evaluated once outside the timed region.

Remaining differences are internal to the libraries: scikit-learn restarts the
hyperparameter search from the initial values at each fit, sbo warm-starts from
the previous optimum; the local acquisition optimizers differ.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
import warnings

import numpy as np
from scipy.stats.qmc import LatinHypercube

from pyRAMBO import sbo

Array = np.ndarray

# ---- shared settings, defined once -----------------------------------------
# HPO bounds in log-space of (l_c, sigma_f, sigma_y), in normalized units.
THETA_BOUNDS_LOG = [(np.log(1e-2), np.log(1e1)),     # length-scale
                    (np.log(1e-2), np.log(1e1)),     # amplitude std
                    (np.log(1e-4), np.log(1e0))]     # noise std
N_RAW_SAMPLES = 2000     # random candidates for the acquisition search

gp_cfg = sbo.GPConfig(normalize_X=True, normalize_y=True,
                      optimize_hyperparams=True, hpo_every=1,
                      theta_bounds_log=THETA_BOUNDS_LOG)
acq_cfg = sbo.AcqConfig(kind="EI", xi=0.01)

METHODS = ("sbo", "scikit-optimize")


@dataclass
class RunResult:
    name: str
    seed: int
    X: Array                    # (n_total, d) all evaluated points, init design first
    y: Array                    # (n_total,)
    n_init: int
    total_time: float           # wall time of the optimization loop (excludes the init design)
    objective_time: float       # part of total_time spent inside the objective
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


def run_sbo(problem, X0, y0, n_iter, bounds, n_restarts, seed) -> RunResult:
    # "refined" = random scan + local L-BFGS-B from the n_restarts best candidates
    # (with the default "random" method n_restarts would be ignored), as skopt does
    optim_cfg = sbo.OptimConfig(method="refined", global_method="random",
                                n_raw_samples=N_RAW_SAMPLES, n_restarts=n_restarts)
    bo_cfg = sbo.BOConfig(n_init=len(X0), n_iter=n_iter, random_state=seed)

    problem.reset_counters()
    t0 = time.perf_counter()
    res = sbo.bayesian_optimization(
        f=problem.cost, bounds=bounds,
        bo_cfg=bo_cfg, gp_cfg=gp_cfg, acq_cfg=acq_cfg, optim_cfg=optim_cfg,
        save_cfg=sbo.SaveConfig(),            # in-memory only, no plots/logs
        X_init=X0, y_init=y0,
    )
    total = time.perf_counter() - t0
    return RunResult("sbo", seed, res.X, res.y, len(X0), total, problem.eval_time)


def run_skopt(problem, X0, y0, n_iter, bounds, n_restarts, seed) -> RunResult:
    from skopt import gp_minimize
    from skopt.learning import GaussianProcessRegressor
    # skopt's own kernel classes: its acquisition optimizer needs kernel.gradient_x
    from skopt.learning.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    # Same kernel and bounds as sbo. sbo uses (length, amplitude std, noise std);
    # scikit-learn uses (length, amplitude VARIANCE, noise VARIANCE): hence the squares.
    (lc_lo, lc_hi), (sf_lo, sf_hi), (sy_lo, sy_hi) = np.exp(THETA_BOUNDS_LOG)
    kernel = (ConstantKernel(gp_cfg.sigma_f**2, (sf_lo**2, sf_hi**2))
              * RBF(gp_cfg.l_c, (lc_lo, lc_hi))
              + WhiteKernel(gp_cfg.sigma_y**2, (sy_lo**2, sy_hi**2)))
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=gp_cfg.normalize_y,       # X is normalized to [0,1] by gp_minimize itself
        alpha=gp_cfg.jitter,
        noise="gaussian",                     # the WhiteKernel is the noise; excluded from predictions
        random_state=seed,
    )

    problem.reset_counters()
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # sklearn convergence warnings
        res = gp_minimize(
            problem.cost, bounds, base_estimator=gpr,
            acq_func=acq_cfg.kind, xi=acq_cfg.xi,
            n_points=N_RAW_SAMPLES, n_restarts_optimizer=n_restarts,
            x0=X0.tolist(), y0=y0.tolist(),
            n_calls=n_iter,                   # with y0 given, n_calls counts only NEW calls
            n_initial_points=0, random_state=seed,
        )
    total = time.perf_counter() - t0
    return RunResult("scikit-optimize", seed, np.asarray(res.x_iters), np.asarray(res.func_vals),
                     len(X0), total, problem.eval_time)


def run_all(problem, bounds, n_init, n_iter, n_restarts, n_runs, master_seed=None) -> dict:
    """Repeat the comparison n_runs times. Each run draws ONE random seed, used by
    both methods and for the shared initial design, so within a run sbo and
    scikit-optimize have the same configuration. master_seed (optional) makes the
    list of drawn seeds reproducible. Returns {method: [RunResult per run]}."""
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    seeds = np.random.default_rng(master_seed).integers(0, 2**31 - 1, size=n_runs)
    results = {m: [] for m in METHODS}

    for i, seed in enumerate(seeds):
        seed = int(seed)
        # shared initial design: Latin hypercube over the bounds, evaluated once
        X0 = lo + (hi - lo) * LatinHypercube(d=len(bounds), seed=seed).random(n_init)
        y0 = np.array([problem.cost(x) for x in X0])

        for run in (run_sbo, run_skopt):
            r = run(problem, X0, y0, n_iter, bounds, n_restarts, seed)
            results[r.name].append(r)
            print(f"run {i + 1}/{n_runs} (seed {seed}) | {r.name:<15} | best = {r.best_y:>10.2f} | "
                  f"time = {r.total_time:6.1f} s (objective {r.objective_time:5.1f} s, "
                  f"optimizer {r.overhead_time:5.1f} s) | n = {len(r.y)}")
    return results
