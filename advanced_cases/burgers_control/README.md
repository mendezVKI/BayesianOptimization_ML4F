# Burgers control: sbo vs scikit-optimize

Find the 3 weights `w` of a linear feedback law `a(t) = w · [u(x1,t), u(x2,t), u(x3,t)]`
that damp a perturbed Burgers wave (see `Burgers/README.md` for the physics).
The cost is the cumulative episode cost (perturbation damping + actuation penalty, `GAMMA = 10`);
each evaluation is one PDE simulation (~0.2 s). The search box is `w_i ∈ [-0.1, 0.1]`.

`sbo` (this repo) and `scikit-optimize` (`skopt.gp_minimize`, scikit-learn based) are compared on the
same cost function, the same initial design and **the same GP/BO settings**.

## Run

```bash
python run_comparison.py                        # 5 runs; 10 shared initial points + 150 evaluations per run
python run_comparison.py --n-runs 10 --master-seed 0   # more runs; master seed makes the drawn seeds reproducible
python run_comparison.py --n-restarts 5         # restarts of the acquisition optimization (default 10); the search-space bounds are set at the top of run_comparison.py
python run_comparison.py --n-iter 50 --n-runs 2   # quick test
python run_comparison.py --plot-only --n-init 10 --n-iter 50 --n-runs 10   # re-plot a saved case (same n_init/n_iter/n_runs as the run)
python presentation.py --gifs                   # space-time maps + GIFs with the best weights found
```

Each run draws one random seed, used by BOTH methods (and for the shared initial design), so within a run they have the same configuration; the study repeats this `--n-runs` times.

Outputs, in `results/n_init<..>_n_iter<..>_n_runs<..>/` (one folder per case): `comparison.png` (mean ± std), `comparison_per_run.png` (final best and time of every run), `summary.csv`, `comparison.npz`; `assets/` for the presentation figures. `presentation.py` reads the most recent case folder (or `--results-dir`).

## Files

| file | role |
|---|---|
| `burgers_case.py` | problem definition: environment, bounds, timed `BurgersControl.cost(w)` |
| `solvers.py` | the shared configuration, `run_sbo`, `run_skopt`, `run_all` |
| `plot_comparison.py` | convergence, final best, and time figure |
| `run_comparison.py` | command-line driver, saves results |
| `presentation.py` | clean rewrite of the old presentation script (maps, GIFs) |
| `Burgers/` | the environment (Fabio and Lorenzo); only imports (gymnasium), paths and printing touched |
| `legacy/` | the original presentation script, unchanged |
| `assets/` | figures and GIFs produced before the refactor |

## What is identical in both methods

Everything is derived from one `GPConfig` / `AcqConfig` / `OptimConfig` in `solvers.py`:

| | sbo | scikit-optimize |
|---|---|---|
| kernel | amplitude · RBF + noise | `ConstantKernel * RBF + WhiteKernel`, same initial values |
| hyperparameter optimization | marginal likelihood, L-BFGS-B, **every iteration** | same (done by scikit-learn's GPR at every fit) |
| HPO bounds (log-space) | `THETA_BOUNDS_LOG` (top of `solvers.py`) | same bounds, converted to scikit-learn's variance parametrization |
| X normalization | `GPConfig.normalize_X=True` ([0,1]^d) | done by `gp_minimize` |
| y normalization | `GPConfig.normalize_y=True` | `GaussianProcessRegressor(normalize_y=True)` |
| acquisition | EI, `xi = 0.01` | same |
| acquisition search | `method="refined"`: 2000 random candidates + L-BFGS-B from the best `n_restarts` | `n_points=2000`, `n_restarts_optimizer=n_restarts` (same scheme) |
| jitter | `1e-10` | `alpha=1e-10` |
| start | one Latin-hypercube design per seed, evaluated once outside the timed region, shared | same |

Remaining differences are internal to the libraries: scikit-learn restarts the hyperparameter search from the
initial values at every fit, sbo warm-starts from the previous optimum; the local acquisition optimizers differ.

## Timing

Wall time of the optimization loop, split into the time spent inside the objective (the same PDE solves
for both methods) and the optimizer overhead (GP fit, hyperparameters, acquisition). Only the overhead
differs between methods.

## Changes made to `Burgers/`

- `Burgers_implicit_env.py`: uses `gymnasium` (the maintained fork; no deprecation banner) and falls back to `gym` if it is not installed, end-of-episode print only with `verbose=True`, `\gamma` escapes.
- `initial_conditions.py`: data path relative to the file (it was the hard-coded relative `Burgers/`, which only worked from the parent folder).
