# Kolmogorov flow control: sbo, sbo with rank-one updates, scikit-optimize

Find the forcing schedule that drives a 2-D Kolmogorov flow to a target flow, with a GP that grows up to
1500 points. The flow is the Kolmogorov environment of [HydroGym](https://github.com/dynamicslab/hydrogym)
(`hydrogym.jax`: pseudo-spectral, Re = 200, 64 x 64, driven by `sin(4y)`). It is actuated by a body force
made of shear modes, `f_c = sum_k a_k sin(k y) + b_k cos(k y)`, `k = 1, 2, 3, 4`, amplitudes in `[-0.5, 0.5]`.

The control is an open-loop schedule: the episode is cut in 3 segments of 1 time unit and the 8 amplitudes
are constant over a segment, hence **24 unknowns**. The target flow is the one a reference schedule gives;
that schedule is hidden from the optimizers, which only see the cost: the squared distance between the
velocities at the 8 x 8 probes of the environment (at the end of every segment) and those of the target
flow, in % of its value without control. So the cost is 100 without control and 0 at the reference schedule.
Each evaluation is one simulation (~0.5 s). The flow is deterministic, so is the cost.

Three optimizers are compared on the same cost function, the same initial design and
**the same GP/BO settings**:

| method | GP update at each iteration |
|---|---|
| `sbo` | kernel matrix and Cholesky factor recomputed from scratch, `O(n^3)` |
| `sbo (rank-one)` | Cholesky factor extended by one row, `O(n^2)` (`GPConfig.rank_one=True`) |
| `scikit-optimize` | recomputed from scratch by scikit-learn, `O(n^3)` |

## Install

```bash
python -m pip install -U jax
python -m pip install "hydrogym[jax]"
```

(`scikit-optimize` is needed too, as for the Burgers case.)

## Run

```bash
python run_comparison.py                        # 1 run: 50 shared initial points + 1450 evaluations per method
python run_comparison.py --n-runs 3 --master-seed 0    # more runs; master seed makes the drawn seeds reproducible
python run_comparison.py --methods sbo          # one method only; the others can be added later
python run_comparison.py --n-init 20 --n-iter 60 --rank-one-threshold 40 --hpo-every 10   # quick test (~4 min)
python run_comparison.py --plot-only            # re-plot a saved case (same options as the run)
python presentation.py                          # vorticity fields with the best schedule found
```

The default study takes about 3 hours. Every finished optimization is saved at once in `runs/`, and a
re-launch only computes what is missing: a study can be stopped, resumed, or run one method at a time.
Timings are only comparable if nothing else loads the machine.

Outputs, in `results/<objective>_d<..>_n_init<..>_n_iter<..>_n_runs<..>/` (one folder per case):
`comparison.png` (mean ± std), `overhead.png` (optimizer time per iteration against the number of points in
the GP), `comparison_per_run.png`, `summary.csv`, `study.json` (settings, seeds), `runs/` (every evaluation
and timing of every run); `assets/` for the presentation figure.

## Files

| file | role |
|---|---|
| `kolmogorov_case.py` | problem definition: HydroGym environment, bounds, timed `KolmogorovControl.cost(x)` |
| `solvers.py` | the shared configuration, `run_sbo`, `run_skopt`, `run_all` |
| `plot_comparison.py` | convergence, final best, time and per-iteration overhead figures |
| `run_comparison.py` | command-line driver, saves results |
| `presentation.py` | vorticity fields: target flow against the flow with the best schedule |

## Number of points, hyperparameter optimization rate and rank-one update

In `sbo`, the rank-one update replaces the full refit only when **both** hold (`core.fit_gp`):

- the GP holds more than `GPConfig.rank_one_threshold` points (default 1000);
- the hyperparameters are not re-optimized at that iteration (new hyperparameters change the whole kernel
  matrix, so the factor has to be recomputed anyway).

Hence the two choices of this case:

- **1500 points** (50 + 1450): `sbo` and `sbo (rank-one)` are the same algorithm for the first 1000 points and
  differ over the last 500.
- **Hyperparameters re-optimized every 25 iterations** (`--hpo-every`), not at every iteration: with
  `hpo_every=1` the rank-one update would never be used. One more point out of several hundred barely moves
  the optimum of the marginal likelihood, and one optimization costs 10 to 100 times a refit.

All three methods re-optimize at the same iterations (0, 25, 50, ...), from the previous optimum.
`skopt.gp_minimize` cannot do that (it re-optimizes at every iteration), so `run_skopt` writes its loop out
with `skopt.Optimizer`, giving it a GP with free or fixed hyperparameters depending on the iteration.

## What is identical in the three methods

Everything is derived from one `GPConfig` / `AcqConfig` / `OptimConfig` in `solvers.py`:

| | sbo, sbo (rank-one) | scikit-optimize |
|---|---|---|
| kernel | amplitude · RBF + noise | `ConstantKernel * RBF + WhiteKernel`, same initial values |
| initial length-scale | `sqrt(d/6)` (see below) | same |
| hyperparameter optimization | marginal likelihood, L-BFGS-B, **every 25 iterations**, warm-started | same iterations, warm-started |
| HPO bounds (log-space) | `THETA_BOUNDS_LOG` (top of `solvers.py`) | same bounds, converted to scikit-learn's variance parametrization |
| X normalization | `GPConfig.normalize_X=True` ([0,1]^d) | done by `skopt.Optimizer` |
| y normalization | `GPConfig.normalize_y=True` | `GaussianProcessRegressor(normalize_y=True)` |
| acquisition | EI, `xi = 0.01` | same |
| acquisition search | `method="refined"`: 1000 random candidates + L-BFGS-B (20 iterations) from the best `n_restarts` (3) | `n_points=1000`, `n_restarts_optimizer=n_restarts`, 20 iterations (same scheme) |
| jitter | `1e-10` | `alpha=1e-10` |
| start | one Latin-hypercube design per seed, evaluated once outside the timed region, shared | same |

Remaining differences are internal to the libraries: sbo differentiates the acquisition and the marginal
likelihood by finite differences, scikit-optimize / scikit-learn use analytical gradients.

**Initial length-scale.** The hyperparameter search starts from `sqrt(d/6)`, the typical distance between two
random points of the unit cube. With a fixed small value (sbo's default is 0.2) and tens of unknowns, every
point is far from every other, the kernel matrix is the identity, the marginal likelihood is flat and its
optimization does not move: with 48 unknowns both libraries returned the initial hyperparameters unchanged
and the optimization was a random search.

## Timing

Wall time of the optimization loop, split into:

- **objective**: time inside the simulations (the same for all methods);
- **GP update**: fit of the GP on the points so far, including the hyperparameter optimization when there
  is one (for sbo: iteration time minus acquisition time, so it also holds its bookkeeping);
- **acquisition search**: the rest (candidates scan + local searches).

The rank-one update only changes the GP update.

## Settings of the HydroGym environment

The flow configuration is HydroGym's default. What this case sets (`kolmogorov_case.py`):

| | HydroGym default | here | why |
|---|---|---|---|
| time step `dt` | `1e-3` | `5e-3` | 5x cheaper; same energies as `2.5e-3`. `1e-2` blows up for some actions |
| duration of an environment step (`action_time`) | 10 | 1 | affordable evaluations |
| forcing modes (`_control_field`) | `sin(k y)`, k = 4, 5, 6, 7 | `sin(k y)` and `cos(k y)`, k = 1, 2, 3, 4 (`--wavenumbers`, `--both-phases`) | more unknowns at no cost |
| integration (`_rollout`) | `RungeKuttaCrankNicolson.solve` | same scheme, same result | `solve` advances every block of steps twice when a step lasts more than 1 time unit |
| episode start | reset: 10 time units without control | same | computed once and reused by every evaluation |

With HydroGym's defaults one evaluation takes ~30 s on a laptop CPU (10 s for the reset, 20 s per
10-time-unit step): 1000 points would take 8 hours per method. With the settings above it takes ~0.5 s:
1000 points in ~10 minutes.

## Choice of the objective

`KolmogorovControl` has two objectives (`--objective`):

- **`tracking`** (default), described above. The probe velocities respond almost linearly to the forcing, so
  the cost is close to a quadratic bowl in 24 dimensions, with its minimum (0) inside the box. A GP needs many
  points to locate it: the best cost is still decreasing after hundreds of evaluations.
- **`energy`**: minus the return of the HydroGym environment, i.e. summed over the segments,
  `alpha * (mean kinetic energy) + sum |a|` (HydroGym's reward with `reward_alpha = alpha`; its default is 0).
  With the action box of the environment, the forcing changes the kinetic energy by a few tenths of a
  percent, almost linearly: the best schedules sit at corners of the box and a GP finds one in a few tens of
  evaluations, with 12 unknowns (HydroGym's 4 modes) as with 48. Raising the number of unknowns does not make
  this objective harder, so it does not justify 1000 points; it is kept for reference
  (`--objective energy --wavenumbers 4 5 6 7 --no-both-phases` is HydroGym's own actuation).

Over more than ~15 time units the flow is chaotic and both objectives turn into noise, at 2 s or more per
evaluation: the case stays in the smooth regime.
