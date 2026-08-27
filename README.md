# BayesianOptimization_ML4F

In-house Bayesian Optimisation library developed at VKI for the RSPA paper
on Reinforcement Twinning applied to the Burger equation control case.

---

## Repository layout

```
BO_ML4F.py        main library      <- import this
BO_func_YL.py     low-level helpers (kernels, GP fit/predict, EI)
conftest.py       puts the repo root on sys.path for pytest
docs/             LaTeX documentation + the ARD implementation note
examples/         runnable test cases and the animation post-processor
tests/            pytest suite
legacy/           earlier prototypes, kept for reference
_generated/       BO outputs from earlier runs (untracked)
literature/       reference papers
```

The two library files stay at the repository root on purpose: downstream
projects (e.g. `RSPA_Paper/Burger_CASE/RT_ML4F`) put this directory on
`sys.path` and `import BO_ML4F` by name.

| File | Description |
|------|-------------|
| `BO_ML4F.py` | Main library. Dataclasses, GP routines, ARD, acquisition functions, and the `bayesian_optimization` driver. |
| `BO_func_YL.py` | Lower-level helper module (kernels, GP fit/predict, EI, plotting utilities). Required by `BO_ML4F.py`. |
| `docs/BO_ML4F_documentation.tex` | Full implementation notes (LaTeX): kernel functions, GP regression, output normalisation, HPO, rank-1 Cholesky update, acquisition functions, and configuration reference. |
| `docs/ARD_IMPLEMENTATION_NOTE.md` | Design and usage note for Automatic Relevance Determination: what changed, the compatibility contract, and when *not* to switch it on. |
| `examples/1D_Test_CASE.py` | 1D smoke-test: minimises f(x) = sin(5x)(1 - tanh(x^2)) + eps on [-2, 2]. Exports per-iteration `.npz` snapshots to `./out/states/`. |
| `examples/3D_Test_CASE.py` | 3D test: minimises f(x) = sin(3x0) + 0.5 cos(5x1) + 0.2 x2^2 + eps on [-2, 2]^3. Exports snapshots to `./out_3d/states/`. |
| `examples/make_animation_1D.py` | Post-processing: loads `state_NNN.npz` snapshots and produces a GIF (or MP4). Run after `1D_Test_CASE.py`. |
| `tests/test_ard.py` | ARD test suite (backwards compatibility, kernel algebra, HPO, prediction, rank-1 update, driver). |
| `tests/test_batch_ei_adam.py` | Pre-existing regression tests for batch EI and optional ADAM refinement. |

Run the examples from inside `examples/`, and the tests from the repository root:

```
cd examples && python 1D_Test_CASE.py
pytest tests/ -q
```

### Key features of `BO_ML4F.py`

- **Two kernels**: squared-exponential (RBF) and Matérn ν=5/2, switchable via `GPConfig(kernel="matern52")`.
- **Automatic Relevance Determination (ARD)**: one length scale per input coordinate, opt-in via `GPConfig(ard=True)` or by passing a vector `l_c`. Off by default; the isotropic path is unchanged bit for bit. See the ARD section below and `docs/ARD_IMPLEMENTATION_NOTE.md`.
- **Reproducible HPO**: `GPConfig(hpo_random_state=...)` seeds the multi-restart hyperparameter search. Left at `None` the restarts are unseeded, as they always were.
- **Output normalisation**: centre and scale y before fitting to keep hyperparameters well-conditioned (`normalize_y=True`).
- **Rank-1 Cholesky update**: O(n²) GP extension once the dataset reaches `rank1_threshold` points, replacing O(n³) full refactorisation. The threshold is checked against the *total* dataset size (initial points + BO iterations), so providing a large initial set can activate it immediately.
- **Multi-restart HPO**: warm start + user init + random restarts in log-space via L-BFGS-B (`optimize_hyperparams=True`, `n_hpo_restarts=3`). Frequency controlled by `hpo_every`.
- **Three acquisition functions**: Expected Improvement (EI), Probability of Improvement (PI), Lower Confidence Bound (LCB).
- **Acquisition optimisation**: random scan, grid scan, or random-then-refine with local L-BFGS-B polishing.
- **Batch acquisition**: request several diverse EI/PI/UCB proposals per BO round with `OptimConfig(n_candidates=...)`.
- **Optional gradient refinement**: refine every acquisition proposal with projected ADAM by passing a gradient callback and `GradientRefinementConfig(enabled=True)`. This mode is disabled by default.
- **User-provided initial data**: pass `X_init` and `y_init` to seed the GP from existing evaluations.
- **Per-iteration state export**: set `export_states=True` in `SaveConfig` to write compressed `.npz` snapshots each iteration (GP posterior, acquisition values, proposed point). Use `make_animation_1D.py` to build a GIF from them.

### General usage example

```python
import numpy as np
import BO_ML4F as bo

# --- Objective function (must accept a 1-D array of shape (d,))
def func(x, noise_level=0.1):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0]**2)) + np.random.randn() * noise_level

# --- (Optional) pre-evaluated initial dataset
#     If omitted, bo_cfg.n_init random points are sampled automatically.
X_init = np.random.uniform(-2, 2, (300, 1))   # shape (n0, d)
y_init = np.array([func(x) for x in X_init])  # shape (n0,)

# --- Search space
bounds = [(-2.0, 2.0)]   # one tuple per dimension: (lower, upper)

# --- BO loop settings
bo_cfg = bo.BOConfig(
    n_init        = 10,        # initial random points (ignored when X_init is given)
    n_iter        = 50,        # number of BO iterations
    random_state  = 42,
    init_sampling = "lhs",     # "uniform" or "lhs"
)

# --- GP settings
gp_cfg = bo.GPConfig(
    kernel               = "matern52",  # "rbf" or "matern52"
    normalize_y          = True,        # centre/scale y before fitting
    optimize_hyperparams = True,        # tune ℓ, σ_f, σ_y by marginal likelihood
    hpo_every            = 5,           # run HPO every 5 iterations
    n_hpo_restarts       = 3,           # warm start + default + 3 random restarts
    rank1_threshold      = 500,         # use O(n²) update once dataset ≥ 500 pts
                                        # (with 300 initial pts: kicks in after iter 200;
                                        #  with 500 initial pts: kicks in from iter 1)
)

# --- Acquisition function
acq_cfg = bo.AcqConfig(
    kind     = "EI",   # "EI", "PI", or "UCB"
    xi       = 0.01,   # exploration jitter (EI/PI)
    maximize = False,  # set True for maximisation problems
)

# --- Acquisition optimiser
optim_cfg = bo.OptimConfig(
    method        = "refined",  # "random", "grid", or "refined"
    n_raw_samples = 2000,       # global random candidates
    n_restarts    = 10,         # L-BFGS-B starts from top candidates
    n_candidates  = 1,          # proposals per BO round; 1 preserves legacy behavior
)

# --- Output / logging
save_cfg = bo.SaveConfig(
    save_path     = "./out",   # directory for all outputs
    log_enabled   = True,
    export_states = True,      # save .npz snapshot each iteration
    n_plot        = 400,       # grid resolution for 1D GP export (ignored in nD)
    # plt_state_enabled = True  # legacy inline plots (1D interactive sessions only)
)

# --- Run
res = bo.bayesian_optimization(
    f                = func,
    bounds           = bounds,
    bo_cfg           = bo_cfg,
    gp_cfg           = gp_cfg,
    acq_cfg          = acq_cfg,
    optim_cfg        = optim_cfg,
    save_cfg         = save_cfg,
    X_init           = X_init,    # omit to let the library sample n_init points
    y_init           = y_init,    # omit to let the library evaluate them
    top_up_to_n_init = False,     # False: use X_init as-is; True: pad to n_init
)

print(f"Best x : {res.best_x}")
print(f"Best y : {res.best_y:.6f}")

# For 1D problems: build a GIF animation of the BO trajectory
#   python make_animation_1D.py ./out/states ./out/animation.gif 3
```

### Optional batch EI with ADAM refinement

The objective remains the source of all archived values. The gradient callback
returns the gradient of that same objective in the original bounded
coordinates:

```python
def objective_gradient(x):
    return np.array([2.0 * (x[0] - 0.3), 4.0 * (x[1] + 0.4)])

res = bo.bayesian_optimization(
    f=func,
    gradient=objective_gradient,
    bounds=bounds,
    bo_cfg=bo_cfg,
    gp_cfg=gp_cfg,
    acq_cfg=acq_cfg,
    optim_cfg=bo.OptimConfig(
        method="refined",
        n_candidates=5,
        batch_distance_scale=0.08,
    ),
    refinement_cfg=bo.GradientRefinementConfig(
        enabled=True,
        n_steps=30,
        learning_rate=0.02,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        distance_threshold=1e-3,
        close_pair_policy="final",  # alternatively "midpoint"
    ),
    save_cfg=save_cfg,
)
```

Each BO round first selects a diverse acquisition batch. Every proposal is
evaluated and then refined by projected ADAM in normalized parameter space.
When the normalized displacement exceeds `distance_threshold`, both the
proposal and refined endpoint enter the GP dataset. Otherwise only the refined
point is retained by default. With `close_pair_policy="midpoint"`, the midpoint
is explicitly evaluated and retained instead; endpoint objective values are
never averaged.

`BOState` and each history entry expose `x_proposed`, `y_proposed`,
`x_refined`, `y_refined`, `refinement_displacement`, and `n_added`. The legacy
`x_next` and `y_next` fields remain the first acquisition proposal and its
objective value.

---

## Automatic Relevance Determination (ARD)

An isotropic kernel assumes the objective decorrelates at the same rate along every coordinate.
Mapping the inputs to a unit box makes coordinates commensurable but not equally influential, and
when they are not, the isotropic kernel books the difference as observation noise: the posterior
mean flattens and the acquisition stops ranking candidates. ARD replaces the scalar length scale by
one per coordinate, which is exactly an isotropic kernel on rescaled inputs:

```
k_ARD(x, x'; l) == k_iso(x/l, x'/l; 1)
```

**ARD is off by default and the isotropic path is untouched.** Existing scripts need no changes.

### Three ways to use it

```python
import numpy as np
import BO_ML4F as bo

# (1) OFFLINE SCREENING -- measure the anisotropy of an objective from an archive.
#     This is what ARD is unambiguously good for.
result = bo.estimate_ard_length_scales(
    X_unit, y,
    kernel="matern52",
    theta_bounds_log=[(np.log(0.05), np.log(20.0)),   # length scales (broadcast to all d)
                      (np.log(0.20), np.log(5.00)),   # sigma_f
                      (np.log(0.02), np.log(1.50))],  # sigma_y
    n_restarts=8,
    random_state=0,
)
result["length_scales"]      # (d,) fitted scales
result["spread"]             # max(l)/min(l): the headline anisotropy number
result["relevance_order"]    # coordinate indices, most relevant first
result["whitening_weights"]  # 1/l, normalised to unit geometric mean

# (2) FROZEN ARD -- declare the measured scales before a campaign, never re-fit them.
gp_cfg = bo.GPConfig(l_c=result["length_scales"], kernel="matern52",
                     normalize_y=True, optimize_hyperparams=False)

# (3) FITTED ARD -- optimise d+2 hyperparameters inside the campaign.
gp_cfg = bo.GPConfig(ard=True, kernel="matern52", normalize_y=True,
                     optimize_hyperparams=True, hpo_random_state=0)
```

A 3-entry `theta_bounds_log` / `theta0_log` is broadcast across the coordinates in ARD mode, so an
existing isotropic HPO box can be reused unchanged.

### When *not* to switch it on

ARD fits `d - 1` extra hyperparameters from the same observations you are searching with. On the
RSPA Burgers policy objective at 55 observations in 13 dimensions, fitted ARD lifts held-out
Spearman from 0.573 to 0.703 but with three times the between-campaign spread (±0.11 against
±0.06); at 30 observations it is *worse* than isotropic on RMSE. A dimensionality reduction derived
beforehand beats both at every budget. If a study's claim is about reproducibility, prefer
option (2) over option (3).

Enabling ARD downstream also couples to things that were derived under isotropy — HPO length-scale
floors, trust-region radii, anything that formats `l_c` with `:.3e`. `docs/ARD_IMPLEMENTATION_NOTE.md`
section 6.4 is the checklist.

---

## Subdirectories

### `legacy/BO_HOML26/`
Earlier prototype implementation used in the Hands-On Machine Learning for
Fluid dynamics (HOML26) course. Contains standalone 1D and 3D BO examples
and a plotting utility for the EI acquisition function. Kept for reference.

### `legacy/Single_Fid_BO_last_version/`
Development branch with versioned snapshots (`BO_ML4F_v0_1.py` through
`v0_3.py`), benchmarks, rank-1 update debugging scripts, and 1D/2D test
cases (Branin, Rosenbrock). Kept as development archive.

---

## Dependencies

```
numpy
scipy
scikit-learn   (rbf_kernel used internally)
matplotlib
pillow         (for GIF export via examples/make_animation_1D.py)
pytest         (to run the test suite)
```

Install with:
```
pip install numpy scipy scikit-learn matplotlib pillow pytest
```

---

## Reference

This library is described in:

> Lecomte, Y. & Mendez, M.A. — *Reinforcement Twinning for the Burger Equation*,
> Proceedings of the Royal Society A, 2026 (in preparation).
