# BayesianOptimization_ML4F

In-house Bayesian Optimisation library developed at VKI for the RSPA paper
on Reinforcement Twinning applied to the Burger equation control case.

---

## Main library (root folder)

| File | Description |
|------|-------------|
| `BO_ML4F.py` | Main library. Contains all dataclasses, GP routines, acquisition functions, and the `bayesian_optimization` driver. |
| `BO_func_YL.py` | Lower-level helper module (kernels, GP fit/predict, EI, plotting utilities). Required by `BO_ML4F.py`. |
| `BO_ML4F_documentation.tex` | Full implementation notes (LaTeX): kernel functions, GP regression, output normalisation, HPO, rank-1 Cholesky update, acquisition functions, and configuration reference. |
| `1D_Test_CASE.py` | 1D smoke-test for `BO_ML4F.py`. Minimises f(x) = sin(5x)(1 − tanh(x²)) + ε on [−2, 2]. Enables per-iteration GP + acquisition plots. |
| `3D_Test_CASE.py` | 3D test for `BO_ML4F.py`. Minimises f(x) = sin(3x₀) + 0.5 cos(5x₁) + 0.2 x₂² + ε on [−2, 2]³. |

### Key features of `BO_ML4F.py`

- **Two kernels**: squared-exponential (RBF) and Matérn ν=5/2, switchable via `GPConfig(kernel="matern52")`.
- **Output normalisation**: centre and scale y before fitting to keep hyperparameters well-conditioned (`normalize_y=True`).
- **Rank-1 Cholesky update**: O(n²) GP extension once the dataset reaches `rank1_threshold` points, replacing O(n³) full refactorisation. The threshold is checked against the *total* dataset size (initial points + BO iterations), so providing a large initial set can activate it immediately.
- **Multi-restart HPO**: warm start + user init + random restarts in log-space via L-BFGS-B (`optimize_hyperparams=True`, `n_hpo_restarts=3`). Frequency controlled by `hpo_every`.
- **Three acquisition functions**: Expected Improvement (EI), Probability of Improvement (PI), Lower Confidence Bound (LCB).
- **Acquisition optimisation**: random scan, grid scan, or random-then-refine with local L-BFGS-B polishing.
- **User-provided initial data**: pass `X_init` and `y_init` to seed the GP from existing evaluations.

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
)

# --- Plotting / logging
save_cfg = bo.SaveConfig(
    plt_state_enabled = True,    # per-iteration plots (1D problems only)
    save_path         = "./out", # directory for plots and log file
    log_enabled       = True,
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
```

---

## Subdirectories

### `BO_HOML26/`
Earlier prototype implementation used in the Hands-On Machine Learning for
Fluid dynamics (HOML26) course. Contains standalone 1D and 3D BO examples
and a plotting utility for the EI acquisition function. Kept for reference.

### `Single_Fid_BO_last_version/`
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
```

Install with:
```
pip install numpy scipy scikit-learn matplotlib
```

---

## Reference

This library is described in:

> Lecomte, Y. & Mendez, M.A. — *Reinforcement Twinning for the Burger Equation*,
> Proceedings of the Royal Society A, 2026 (in preparation).
