# -*- coding: utf-8 -*-
"""
3D test case for BO_ML4F — minimises
  f(x) = sin(3x₀) + 0.5 cos(5x₁) + 0.2 x₂² + ε   on [-2, 2]³.

Per-iteration snapshots are saved to ./out_3d/states/state_NNN.npz.
These contain X, y, x_next, y_next and acquisition values at each step
and can be used for post-processing / convergence analysis.

@author: mendez, lecomte
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import BO_ML4F as bo


# -------------------------------------------------------------------
# Objective function
# -------------------------------------------------------------------
def func(x, noise_level=0.1):
    x = np.asarray(x)
    noise = np.random.randn() * noise_level
    return (
        np.sin(3 * x[0])
        + 0.5 * np.cos(5 * x[1])
        + 0.2 * x[2]**2
        + noise
    )


# -------------------------------------------------------------------
# BO configuration
# -------------------------------------------------------------------
LL = 2.0
bounds = [(-LL, LL), (-LL, LL), (-LL, LL)]

res = bo.bayesian_optimization(
    f         = func,
    bounds    = bounds,
    bo_cfg    = bo.BOConfig(
        n_init       = 10,
        n_iter       = 30,
        random_state = 1234,
    ),
    gp_cfg    = bo.GPConfig(),
    acq_cfg   = bo.AcqConfig(kind="EI"),
    optim_cfg = bo.OptimConfig(method="random"),
    save_cfg  = bo.SaveConfig(
        save_path     = "./out_3d",
        export_states = True,    # save .npz per iteration for post-processing
        log_enabled   = True,
    ),
)

print(f"\nDone.  Best x = {res.best_x},  best f = {res.best_y:.4f}")
print("Per-iteration snapshots saved to ./out_3d/states/")
