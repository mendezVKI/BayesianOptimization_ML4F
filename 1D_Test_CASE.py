# -*- coding: utf-8 -*-
"""
1D test case for BO_ML4F — minimises f(x) = sin(5x)(1 - tanh(x²)) + ε on [-2, 2].

Run this script to perform the BO run.  Per-iteration snapshots are saved to
./out/states/state_NNN.npz.  Use make_animation_1D.py afterwards to generate
a GIF of the GP posterior and acquisition function evolution.

@author: mendez, lecomte
"""

import numpy as np
import BO_ML4F as bo

# -------------------------------------------------------------------
# Objective function
# -------------------------------------------------------------------
np.random.seed(237)

def func(x, noise_level=0.1):
    noise = np.random.randn() * noise_level
    return np.sin(5 * x) * (1 - np.tanh(x ** 2)) + noise


# -------------------------------------------------------------------
# BO configuration
# -------------------------------------------------------------------
LL = 2.0
bounds = [(-LL, LL)]

res = bo.bayesian_optimization(
    f         = func,
    bounds    = bounds,
    bo_cfg    = bo.BOConfig(
        n_init       = 5,
        n_iter       = 20,
        random_state = 1234,
    ),
    gp_cfg    = bo.GPConfig(),          # RBF kernel, fixed hyperparams
    acq_cfg   = bo.AcqConfig(),         # EI, xi=0.01
    optim_cfg = bo.OptimConfig(),       # random search, N=2000
    save_cfg  = bo.SaveConfig(
        save_path     = "./out",
        export_states = True,           # save .npz per iteration → use make_animation_1D.py
        log_enabled   = True,
    ),
)

print(f"\nDone.  Best x = {res.best_x},  best f = {res.best_y:.4f}")
print("Per-iteration snapshots saved to ./out/states/")
print("Run make_animation_1D.py to generate the animation.")
