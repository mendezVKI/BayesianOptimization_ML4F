# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:28:02 2026

@author: Yannick Lecomte
"""

#%% Initialization

import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Make the `examples` package importable, so this file runs both as
#     python -m examples.sbo.main_1D_sin_large_scale
# and directly (Spyder / VS Code "Run file"), from any working directory.
# ONLY the repository root is added -- never examples/ itself -- so shared
# code is reached as `examples.benchmarks`, never as a bare `benchmarks`.
# Importing `examples` is what puts src/ on the path, so it has to come
# before the library import below. See examples/README.md.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simple_cases import _common  # noqa: E402
from simple_cases.benchmarks import sinusoidal_1d_large_scale  # noqa: E402
from pyRAMBO import sbo as bo  # noqa: E402

_common.use_paper_style()

# Every example writes here, whatever the working directory (gitignored).
OUT_PATH = _common.output_dir("sbo_sinus_1D_large_scale")


# Define a given random seed for reproducability
func_rng = 47
rng = np.random.default_rng(func_rng)

# Define the nois level
noise_level = 10

#%% Visualize the function to optimize

# Visualization of the generated noisy function
n_real = 100 # Number of training points
LL = 10.0 # Size of the domain
d=1 # dimension of the case
# Compute the input/output
X_real = np.linspace(-LL, LL, n_real)
y_true = sinusoidal_1d_large_scale(X_real, noise_level=0)
# Get the min point location 
min_idx = np.argmin(y_true)
X_min_true = X_real[min_idx]
y_min_true = y_true[min_idx]

# Plot
plt.figure(figsize=(5, 3))
plt.plot(X_real, y_true, linestyle="--", c='darkred', label="True function", zorder=2)
plt.scatter(X_min_true, y_min_true, marker="x", c="k", label="True min", zorder=3)
plt.fill_between(
    X_real,
    y_true - 1.96 * noise_level,
    y_true + 1.96 * noise_level,
    alpha=0.2,
    color="darkred",
    label=r"95\% noise band", 
    zorder=2
)
plt.xlabel("x"); plt.ylabel("y")
plt.grid(True); plt.legend()
plt.xlim(-LL - 0.1, LL + 0.1)
figname = os.path.join(OUT_PATH, f"True function_noise_level{noise_level}.png")
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.close()

#%% Run all the BO configuration and compare the computational time
     
LL = 10.0 # size of the domain
bounds = [(-LL, LL)] # Define the bounds of the function

n_init = 5
n_iter = 10
xi = 0.01

f = lambda x: sinusoidal_1d_large_scale(x, noise_level=noise_level, rng=rng)
f_true = lambda x: sinusoidal_1d_large_scale(x, noise_level=0, rng=rng)

# -------- Run WITH rank one update
start_time = time.time()

gp_cfg = bo.GPConfig(
    optimize_hyperparams=True,
    hpo_every=1,
    theta_bounds_log=[
        (-4.0, 0.0),
        (-3.0, 2.0),
        (-8.0, -1.0),
    ]
)

save_cfg = bo.SaveConfig(
    out_path = OUT_PATH,
    create_timestamp=False,
    plt_all=True,
)

res = bo.bayesian_optimization(
    f=f,
    f_true=f_true,
    bounds=bounds,
    bo_cfg=bo.BOConfig(
        n_init=n_init, 
        n_iter=n_iter, 
        ),
    gp_cfg=gp_cfg,
    acq_cfg=bo.AcqConfig(xi=xi),
    optim_cfg=bo.OptimConfig(
        method = "random",
        global_method="refine"
        ),
    save_cfg=save_cfg
)

run_time = time.time() - start_time
print("------------------------------------")
print("Home-Made BO")
print(f"  > Elapsed time = {run_time:.4f} s")
print(f"  > Best x = {res.best_x[0]:.4f} - Best y {res.best_y:.4f}")



#%% Create a GIF (only if plt_state_enabled = True or plt_all = True)
# Assembles <run folder>/plots/GIF/it_*.png into <run folder>/plots/evolution.gif
gif_path = bo.make_gif(res.out_path, fps=2)
print(f"  > GIF saved to {gif_path}")






