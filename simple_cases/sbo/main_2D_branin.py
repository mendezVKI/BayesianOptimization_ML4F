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

# Make the `simple_cases` package importable, so this file runs both as
#     python -m simple_cases.sbo.main_2D_branin
# and directly (Spyder / VS Code "Run file"), from any working directory.
# ONLY the repository root is added -- never simple_cases/ itself -- so shared
# code is reached as `simple_cases.benchmarks`, never as a bare `benchmarks`.
# Importing `simple_cases` is what puts src/ on the path, so it has to come
# before the library import below. See simple_cases/README.md.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simple_cases import _common  # noqa: E402
from simple_cases.benchmarks import branin_2d  # noqa: E402
from pyRAMBO import sbo as bo  # noqa: E402

_common.use_paper_style()

# Every example writes here, whatever the working directory (gitignored).
OUT_PATH = _common.output_dir("sbo_branin_2D")


# Define a given random seed for reproducability
func_rng = 47
rng = np.random.default_rng(func_rng)

# Define the nois level
noise_level = 0.1

#%% Visualize the function to optimize

# Visualization of the generated noisy function
n_real = 100 # Number of training points
LL = 2 # Size of the domain

# Create the input mesh
x1 = np.linspace(-LL, LL, n_real)
x2 = np.linspace(-LL, LL, n_real)
X1, X2 = np.meshgrid(x1, x2)
# Build (n², 2) input
X_real = np.stack([X1.ravel(), X2.ravel()], axis=1)
# Evaluate function
y_true = branin_2d(X_real, noise_level=0)
# Reshape for visualization
Z = y_true.reshape(n_real, n_real)

# Get the min point location 
min_idx = np.argmin(Z)
X_min_true = X_real[min_idx]
y_min_true = y_true[min_idx]


# Plot
fig, ax = plt.subplots(figsize=(4, 4))
cs = ax.contourf(X1, X2, Z, levels=50)
cbar = plt.colorbar(cs)
ax.scatter(X_min_true[0], X_min_true[1], c='r', label='Minimum')
ax.legend()
ax.set_aspect('equal', adjustable='box')
# Save the fig
figname = os.path.join(OUT_PATH, "True function.png")
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.close()


#%% Run 
     
LL = 2.0 # size of the domain
bounds = [(-LL, LL), (-LL, LL)] # Define the bounds of the function

n_init = 50
n_iter = 10
xi = 0.01

f = lambda x: branin_2d(x, noise_level=noise_level, rng=rng)
f_true = lambda x: branin_2d(x, noise_level=0, rng=rng)

# -------- Run
start_time = time.time()

gp_cfg = bo.GPConfig(
    optimize_hyperparams=True,
    hpo_every=10,
)

save_cfg = bo.SaveConfig(
    out_path = OUT_PATH,
    run_naming="run_id",
    plt_all=True,
)

res = bo.bayesian_optimization(
    f=f,
    f_true=f_true,
    bounds=bounds,
    bo_cfg=bo.BOConfig(
        n_init=n_init,
        n_iter=n_iter,
        random_state=1234
        ),
    gp_cfg=gp_cfg,
    acq_cfg=bo.AcqConfig(xi=xi),
    optim_cfg=bo.OptimConfig(method = "random"),
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
