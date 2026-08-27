# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:28:02 2026

@author: Yannick Lecomte
"""

#%% Initialization

# Import main packages
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Import the home-made BO lib
import BO_ML4F_v0_3 as bo

# Import a benchmark funciton 
from benchmarks import sinusoidal_1d

#Customization of the plot 
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=12)

# Define and create (if needed) the saving path
OUT_PATH = "./sinus_1D_out"
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

# Define a given random seed for reproducability
func_rng = 47
rng = np.random.default_rng(func_rng)

# Define the nois level
noise_level = 0.1

#%% Visualize the function to optimize

# Visualization of the generated noisy function
n_real = 100 # Number of training points
LL = 2 # Size of the domain
d=1 # dimension of the case
# Compute the input/output
X_real = np.linspace(-LL, LL, n_real)
y_true = sinusoidal_1d(X_real, noise_level=0)
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
    label="95\% noise band", 
    zorder=2
)
plt.xlabel("x"); plt.ylabel("y")
plt.grid(True); plt.legend()
plt.xlim(-LL - 0.1, LL + 0.1)
figname = os.path.join(OUT_PATH, "True function.png")
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()


#%% Run all the BO configuration and compare the computational time
     
LL = 2.0 # size of the domain
bounds = [(-LL, LL)] # Define the bounds of the function

n_init = 5
n_iter = 10
xi = 0.01

f = lambda x: sinusoidal_1d(x, noise_level=noise_level, rng=rng)
f_true = lambda x: sinusoidal_1d(x, noise_level=0, rng=rng)

# -------- Run WITH rank one update
start_time = time.time()

res = bo.bayesian_optimization(
    f=f,
    f_true=f_true,
    bounds=bounds,
    bo_cfg=bo.BOConfig(
        n_init=n_init, 
        n_iter=n_iter, 
        random_state=1234
        ),
    gp_cfg=bo.GPConfig(
        optimize_hyperparams=True,
        hpo_every=1,
        ),
    acq_cfg=bo.AcqConfig(xi=xi),
    optim_cfg=bo.OptimConfig(
        method = "random",
        global_method="refine"
        ),
    save_cfg=bo.SaveConfig(
        out_path = OUT_PATH,
        export_enabled=False, 
        plt_all=True,
        plot_every=10
        )
)

rank_run_time = time.time() - start_time
print("------------------------------------")
print("Home-Made BO; With Rank-One Update")
print(f"  > Elapsed time = {rank_run_time:.4f} s")
print(f"  > Best x = {res.best_x[0]:.4f} - Best y {res.best_y:.4f}")


#%%
# # -------- Run WITHOUT rank one update
# start_time = time.time()

# res = bo.bayesian_optimization(
#     f=f,
#     bounds=bounds,
#     bo_cfg=bo.BOConfig(
#         n_init=n_init, 
#         n_iter=n_iter, 
#         random_state=1234
#         ),
#     gp_cfg=bo.GPConfig(
#         optimize_hyperparams=True,
#         hpo_every=10,
#         rank_one=False
#         ),
#     acq_cfg=bo.AcqConfig(xi=xi),
#     optim_cfg=bo.OptimConfig(
#         method = "random",
#         global_method="refine"
#         ),
#     plt_cfg=bo.PlotConfig(),  
# )

# norank_run_time = time.time() - start_time
# print("------------------------------------")
# print("Home-Made BO; Without Rank-One Update")
# print(f"  > Elapsed time = {norank_run_time:.4f} s")
# print(f"  > Best x = {res.best_x[0]:.4f} - Best y {res.best_y:.4f}")


