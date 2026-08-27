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
import bo_ml4f as bo

# Import a benchmark funciton 
from benchmarks import branin_2d

#Customization of the plot 
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=12)

# Define and create (if needed) the saving path
OUT_PATH = "./branin_2D_out"
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
plt.show()


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
        hpo_every=10,
        rank_one=True
        ),
    acq_cfg=bo.AcqConfig(xi=xi),
    optim_cfg=bo.OptimConfig(
        method = "random",
        global_method="refine"
        ),
    save_cfg=bo.SaveConfig(
        out_path = OUT_PATH,
        plt_all=True,
        plot_every=1
        )
)

rank_run_time = time.time() - start_time
print("------------------------------------")
print("Home-Made BO; With Rank-One Update")
print(f"  > Elapsed time = {rank_run_time:.4f} s")
print(f"  > Best x = {res.best_x[0]:.4f} - Best y {res.best_y:.4f}")

