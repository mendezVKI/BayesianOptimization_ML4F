# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:28:02 2026

@author: mendez, lecomte
"""

#%% Initialization

import numpy as np
import matplotlib.pyplot as plt

import BO_ML4F as bo

#Customization of the plot 
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=12)


#%% Define a 3D noisy function

def func(x, noise_level=0.1):
    x = np.asarray(x)

    noise = np.random.randn() * noise_level

    return (
        np.sin(3 * x[0])
        + 0.5 * np.cos(5 * x[1])
        + 0.2 * x[2]**2
        + noise
    )

# def func(x, noise_level=0.1):
#     x = np.asarray(x)
#     noise = np.random.randn() * noise_level
#     return np.sin(5 * x) * (1 - np.tanh(x ** 2)) + noise
          


#%% Generate and visualize a random set of initial point

n_plt = 500 # Number points of the plot 
LL = 2 # Size of the domain
bounds = [(-LL, LL), (-LL, LL), (-LL, LL)]



# Extract the BOConfig container from the lib
bo_cfg = bo.BOConfig(
    n_init=n_plt, 
    random_state=1234
    )
rng = bo._rng(bo_cfg.random_state)


X, y = bo.init_dataset(
    f=func,
    bounds=bounds,
    bo_cfg=bo_cfg,
    rng=rng
)
    
    
# --- Plot the points
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    X[:, 0],
    X[:, 1],
    X[:, 2],
    c=y,                 # color by output y
    cmap='viridis',      
    s=30,
    alpha=0.8
)

ax.set_xlabel(r"$x_0$")
ax.set_ylabel(r"$x_1$")
ax.set_zlabel(r"$x_2$")
ax.set_title("3D input space colored by output $y$")
# Colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.1)
cbar.set_label(r"$y$")
plt.show()



#%% Run Bayesian Optimization
res = bo.bayesian_optimization(
    f=func,
    bounds=bounds,
    bo_cfg=bo.BOConfig(
        n_init=10,
        n_iter=30,
        random_state=1234
    ),
    gp_cfg=bo.GPConfig(),
    acq_cfg=bo.AcqConfig(kind="EI"),
    optim_cfg=bo.OptimConfig(method="random"),
    save_cfg=bo.SaveConfig(
        plt_state_enabled=False,  # REQUIRED for d > 1
        save_path="./out_3d"
    ),
)

#%% Inspect results
print("Best x found:", res.best_x)
print("Best y found:", res.best_y)