# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:28:02 2026

@author: mendez, lecomte
"""

#%% Initialization

# Import main packages
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the home-made BO lib
import BO_ML4F as bo

#Customization of the plot 
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=12)

# Define and create (if needed) the saving path
OUT_PATH = "./BO_lib_res_1D"
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

#%% Define the noisy function to fit
np.random.seed(237)

def func(x, noise_level=0.1):
    noise = np.random.randn() * noise_level
    return np.sin(5 * x) * (1 - np.tanh(x ** 2)) + noise
           
              
# Visualization of the generated noisy function
n_real = 100 # Number of training points
LL = 2 # Size of the domain
d=1 # dimension of the case
X_real = np.linspace(-LL, LL, n_real)
y_real = func(X_real) 

# Plot
plt.figure(figsize=(5, 3))
plt.scatter(X_real, y_real, c='darkred', marker="x", label="Real Points")
plt.xlabel("x"); plt.ylabel("y")
plt.grid(True); plt.legend()
plt.xlim(-LL - 0.1, LL + 0.1)
plt.show()

#%%

LL = 2 # size of the domain
bounds = [(-LL, LL)] # Define the bounds of the function

# Config of the confgiuration for the BO acquisition
bo_cfg = bo.BOConfig(n_init=5, n_iter=15)

# Config of the search acquisition
acq_cfg = bo.AcqConfig(xi=0.01)


res = bo.bayesian_optimization(
    f=func,
    bounds=bounds,
    bo_cfg=bo_cfg,
    gp_cfg=bo.GPConfig(),
    acq_cfg=acq_cfg,
    optim_cfg=bo.OptimConfig(),
    save_cfg=bo.SaveConfig(plt_state_enabled=True,
                          save_path=OUT_PATH),  
)

    


# plt.figure(figsize=(5, 3))
# plt.scatter(res.X, res.y)
# plt.xlabel("x"); plt.ylabel("y")
# plt.grid(True); plt.legend()
# plt.xlim(-LL - 0.1, LL + 0.1)
# plt.show()

