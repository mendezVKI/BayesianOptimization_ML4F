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


#%% Define the noisy function to fit

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


res = bo.bayesian_optimization(
    f=func,
    bounds=bounds,
    bo_cfg=bo.BOConfig(random_state=237),
    gp_cfg=bo.GPConfig(),
    acq_cfg=bo.AcqConfig(),
    optim_cfg=bo.OptimConfig(),
    plt_cfg=bo.PlotConfig(plt_state_enabled=True,
                          state_save_path="./GIFs"),  
)

    



# plt.figure(figsize=(5, 3))
# plt.scatter(res.X, res.y)
# plt.xlabel("x"); plt.ylabel("y")
# plt.grid(True); plt.legend()
# plt.xlim(-LL - 0.1, LL + 0.1)
# plt.show()

