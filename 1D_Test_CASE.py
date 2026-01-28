# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 10:39:11 2026

@author: mendez
"""

#%% STEP 0 - Initialization

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve

# Import the functions of the home-made BO
import functions_control_Burgers as fct


#Customization of the plot 
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=12)

#%% STEP 1 - Generate Training Data Set - noisy

def true_func(x):
    x = np.asarray(x).reshape(-1)

    return (
        np.sin(1.5 * x)
        + 0.3 * np.sin(6.0 * x)
    )

np.random.seed(0)
n_train = 5
noise_std = 0.1  # known noise level
LL = 1 # Size of the domain
X_train = np.random.uniform(-LL, LL, (n_train, 1 ))
y_train = true_func(X_train) + noise_std * np.random.randn(n_train)


plt.scatter(X_train, y_train); plt.show()


#%% STEP 2 - Tune the hyperparamters

MLE_history = []
param_history = []

# The initial guess it filled with the HPO run separatly only on the model 
params_init = np.log([1.0, 1.0, 0.1])
# Minimize the log marginal likelihood
res = minimize(
    fct.log_marginal_likelihood,   # function to minimize
    params_init,               # initial parameters
    args=(X_train, y_train, MLE_history, param_history), 
    method='L-BFGS-B'          # could use other methods
)
# The opti is based on a log function, so now we have to transform in physical terms
opt_theta = np.exp(res.x)
# Plot the optimization history
fct.plot_hpo(MLE_history, param_history, -1, debug=True)

print(f"l={opt_theta[0]:.4f} - var={opt_theta[1]:.4f} - noise={opt_theta[2]:.4f}")







