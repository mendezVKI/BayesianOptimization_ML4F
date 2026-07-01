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
import BO_func_YL as fct


#Customization of the plot 
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=12)

#%% STEP 1 - Generate Noisy Training Data Set

def func(x, noise_std=0.1):
    x = np.asarray(x).reshape(-1)
    noise = noise_std * np.random.randn(len(x))
    y = np.sin(1.5 * x) + 0.3 * np.sin(6.0 * x) + noise
    return y

n_train = 1 # Number of training points
LL = 1 # Size of the domain
X_train = np.random.uniform(-LL, LL, (n_train, 1 ))
y_train = func(X_train) 

# Plot the pionts distribution
plt.figure(figsize=(6, 2))
plt.scatter(X_train, y_train, marker="*", label="Training Points")
plt.xlabel("x"); plt.ylabel("y")
plt.grid(True); plt.legend()
plt.xlim(-LL, LL)
plt.show()


n_real = 100 # Number of training points
LL = 1 # Size of the domain
X_real = np.random.uniform(-LL, LL, (n_real, 1 ))
y_real = func(X_real) 

# Plot the pionts distribution
plt.figure(figsize=(6, 2))
plt.scatter(X_real, y_real, marker="*", label="Real Points")
plt.xlabel("x"); plt.ylabel("y")
plt.grid(True); plt.legend()
plt.xlim(-LL, LL)
plt.show()


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
prev_theta = res.x.copy() # Iinitial prediction for iteration 1 !! in log scale !!
# Plot the optimization history
fct.plot_hpo(MLE_history, param_history, -1, debug=True)

print(f"l={opt_theta[0]:.4f} - var={opt_theta[1]:.4f} - noise={opt_theta[2]:.4f}")



#%% STEP 3 - Check visualy how it currently looks

alpha, L = fct.gp_fit(X_train, y_train, opt_theta)

# Dense grid for visualization
n_test = 400
X_test = np.linspace(-LL, LL, n_test).reshape(-1, 1)

# Posterior predictions
mu , std, cov = fct.gp_predict(X_test, X_train, alpha, L, opt_theta, return_cov=True)

# Numerical stabilization
cov += 1e-6 * np.eye(len(X_test))

L_post = np.linalg.cholesky(cov)

n_samples = 40
f_post = mu.reshape(-1, 1) + L_post @ np.random.randn(len(X_test), n_test)
                                                      
plt.figure(figsize=(6, 2))
plt.scatter(X_train, y_train, marker="x", label="$X$", zorder=3)
plt.plot(X_test, mu, c="r", linestyle="--", zorder=2, label="$\\mathbf{\\mu}_{*}$")
plt.plot(X_test, f_post, alpha = 0.2, linewidth= 1, zorder=1)
plt.xlabel("x"); plt.ylabel("y")
plt.grid(True); plt.legend()
plt.xlim(-LL, LL)
plt.show()


#%% STEP 4 - Explore the domain by sampling new points

# Initialize the search domain
bounds = [(-LL, LL)] # Search domain  
xi = 0.01 # Exploration factor
n_iter = 10 # Number of BO iterations
n_init = 5  # Number of points for the initialization of the GPr
n_params = 1 # Re-optimization of the hyper-parameters

# List to save the optimization history 
conv_hist = []
explo_hist = []

bounds_MLE = [
    (-3.0,  1.0),   # log length-scale  → ℓ ∈ [0.05, 2.7]
    (-5.0,  3.0),   # log signal variance
    (-10.0, -2.0),  # log noise std     → σ_n ∈ [1e-5, 0.1]
]


# -----------------------------------------------------------------------------
# --- Start the optimization
# -----------------------------------------------------------------------------

X = X_train
y = y_train

for i in range(n_iter):
    print(f'Iteration {i+1}/{n_iter}')
    # Store the current best value for the convergence history
    conv_hist.append(np.max(y))
    print(f"  > Current best: y={np.min(y):.3f}")

    # -----------------------------------------------------------------
    # STEP 1: If requried, re-optimize the hyper-parameters
    # -----------------------------------------------------------------
    perform_hpo = (i % n_params == 0)
    if perform_hpo:
        MLE_history = []
        param_history = []
        # Minimize the log marginal likelihood
        res = minimize(
            fct.log_marginal_likelihood,   # function to minimize
            prev_theta,                    # initial parameters
            args=(X, y, MLE_history, param_history), 
            method='L-BFGS-B',             # could use other methods
            bounds=bounds_MLE
        )
        # The opti is based on a log function, so now we have to transform in physical terms
        theta = np.exp(res.x)
        prev_theta = res.x.copy()#  Iinitial prediction for iteration 1 !! in log scale !!
        print(f"  > New hyperparameters: l={theta[0]:.4f} - var={theta[1]:.4f} - noise={theta[2]:.4f}")
        # Plot and save the results 
        # fct.plot_hpo(MLE_history, param_history, i, debug=True)


    # -----------------------------------------------------------------
    # STEP 2: Evaluate the next point to sample
    # -----------------------------------------------------------------
    X_s = fct.propose_location(X, y, theta, bounds, xi, 
                                  grid_resolution=50, n_restarts=5)
    print(f"  > X_s={X_s}")

    
    # -----------------------------------------------------------------
    # STEP 3: Run the sequences at the next point
    # -----------------------------------------------------------------
    
    y_s = func(X_s)     # Run an epsiode and get the cumulative reward
    explo_hist.append(y_s)
    print(f"  > Reward at the next point: y_s={y_s}")

    # -----------------------------------------------------------------
    # STEP 5: Add the new exploration in the data set 
    # -----------------------------------------------------------------
    # Append physical data
    X = np.vstack((X, X_s))
    y = np.hstack((y, y_s))

    # -----------------------------------------------------------------
    # STEP 4: Plot the current prediction and the next sampling
    # -----------------------------------------------------------------
    
    # Dense grid for visualization
    n_test = 400
    
    X_test = np.linspace(-LL, LL, n_test).reshape(-1, 1)
    
    # Posterior predictions
    alpha, L = fct.gp_fit(X, y, opt_theta)

    mu , std, cov = fct.gp_predict(X_test, X, alpha, L, opt_theta, return_cov=True)
    
    # Numerical stabilization
    cov += 1e-6 * np.eye(len(X_test))
    
    L_post = np.linalg.cholesky(cov)
    
    n_samples = 40
    f_post = mu.reshape(-1, 1) + L_post @ np.random.randn(len(X_test), n_test)
                                                          
    plt.figure(figsize=(6, 2))
    plt.scatter(X, y, marker="x", label="$X$", zorder=3)
    plt.plot(X_test, mu, c="r", linestyle="--", zorder=2, label="$\\mathbf{\\mu}_{*}$")
    plt.plot(X_test, f_post, alpha = 0.2, linewidth= 1, zorder=1)
    plt.xlabel("x"); plt.ylabel("y")
    plt.grid(True); plt.legend()
    plt.xlim(-LL, LL); plt.ylim(-2, 2)
    plt.show()


print('Optimization done!')


# Plot the convergence
plt.figure(figsize = (5, 3))
plt.plot(np.arange(n_iter), conv_hist, '--bo')
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('min $\\mathcal{R}(\mathbf{w})$ over $n$ calls')
plt.show()




