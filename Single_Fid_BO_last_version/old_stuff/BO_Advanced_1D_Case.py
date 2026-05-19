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
import time

# Scikit-optimize lib
# from skopt import gp_minimize
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

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

# def func(x, noise_level=0):
#     x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    
#     noise = np.random.randn() * noise_level
    
#     # 1D case
#     val = np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2))
    
#     return float(val + noise)

def func(x, noise_level=0):
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    
    noise = np.random.randn() * noise_level
    
    # 1D case
    val = np.sin(5 * x) * (1 - np.tanh(x ** 2))
    
    return val + noise

    
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

#%% Run all the BO configuration and compare the computational time
     

LL = 2.0 # size of the domain
bounds = [(-LL, LL)] # Define the bounds of the function

n_init = 5
n_iter = 100
total_calls = n_init + n_iter
xi = 0.01

# -------- Run WITHOUT rank one update
start_time = time.time()

res = bo.bayesian_optimization(
    f=func,
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
    save_cfg=bo.SaveConfig(),  
)

rank_run_time = time.time() - start_time
print("------------------------------------")
print("Home-Made BO; Without Rank-One Update")
print(f"  > Elapsed time = {rank_run_time:.4f} s")
print(f"  > Best x = {res.best_x[0]:.4f} - Best y {res.best_y:.4f}")

# -------- Run WITH rank one update
start_time = time.time()

res = bo.bayesian_optimization(
    f=func,
    bounds=bounds,
    bo_cfg=bo.BOConfig(
        n_init=n_init, 
        n_iter=n_iter, 
        random_state=1234
        ),
    gp_cfg=bo.GPConfig(
        optimize_hyperparams=True,
        hpo_every=10,
        rank_one=False
        ),
    acq_cfg=bo.AcqConfig(xi=xi),
    optim_cfg=bo.OptimConfig(
        method = "random",
        global_method="refine"
        ),
    save_cfg=bo.SaveConfig(),  
)

norank_run_time = time.time() - start_time
print("------------------------------------")
print("Home-Made BO; With Rank-One Update")
print(f"  > Elapsed time = {norank_run_time:.4f} s")
print(f"  > Best x = {res.best_x[0]:.4f} - Best y {res.best_y:.4f}")




#%%
from skopt.learning.gaussian_process.kernels import Matern

kernel = 1.0 * Matern(
    length_scale=0.3,
    length_scale_bounds=(1e-3, 1e3),
    nu=2.5
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=False,
    noise=0.1**2,     # skopt handles noise separately
    random_state=1234
)


opt = Optimizer(
    dimensions=bounds,
    base_estimator=gp,
    acq_func="EI",
    acq_func_kwargs={"xi": 0.01},
    random_state=1234
)

start_time = time.time()

for i in range(total_calls):

    x = opt.ask()
    y = func(x)
    opt.tell(x, y)

    # Hyperparameter optimization every 10 iterations
    if i % 10 == 0 and i > 0:
    
        model = opt.base_estimator_
    
        model.optimizer = "fmin_l_bfgs_b"
        model.n_restarts_optimizer = 5
    
        model.fit(opt.Xi, opt.yi)
    
        model.optimizer = None


scipy_run_time = time.time() - start_time

res = opt.get_result()

print("------------------------------------")
print("SKOPT BO (Matched Settings)")
print(f"  > Elapsed time = {scipy_run_time:.4f} s")
print(f"  > Best x = {res.x[0]:.3f} - Best y {res.fun:.3f}")



# # -------- Run using the LIBRARY
# start_time = time.time()

# res = gp_minimize(func,                  # the function to minimize
#                   bounds,      # the bounds on each dimension of x
#                   acq_func="EI",      # the acquisition function
#                   n_calls=total_calls,         # the number of evaluations of f
#                   n_random_starts=n_init,  # the number of random initialization points
#                   xi=xi,
#                   random_state=1234)   # the random seed

# scipy_run_time = time.time() - start_time
# print("------------------------------------")
# print("SKOPT BO")
# print(f"  > Elapsed time = {scipy_run_time:.4f} s")
# print(f"  > Best x = {res.x[0]} - Best y  {res.fun}")


