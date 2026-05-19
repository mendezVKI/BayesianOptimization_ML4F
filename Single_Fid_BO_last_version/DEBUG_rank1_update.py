# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:10:15 2026

@author: Yannick Lecomte
"""


#%% Initialization

# Import main packages
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tqdm

# Import the home-made BO lib
import BO_ML4F as bo

# Customization of the plot 
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=12)

# Define and create (if needed) the saving path
OUT_PATH = "./BO_lib_res_1D"
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)


# Constant random seed to compare both implementation (with and without rank-1)
np.random.seed(237)


#%% LOOP OVER N_INIT
     

LL = 2.0 # size of the domain
bounds = [(-LL, LL)] # Define the bounds of the function

n_iter = 100  # this parameter is FIXED
xi = 0.01


# Define the list of initial set size over which to loop
n_init_min, n_init_max = 500, 1200
Ns = 10
n_init_list = np.round(np.linspace(n_init_min, n_init_max, Ns)).astype(int)

# Initialize the saveing lists
rank_time_list = []
norank_time_list = []

for n_init in tqdm.tqdm(n_init_list):

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
            optimize_hyperparams=False,
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
    norank_time_list.append(norank_run_time)
    
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
            optimize_hyperparams=False,
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
    rank_time_list.append(rank_run_time)


# Plot

plt.figure(figsize=(5,3))
plt.plot(n_init_list, norank_time_list, label="No Rank-1")
plt.plot(n_init_list, rank_time_list, label="Rank-1")
plt.legend(); plt.grid()
plt.xlabel("Initial set size, n_init")
plt.ylabel("Time [s]")
plt.title(f"Each case has n_iter={n_iter}, no HPO")
plt.savefig("rank_vs_norank.png", dpi=300, bbox_inches='tight')
plt.show()


#%% LOOP OVER N_ITER

LL = 2.0 # size of the domain
bounds = [(-LL, LL)] # Define the bounds of the function

n_init = 100  # this parameter is FIXED
xi = 0.01


# Define the list of initial set size over which to loop
n_init_min, n_init_max = 500, 1200
Ns = 10
n_init_list = np.round(np.linspace(n_init_min, n_init_max, Ns)).astype(int)

# Initialize the saving lists
rank_time_list = []
norank_time_list = []

for n_init in tqdm.tqdm(n_init_list):

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
            optimize_hyperparams=False,
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
    norank_time_list.append(norank_run_time)
    
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
            optimize_hyperparams=False,
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
    rank_time_list.append(rank_run_time)


# Plot
plt.figure(figsize=(5,3))
plt.plot(n_init_list, norank_time_list, label="No Rank-1")
plt.plot(n_init_list, rank_time_list, label="Rank-1")
plt.legend(); plt.grid()
plt.xlabel("Initial set size, n_init")
plt.ylabel("Time [s]")
plt.title(f"Each case has n_iter={n_iter}, no HPO")
plt.savefig("rank_vs_norank.png", dpi=300, bbox_inches='tight')
plt.show()



