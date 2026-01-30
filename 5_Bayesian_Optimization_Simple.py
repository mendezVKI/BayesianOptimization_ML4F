# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 10:35:23 2025

@author: mendez
"""

# This exercise is adapted from
# https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html

import numpy as np
np.random.seed(237)
import matplotlib.pyplot as plt
# This is a function to plot gaussian processes
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import cholesky, cho_solve
# Function for normalized gaussian
from scipy.stats import norm
import imageio.v2 as imageio
import os



#%% Cost function definition
np.random.seed(237)

def func(x, noise_level=0.1):
    noise = np.random.randn() * noise_level
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + noise
           
#%% Some home made functions 

# gp_fit: Fit the Gaussian Process
def gp_fit(x_s, y_s, l_c=0.3, sigma_y=0.1):
    # Compute the kernel matrix (training points only)
    K_ss = rbf_kernel(x_s, x_s, gamma=0.5 / l_c**2)
    # Cholesky decomposition
    L = cholesky(K_ss + sigma_y**2 * np.eye(len(x_s)), lower=True)
    # Solve for alpha
    alpha_v = cho_solve((L, True), y_s)
    return alpha_v, L

# gp_predict: Predictive mean and variance
def gp_predict(x, x_s, alpha_v, L, l_c=0.3, sigma_y=0.1):
    # Compute kernel matrices
    K = rbf_kernel(x, x, gamma=0.5 / l_c**2)
    K_s = rbf_kernel(x, x_s, gamma=0.5 / l_c**2)
    # Solve for K_alpha
    K_alpha = cho_solve((L, True), K_s.T)
    # Predictive mean and covariance
    mu_y = K_s @ alpha_v
    Sigma_yy = K - K_s @ K_alpha
    return mu_y, np.diag(Sigma_yy)


# Expected Improvement (EI) acquisition function
def expected_improvement(x, x_s, y_s, alpha_v, L, y_best, l_c=0.3, sigma_y=0.1, xi=0.01):
    # Reshape x for compatibility

    # Predictive mean and variance
    mu, sigma = gp_predict(x, x_s, alpha_v, L, l_c, sigma_y)
    # Normalize improvement
    with np.errstate(divide='warn'):
        Z = (y_best - mu - xi) / sigma
        ei = (y_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0  # Handle cases where sigma is zero
    return ei

    
def plt_state(x_s, y_s, mu_y, uncertainty, x_grid, EI, it, figname=None):
    # True (noisy) function
    y_true = [func(xi, noise_level=0) for xi in x_grid]

    best_idx = np.argmax(EI)
    a = EI[best_idx]
    # Plot
    fig, axs = plt.subplots(
        2, 1, figsize=(6, 5),
        constrained_layout=True,
        sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1])
    )

    # ---- Top: function + GP
    axs[0].plot(x_grid, y_true, "k--", lw=1.0, label="True (unknown)")
    axs[0].plot(x_grid, mu_y, "C0", lw=2, label="$\\mu_{\\mathcal{GP}}$")
    axs[0].fill_between(
        x_grid[:, 0],
        mu_y - uncertainty,
        mu_y +  uncertainty,
        color="C0",
        alpha=0.25,
        label="$\\mu_{\\mathcal{GP}} \\pm 2 \\sigma$",
    )
    axs[0].scatter(x_s, y_s, c="k", s=20, zorder=10, label="Observations")
    axs[0].axhline(np.min(y_s), c="r", linestyle="--", label="Current best $y_{best}$")
    axs[0].set_ylabel("f(x)")
    axs[0].set_title(f"Iteration {it}")
    axs[0].legend(fontsize=8, loc='upper right')

    # ---- Bottom: acquisition
    axs[1].plot(x_grid, EI, "C1", lw=1.5, label="EI")
    axs[1].fill_between(
        x= x_grid[:,0], 
        y1= EI, 
        color= "C1",
        alpha= 0.2
    )
    axs[1].scatter(x_next, a, c="C1", label='Next query point')
    axs[1].legend(fontsize=8, loc='lower right')

    axs[1].set_ylabel("acq(x)")
    axs[1].set_xlabel("x")

    if figname:
        plt.savefig(figname, dpi=250)
    plt.show()
    

#%% Run the BO and save the intermediate results
n_iter=15 # max number of iterations
n_ini=5 # initial number for the model definition
bounds=np.array([-2.0,2.0]) # bounds for the optimization
# Parameters of the GPR (very hard to guess!)
L_C=0.25
sigma_y=0.1
XI=0.01 # hyperparameter for the exploration

# define random initial samples:
x_init = np.random.uniform(bounds[0], bounds[1], size=(n_ini, 1))
y_init = np.array([func(x) for x in x_init])

# Initialize the set of points for the plotting
x_s,y_s= x_init,y_init
x_grid = np.linspace(bounds[0], bounds[1], 500).reshape(-1, 1)

# Prepare the convergence history
CONV=np.zeros(n_iter)


NAME_FOLDER='BO_simple_res_1D'
if not os.path.exists(NAME_FOLDER):
    os.makedirs(NAME_FOLDER)
    print(f"Folder '{NAME_FOLDER}' created.")
    
GIF_FOLDER=os.path.join(NAME_FOLDER, 'GIF')
if not os.path.exists(GIF_FOLDER):
    os.makedirs(GIF_FOLDER)
    print(f"GIF folder '{GIF_FOLDER}' created.")

frames = []

for i in range(n_iter): # BO loop
    # identify your current best solution
    y_best=np.min(y_s)
    CONV[i]=y_best
    # train the model on the current data available
    alpha_v, L = gp_fit(x_s, y_s, l_c=L_C, sigma_y=sigma_y) 
    # Show the current GPR on the grid (for plotting purposes only)
    mu_y, Sigma_y=gp_predict(x_grid,x_s, alpha_v, L, l_c=L_C, sigma_y=sigma_y )
    uncertainty = 1.96 * np.sqrt(Sigma_y)
    # Compute the EI on the current grid
    EI=expected_improvement(x_grid, x_s, y_s, alpha_v, L, y_best, l_c=L_C, sigma_y=sigma_y, xi=XI)
    # Identify the next best location (sticking to the grid!)
    x_next=x_grid[np.argmax(EI)] # this step should be replaced by an optimization of EI 
    # Plot the current state
    frame_path = os.path.join(GIF_FOLDER, f"it_{i:03d}.png")
    plt_state(x_s, y_s, mu_y, uncertainty, x_grid, EI, i, figname=frame_path)
    frames.append(imageio.imread(frame_path)) 
    # Evaluate the cost function in the new proposal
    y_next = func(x_next)
    # Update the available dataset
    x_s = np.vstack((x_s, x_next))
    y_s = np.append(y_s, y_next)
    # Plot current GP and acquisition        
    print(f"Iteration {i + 1}: x_next = {x_next}, y_next = {y_next}")


# Save the GIF
gif_path = os.path.join(NAME_FOLDER, "GIF.gif")
imageio.mimsave(gif_path, frames, duration=10.0)


# Plot the convergence
fig, ax = plt.subplots(figsize=(5, 3)) 
plt.plot(CONV,'ko:')
ax.set_xlabel('Iterations',fontsize=16)
ax.set_ylabel('min $f(x)$ after $n$ calls',fontsize=16)  
figname= os.path.join(NAME_FOLDER, 'BO_Convergence.png')
plt.savefig(figname,dpi=200,bbox_inches="tight")
plt.show()
plt.close('all')





    
