# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 11:03:57 2026

@author: lyann
"""


from scipy.linalg import cholesky, cho_solve
import numpy as np
from scipy.stats import norm
import itertools
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter


def rbf_kernel(X1, X2, length_scale, variance):
    sqdist = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)
    return variance * np.exp(-0.5 * sqdist / length_scale**2)



# def matern32_ard_kernel(X1, X2, length_scale, variance):

#     # Scale each dimension separately
#     X1_scaled = X1 / length_scale
#     X2_scaled = X2 / length_scale

#     sqdist = np.sum((X1_scaled[:, None, :] - X2_scaled[None, :, :])**2, axis=2)
#     r = np.sqrt(sqdist + 1e-12)

#     sqrt3 = np.sqrt(3.0)
#     return variance ** 2 * (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)


def gp_fit(X_train, y_train, params):
    # Extract the hyper-parameters
    length_scale, variance, noise_std = params
    
    K = rbf_kernel(X_train, X_train, length_scale, variance) + noise_std**2 * np.eye(len(X_train))
    
    L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), y_train)
    return alpha, L


def gp_predict(X_test, X_train, alpha, L, params):
    # Extract the hyper-parameters
    length_scale, variance, noise_std = params

    K_s = rbf_kernel(X_test, X_train, length_scale, variance)
    K_ss = rbf_kernel(X_test, X_test, length_scale, variance)
    v = cho_solve((L, True), K_s.T)
    mu = K_s @ alpha
    cov = K_ss - K_s @ v  # Covariance matrix
    var = np.diag(cov)    # Variance matrix (sigma **2)
    std  = np.sqrt(var)   # Standard deviation sigama
    return mu, std

def log_marginal_likelihood(params, X_train, y_train, MLE_hist, param_hist):
    length_scale, variance, noise_std = np.exp(params)
    K = rbf_kernel(X_train, X_train, length_scale, variance) +  noise_std**2 * np.eye(len(X_train))
    try:
        L = cholesky(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e6
    alpha = cho_solve((L, True), y_train)
    ll = -0.5 * y_train.T @ alpha - np.sum(np.log(np.diagonal(L))) - 0.5 * len(y_train) * np.log(2 * np.pi)
    
    # Add the values to the history arrays
    MLE_hist.append(ll)
    param_hist.append([length_scale, variance, noise_std])
    
    return -ll  # minimize negative log marginal likelihood

def EI(X_test, X_train, y_train, params, xi, alpha, L):
    X_test = np.atleast_2d(X_test)  # <- ensures it's (1, D) not (D,)
	# Compute the prediction based on the current GP
    mu , std = gp_predict(X_test, X_train, alpha, L, params)
    # Compute current best from the high-fidelity
    y_best = np.min(y_train)
    if std == 0:
        return 0.0
    # Compute the EI with the given formula
    Z = (y_best - mu - xi) / std
    return (y_best - mu - xi) * norm.cdf(Z) + std * norm.pdf(Z)


def propose_location(X_train, y_train, params, bounds, xi, method="Classic",
                     grid_resolution=200, n_restarts=5, R=0.25):

    # === Step 1: Coarse Grid Search ===
    grid_axes = [np.linspace(b[0], b[1], grid_resolution) for b in bounds]
    grid_points = np.array(list(itertools.product(*grid_axes)))

    alpha, L = gp_fit(X_train, y_train, params)
    ei_values = np.array([
        EI(x, X_train, y_train, params, xi, alpha, L)[0]
        for x in grid_points
    ])

    best_idx = np.argmax(ei_values)
    x_best = grid_points[best_idx] 

    # === Step 2: Optional local refinement ===
    if method == "Hybrid":
        min_val = np.inf
        x_refined = x_best.copy()

        for _ in range(n_restarts):
            # Define local bounds box centered around the circle knowing the radius
            local_bounds = [(max(b[0], c - R), min(b[1], c + R))
                            for b, c in zip(bounds, x_best)]
            # Generate a random initial start point inside this point
            x0 = random_point_within_circle(x_best, R=R)
            # Maximize EI with BFGS
            res = minimize(
                lambda x: -EI(x, X_train, y_train, params, xi)[0],
                x0=x0,
                bounds=local_bounds,
                method="L-BFGS-B"
            )
            # If it found a better maximium locaiton, replace it 
            if res.fun < min_val:
                min_val = res.fun
                x_refined = res.x

        return x_refined

    return x_best



def random_point_within_circle(center_point, R=0.25):
    # Random radius (sqrt for uniform distribution in area)
    r = R * np.sqrt(np.random.uniform(0, 1))
    # Random angle between 0 and 2*pi
    theta = np.random.uniform(0, 2 * np.pi)
    # Polar to Cartesian
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    # Shift to center
    point = center_point + np.array([dx, dy])
    return point


def plot_hpo(log_likelihood_history, param_history, iteration,
                    debug=False, PLT_PATH=None, DATA_PATH=None):
    # --- Plot
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(log_likelihood_history, 'o-')
    axs[0].set_title("Log Marginal Likelihood")
    axs[0].set_xlabel("Iteration")
    axs[0].grid(True)
    
    param_arr = np.array(param_history)
    labels = ["$\\ell$", "$\\sigma_f^2$", "$\\sigma_n$"]
    for j in range(param_arr.shape[1]):
        axs[1].plot(param_arr[:, j], label=labels[j] + f' = {param_arr[-1, j]:.4f}')
        
    axs[1].legend()
    axs[1].set_title("Parameter History")
    axs[1].grid(True)
    
    fig.suptitle(f"Iteration {iteration+1}", fontsize=16)
    fig.tight_layout()
    # Save the figure only if the path is specified
    if DATA_PATH:
        file_PATH = os.path.join(PLT_PATH, 'HPO_plots')
        os.makedirs(file_PATH, exist_ok=True)
        file_NAME = os.path.join(file_PATH, f'Iteration_{iteration+1:02d}_HPO.png')
        fig.savefig(file_NAME, dpi=300, bbox_inches='tight')
    # Show the plot only if specified
    elif debug:
        plt.show()
        
    # --- Save data
    if DATA_PATH:
        data_PATH = os.path.join(DATA_PATH, 'HPO_data')
        os.makedirs(data_PATH, exist_ok=True)
        np.save(os.path.join(data_PATH, f'Iteration_{iteration+1:02d}_log_likelihood.npy'), log_likelihood_history)
        np.save(os.path.join(data_PATH, f'Iteration_{iteration+1:02d}_param_history.npy'), param_arr)



def plot_n_save_BO(X_train, y_train, params, n_s_hf, x_next, bounds, plt_path,
                   data_path, xi, iteration, resolution=50):
    
    # Create the grid
    kp = np.linspace(bounds[0][0], bounds[0][1], resolution)
    kd = np.linspace(bounds[1][0], bounds[1][1], resolution)
    KP, KD = np.meshgrid(kp, kd)
    X_grid = np.column_stack((KP.ravel(), KD.ravel()))

	# Predict the mean and cov on the grid points set
    alpha, L = gp_fit(X_train, y_train, params)
    mu, std = gp_predict(X_grid, X_train, alpha, L, params)
    
    # Compute the best result -> from HIGH fidelity
    y_best = np.min(y_train)
    # Best point location
    best_idx = np.argmin(y_train)
    best_point = X_train[best_idx]
    
    # EI computation over the dense grid
    Z = (y_best - mu - xi) / std
    EI = (y_best - mu - xi) * norm.cdf(Z) + std * norm.pdf(Z)
    EI[std == 0.0] = 0.0

    # Reshape for plotting
    mu_grid = mu.reshape(KP.shape)
    std_grid = std.reshape(KP.shape)
    ei_grid = EI.reshape(KP.shape)

    # Plot setup
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['Mean Prediction $\\mu(K_p, K_d)$',
              'Uncertainty $\\sigma(K_p, K_d)$',
              'Expected Improvement (EI)']
    color_maps = ['viridis', 'plasma',  'cividis']
    data = [mu_grid, std_grid, ei_grid]

    for ax, d, title, col_map in zip(axs, data, titles, color_maps):

        c = ax.contourf(KP, KD, d, levels=100, cmap=col_map)
        # Plot explored points, next, best current (unchanged)
        ax.scatter(X_train[:n_s_hf, 0], X_train[:n_s_hf, 1], c='green', edgecolor='k', s=40, label='High-Fidelity Training')
        ax.scatter(X_train[n_s_hf:, 0], X_train[n_s_hf:, 1], c='blue', edgecolor='k', s=40, label='High-Fidelity Explored')
        ax.scatter(best_point[0], best_point[1], c='blue', edgecolor='k', marker='D', s=40, label='Best')
        ax.scatter(x_next[0], x_next[1], c='white', edgecolor='k', marker='*', s=100, label='Next')

        ax.set_title(title)
        ax.set_xlabel('$K_p$')
        ax.set_ylabel('$K_d$')
        cbar = fig.colorbar(c, ax=ax)
        
        if title == 'Expected Improvement (EI)':
            cbar.formatter = FuncFormatter(lambda x, _: f'{x:.2e}')
            cbar.update_ticks()
    # Add spacing below the plots
    fig.subplots_adjust(bottom=0.25, wspace=0.3)

    # Add a single shared legend below all subplots
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.05))
    
    #plt.tight_layout()

    # Save plot
    plt_file_path = os.path.join(plt_path, "BO_maps")
    os.makedirs(plt_file_path, exist_ok=True)
    filename = f"BO_maps_iter_{iteration:03d}.png"
    full_path = os.path.join(plt_file_path, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # ==== Save Inputs ====
    inputs_data = {
        'X_H': X_train,
        'L' : L, 
        'alpha': alpha,
        'x_next': x_next,
        'bounds': bounds,
        'n_s_hf': n_s_hf,
        'xi': xi,
        'resolution': resolution,
        'iteration': iteration,
        'y_best': y_best
    }

    # Save as compressed npz file
    inputs_filename = os.path.join(data_path, f'iteration_{iteration}',)
    os.makedirs(inputs_filename, exist_ok=True)
    inputs_path = os.path.join(inputs_filename,  "BO_inputs.npz")
    np.savez_compressed(inputs_path, **inputs_data)