# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:35:19 2025

@author: Y. Lecomte
"""

#%% Import the packages========================================================

# import time
# import smbus2
# import math
# import board
# import busio
# from adafruit_ads1x15.ads1115 import ADS1115
# from adafruit_ads1x15.analog_in import AnalogIn
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import os
import struct
from tqdm import tqdm
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver
from skopt import load
import joblib
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import trange
import itertools
import pickle  
from scipy.linalg import cholesky, cho_solve
# from rpi_hardware_pwm import HardwarePWM
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter

#%% Initialization ============================================================

gamma_ref = 40

# ==== Define system constants =====
# With the panels
#theta_0 = 5.597469
#C = -4.712508

# Without the panels
# from campaign 4
theta_0 = 2.38018978
C = -1.54209496


# Calibration loading
CALIB_PATH_ENC = "data_calib_encoder/calib_value.txt"
theta_calib_enc = np.loadtxt(CALIB_PATH_ENC)

gamma_clip = 8
#%% BALANCE FUNCTIONS =========================================================


class AMT23AngleSensor:
    def __init__(self):
        # I2C configuration
        self.address = 0x04  # Same as set on the Arduino
        self.bus = smbus2.SMBus(1)
        self.rx_bytes = [0, 0, 0, 0]
        self.angle = 0.0

    def read_angle(self):
        try:
            self.rx_bytes = self.bus.read_i2c_block_data(self.address, 0, 4)
            value = struct.unpack('<f', bytes(self.rx_bytes))[0]
        except Exception as e:
            print(f"I2C read error: {e}")
            value = 0

        self.angle = float(value) /4

        if 0.0 <= self.angle <= 360.0:
            return self.angle
        else:
            return 0.0  # Clamp out-of-range values


def setup():    
    # ===== motor setup ===== 
    global pL
    global pR
    pL = HardwarePWM(pwm_channel=1, hz = 50, chip=2)    #GPIO13
    pR = HardwarePWM(pwm_channel=2, hz = 50, chip=2)    #GPIO18
    pL.start(0)
    pR.start(0)
    print('------Arming ESC------')
    t1 = time.time()
    while (time.time() - t1 <= 3):
        pL.change_duty_cycle(5.5)  
        pR.change_duty_cycle(5.5)
        time.sleep(0.5) 
    return pL, pR


def unmap_pwm(pwm):
    return 0.05*pwm + 5


def loop(X, target_list, encoder, Ns):
    # ==== Unmap the controller ====
    Kp, Kd = X

    # ===== Fixed time step =====
    dt = 1/400  
    duration = dt * Ns
    
    # ===== Initialize lists =====
    times = np.zeros(Ns)
    theta = np.zeros(Ns)
    error = np.zeros(Ns)
    gamma_control = np.zeros(Ns)
    gamma_cmd_L = np.zeros(Ns)
    gamma_cmd_R = np.zeros(Ns)

    times[0] = 0
    threshold = 10
    derivative_prev = 0
    # Warm up at the frist value of the target list 
    # before starting a full sequence
    t1 = time.time()
    while (time.time() - t1 <= 10):
        delta_gamma_S_start = (target_list[0] - theta_0)/C
        gamma_cmd_L_start = gamma_ref - 0.5 * delta_gamma_S_start
        gamma_cmd_R_start = gamma_ref + 0.5 * delta_gamma_S_start
        pL.change_duty_cycle(unmap_pwm(gamma_cmd_L_start))
        pR.change_duty_cycle(unmap_pwm(gamma_cmd_R_start))
    
    # Start the sequence
    with tqdm(total=duration, desc="Control", unit="s") as pbar:
        for i in range(1, Ns):
            loop_start = time.time()
                        
            # ===== Measure current position =====
            theta_val = encoder.read_angle() - theta_calib_enc
            if abs(theta_val - theta[i-1]) > threshold:
                theta[i] = theta[i-1]
            else:
                theta[i] = theta_val
            
            # ===== Time update =====
            times[i] = times[i-1] + dt

            # ===== Compute PD error =====
            error[i] = theta[i] - target_list[i]
            # derivative = (error[i] - error[i-1]) / dt
            
            # Evaluate the filtered error value 
            # alpha = tau_d / (tau_d + dt)
            alpha = 0.01
            derivative_filtered = alpha * derivative_prev + (1 - alpha) * ((error[i] - error[i-1]) / dt)
            derivative_prev = derivative_filtered
            # ===== Compute error to apply to each propeller ====
            delta_gamma_S = (target_list[i] - theta_0)/C
            gamma_control[i] = np.clip(delta_gamma_S + (Kp * error[i] + Kd * derivative_filtered), delta_gamma_S-gamma_clip, delta_gamma_S+gamma_clip)
            gamma_cmd_L[i] = np.clip((gamma_ref - 0.5 * gamma_control[i]), 20, 60)
            gamma_cmd_R[i] = np.clip((gamma_ref + 0.5 * gamma_control[i]), 20, 60)
            
            # ===== Apply to motors =====
            pL.change_duty_cycle(unmap_pwm(gamma_cmd_L[i]))
            pR.change_duty_cycle(unmap_pwm(gamma_cmd_R[i]))

            # ===== tqdm display =====
            pbar.set_postfix(theta=f"{theta[i]:.2f}°")
            pbar.update(dt)

            # ===== Wait to enforce constant dt =====
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    return times, theta


# Define the function that computes the error of a complete sequence
def system_cost(X, target_list, cost_params, cost_type, encoder, Ns):
    # Run the entire sequence to obtain the list of real theta positions
    time_list, theta_list = loop(X, target_list, encoder, Ns)
    
    if cost_type == 'penalty':
        # Unpack the cost parameters
        alpha_p, alpha_d = cost_params
        # Unpack the controller coefficients
        kp, kd = X
        # Compute the error with penalty terms
        err_ = np.linalg.norm(theta_list - target_list) / np.sqrt(Ns)
        err = err_ + alpha_p*kp + alpha_d*kd

    elif cost_type == 'classical':
        # Compute the error with without penalty terms
        err = np.linalg.norm(theta_list - target_list) / np.sqrt(Ns)
    return err, time_list, theta_list 


#%% Multi-Fidelity GPR functions ====================================================


def rbf_kernel(X1, X2, length_scale, variance):
	sqdist = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)
	return variance * np.exp(-0.5 * sqdist / length_scale**2)


def build_joint_covariance(X_L, X_H, theta, rho):
    """Build full joint covariance matrix for low- and high-fidelity data"""
    length_lf, var_lf, length_d, var_d, sigma_L, sigma_H = theta
    
    # Covariance blocks
    K_LL = rbf_kernel(X_L, X_L, length_lf, var_lf) + sigma_L**2 * np.eye(len(X_L))
    K_LH = rbf_kernel(X_L, X_H, length_lf, var_lf)
    K_HL = K_LH.T
    K_HH = rho**2 * rbf_kernel(X_H, X_H, length_lf, var_lf) \
           + rbf_kernel(X_H, X_H, length_d, var_d) + sigma_H**2 * np.eye(len(X_H))

    # Assemble full covariance matrix
    top = np.hstack([K_LL, rho * K_LH])
    bottom = np.hstack([rho * K_HL, K_HH])
    return np.vstack([top, bottom])


def mf_gp_fit(X_L, y_L, X_H, y_H, rho, theta):
    """
    Fit the multi-fidelity GP.
    Returns: (alpha, L) which are reused in prediction.
    """
    y = np.concatenate([y_L, y_H])
    K = build_joint_covariance(X_L, X_H, theta, rho)
    L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), y)
    return alpha, L

def mf_gp_predict(X_test, X_L, X_H, rho, theta, alpha, L):
    """
    Predict at new test points using fitted MF-GP (with precomputed alpha).
    """
    length_lf, var_lf, length_d, var_d, sigma_L, sigma_H = theta

    # Cross-covariance terms between test points and training points
    K_s_L = rbf_kernel(X_test, X_L, length_lf, var_lf)
    K_s_H_lf = rbf_kernel(X_test, X_H, length_lf, var_lf)
    K_s_H_d = rbf_kernel(X_test, X_H, length_d, var_d)

    # Combined cross-covariance
    K_s = np.hstack([rho * K_s_L, rho**2 * K_s_H_lf + K_s_H_d])

    # Prior covariance between test points
    K_ss_lf = rbf_kernel(X_test, X_test, length_lf, var_lf)
    K_ss_d = rbf_kernel(X_test, X_test, length_d, var_d)
    K_ss = rho**2 * K_ss_lf + K_ss_d
    
    v = cho_solve((L, True), K_s.T)
    mu = K_s @ alpha
    cov = K_ss - K_s @ v
    return mu , np.clip(np.sqrt(np.diag(cov)), 0, None)


MLE_hist = []
param_hist = []



def log_marginal_likelihood(params, X_L, y_L, X_H, y_H, MLE_hist, param_hist):
    """Compute negative log marginal likelihood (to minimize)"""
    rho = params[0]                    # scalar autoregressive parameter
    theta = np.exp(params[1:])         # kernel params (in log space)
    y = np.concatenate([y_L, y_H])     # combined output vector
    K = build_joint_covariance(X_L, X_H, theta, rho)  # full covariance
    # Idea: if kernels are not positive definite, put a big penalty.
    try:
        L = cholesky(K, lower=True)    # Cholesky decomposition
    except np.linalg.LinAlgError:
        return 1e6                     # return large value if matrix not PD

    alpha = cho_solve((L, True), y)
    ll = -0.5 * y.T @ alpha - np.sum(np.log(np.diagonal(L))) - 0.5 * len(y) * np.log(2 * np.pi)
    
    # Add the values to the history arrays
    MLE_hist.append(ll)
    param_hist.append([rho] + list(theta))
    return -ll  # minimize negative log likelihood




#%% Bayesian Opti functions ===================================================


def EI(X_test, X_L, y_L, X_H, y_H, rho, theta, xi):
    X_test = np.atleast_2d(X_test)  # <- ensures it's (1, D) not (D,)
	# Compute the prediction based on the current GP
    alpha, L = mf_gp_fit(X_L, y_L, X_H, y_H, rho, theta)
    mu , std = mf_gp_predict(X_test, X_L, X_H, rho, theta, alpha, L)
    # Compute current best from the high-fidelity
    y_best = np.min(y_H)
    if std == 0:
        return 0.0
    # Compute the EI with the given formula
    Z = (y_best - mu - xi) / std
    return (y_best - mu - xi) * norm.cdf(Z) + std * norm.pdf(Z)


def cost_aware_EI(X_test, X_L, y_L, X_H, y_H, rho, theta, xi, cost_L, cost_H):
	# Compute the classic expected improvement 
    ei = EI(X_test, X_L, y_L, X_H, y_H, rho, theta, xi)
    # Normalize by the respectiv cost
    ei_L = ei/cost_L
    ei_H = ei/cost_H
    return ei_L, ei_H


def propose_location_hybrid(X_L, y_L, X_H, y_H, rho, theta, bounds, xi, grid_resolution=40, n_restarts=5):

    # === Step 1: Coarse Grid Search ===
    grid_axes = [np.linspace(b[0], b[1], grid_resolution) for b in bounds]
    grid_points = np.array(list(itertools.product(*grid_axes)))

    ei_values = np.array([
        cost_aware_EI(x, X_L, y_L, X_H, y_H, rho, theta, xi, cost_L=1, cost_H=1)[0]
        for x in grid_points
    ])

    max_idx = np.argmax(ei_values)
    grid_best_x = grid_points[max_idx]

    # === Step 2: Local BFGS optimization within local box ===
    min_val = 1e20
    min_x = None

    for _ in range(n_restarts):
        # Random initial guess inside local_bounds
        R = 0.25 # radius of the circle around the opti point
        # Define loca bounds bound, box centered around the circle knowing the radius
        local_bounds = [(max(b[0], c - R), min(b[1], c + R)) for b, c in zip(bounds, grid_best_x)]
        # Generate a random inital start point inside this point
        x0 = random_point_within_circle(grid_best_x, R=R)
        res = minimize(lambda x: -cost_aware_EI(x, X_L, y_L, X_H, y_H, rho, theta, xi, cost_L=1, cost_H=1)[0],
                       x0=x0, bounds=local_bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x

    return min_x, grid_resolution, n_restarts, R


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



#%% Post processing functions ============================================================


def plot_n_save_BO(X_L, y_L, X_H, y_H, rho, theta, n_s_hf, n_s_lf, x_next, bounds, plt_path, data_path, xi, iteration, resolution=50):
    # Create the grid
    kp = np.linspace(bounds[0][0], bounds[0][1], resolution)
    kd = np.linspace(bounds[1][0], bounds[1][1], resolution)
    KP, KD = np.meshgrid(kp, kd)
    X_grid = np.column_stack((KP.ravel(), KD.ravel()))

	# Predict the mean and cov on the grid points setµ
    alpha, L = mf_gp_fit(X_L, y_L, X_H, y_H, rho, theta)
    mu, std = mf_gp_predict(X_grid, X_L, X_H, rho, theta, alpha, L)
    
    # Compute the best result -> from HIGH fidelity
    y_best = np.min(y_H)
    # Best point location
    best_idx = np.argmin(y_H)
    best_point = X_H[best_idx]
    
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
        ax.scatter(X_L[:, 0], X_L[:, 1], c='red', edgecolor='k', s=40, label='Low-Fidelity Training')
        ax.scatter(X_H[:n_s_hf, 0], X_H[:n_s_hf, 1], c='green', edgecolor='k', s=40, label='High-Fidelity Training')
        ax.scatter(X_H[n_s_hf:, 0], X_H[n_s_hf:, 1], c='blue', edgecolor='k', s=40, label='High-Fidelity Explored')
        ax.scatter(best_point[0], best_point[1], c='white', edgecolor='k', marker='D', s=40, label='Best High-Fidelity')
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
        'X_L': X_L,
        'X_H': X_H,
        'L' : L, 
        'alpha': alpha,
        'x_next': x_next,
        'bounds': bounds,
        'n_s_hf': n_s_hf,
        'n_s_lf': n_s_lf,
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




def plot_n_save_hpo(log_likelihood_history, param_history, iteration, PLT_PATH, DATA_PATH):
    # ------------------------------------------------------------------
    # STEP 1: Plot
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(log_likelihood_history, 'o-')
    axs[0].set_title("Log Marginal Likelihood")
    axs[0].set_xlabel("Iteration")
    axs[0].grid(True)
    
    param_arr = np.array(param_history)
    labels = ["rho", "l_lf", "var_lf", "l_delta", "var_delta", "noise_L", "noise_H"]
    for j in range(param_arr.shape[1]):
        axs[1].plot(param_arr[:, j], label=labels[j] + f' = {param_arr[-1, j]:.4f}')
    axs[1].legend()
    axs[1].set_title("Parameter History")
    axs[1].grid(True)
    
    fig.tight_layout()
    file_PATH = os.path.join(PLT_PATH, 'HPO_plots')
    os.makedirs(file_PATH, exist_ok=True)
    file_NAME = os.path.join(file_PATH, f'Iteration_{iteration+1:02d}_HPO.png')
    fig.savefig(file_NAME, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # STEP 2: Save
    # ------------------------------------------------------------------
    data_PATH = os.path.join(DATA_PATH, 'HPO_data')
    os.makedirs(data_PATH, exist_ok=True)
    np.save(os.path.join(data_PATH, f'Iteration_{iteration+1:02d}_log_likelihood.npy'), log_likelihood_history)
    np.save(os.path.join(data_PATH, f'Iteration_{iteration+1:02d}_param_history.npy'), param_arr)
