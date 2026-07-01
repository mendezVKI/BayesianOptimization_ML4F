# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 10:39:11 2026

@author: mendez
"""

#%% Initialization

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube

# Import the functions of the home-made BO
#import functions_control_Burgers as fct

#Customization of the plot 
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=12)


#%% Generate Training Data Set

# --- Function to optimize
def latent_func(X):
    """
    X: array of shape (N, 3)
    returns: array of shape (N,)
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]

    return (
        np.sin(x)
        + 0.5 * np.cos(y)
        + 0.25 * z**2
        + 0.3 * np.sin(x * y)
    )

# --- Generate a set of random input points, in 3D
np.random.seed(42)
N = 200  # number of samples
L = 1 # size of the squared domain
# Input domain
bounds = np.array([
    [-L, L],   # x1
    [-L, L],   # x2
    [-L, L],   # x3
])
# Limits of the boundaries
low = bounds[:, 0]
high = bounds[:, 1]
# Latin Hypercube sampler
sampler = LatinHypercube(d=3) # 3d dimension
# Sample in [0, 1]^d
sample = sampler.random(n=N)  # shape (N, 3)
# Affine mapping to physical domain
X = low + sample * (high - low)

# --- Generate noisy observations
sigma_noise = 0.2

y_true = latent_func(X)
noise = sigma_noise * np.random.randn(N)
y = y_true + noise

# --- Basic sanity checks
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Noise std:", np.std(y - y_true))


# --- Plot the SCALED points
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


#%% Fit the GP on the Training Set



