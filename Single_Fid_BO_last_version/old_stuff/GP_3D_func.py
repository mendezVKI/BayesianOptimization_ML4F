# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 10:30:52 2026

@author: lyann
"""

import numpy as np
import matplotlib.pyplot as plt
# Import the functions of the home-made BO
import functions_control_Burgers as fct
from scipy.stats.qmc import LatinHypercube



# ==============================
# Random seed for reproducibility
# ==============================
np.random.seed(42)

# ==============================
# Define the latent 3D function
# ==============================
def latent_function(X):
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

# ==============================
# Generate input samples
# ==============================
N = 200  # number of samples

L = 1

# Input domain
bounds = np.array([
    [-L, L],   # x1
    [-L, L],   # x2
    [-L, L],   # x3
])

X = np.random.uniform(
    low=bounds[:, 0],
    high=bounds[:, 1],
    size=(N, 3)
)

# Latin Hypercube sampler
sampler = LatinHypercube(d=3) # 3d dimension
# Sample in [0, 1]^d
sample = sampler.random(n=N)  # shape (N, 3)
X = bounds * sample      # W_init -> X input in the GPr notation


# ==============================
# Generate noisy observations
# ==============================
sigma_noise = 0.2

y_true = latent_function(X)
noise = sigma_noise * np.random.randn(N)
y = y_true + noise

# ==============================
# Basic sanity checks
# ==============================
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Noise std:", np.std(y - y_true))

# ==============================
# Optional visualization (pairwise projections)
# ==============================
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

axs[0].scatter(X[:, 0], y, c='k', s=15)
axs[0].set_xlabel("x1")
axs[0].set_ylabel("y")

axs[1].scatter(X[:, 1], y, c='k', s=15)
axs[1].set_xlabel("x2")

axs[2].scatter(X[:, 2], y, c='k', s=15)
axs[2].set_xlabel("x3")

plt.tight_layout()
plt.show()
