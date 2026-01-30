# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:28:02 2026

@author: mendez, lecomte
"""

#%% Initialization

# IMport main packages
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio

# Import the home-made BO lib
import BO_ML4F as bo

#Customization of the plot 
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=12)

# Define and create (if needed) the saving path
OUT_PATH = "./BO_lib_res_3D"
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
    
# Define and create (if needed) the gif path
GIF_PATH = os.path.join(OUT_PATH, "GIFs")
if not os.path.exists(GIF_PATH):
    os.makedirs(GIF_PATH)
    
#%% STEP 1 - Define a 3D noisy function

np.random.seed(37)


def func(x, noise_level=0.05):
    x = np.asarray(x)

    # nonlinear coordinate warping
    u = x1 + 0.6 * np.sin(1.5 * x2)
    v = x2 + 0.4 * np.sin(1.2 * x3)

    f = (
        (x3 - np.tanh(u))**2
        + 0.5 * (v - 0.3 * u**2)**2
    )

    noise = noise_level * np.random.randn()
    return f + noise

import numpy as np




#%% STEP 2 - Generate and visualize a random set of initial point

n_plt = 50 # Number points of the plot 
LL = 2 # Size of the domain
bounds = [(-LL, LL), (-LL, LL), (-LL, LL)]


# Extract the BOConfig container from the lib
bo_cfg = bo.BOConfig(
    n_init=n_plt, 
    random_state=1234
    )
rng = bo._rng(bo_cfg.random_state)

# Get the inital random set of points
X_train, y_train = bo.init_dataset(
    f=func,
    bounds=bounds,
    bo_cfg=bo_cfg,
    rng=rng
)
    
# Plot the random points
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    X_train[:, 0],
    X_train[:, 1],
    X_train[:, 2],
    c=y_train,
    cmap="viridis",
    s=30,
    alpha=0.8, 

)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$x_3$")
ax.set_title("Initial training set of $n_{init}$"+f"={n_plt}", fontsize=16)
# Colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.1)
cbar.set_label(r"$y$")
figname = os.path.join(OUT_PATH, f"training_n{n_plt}.png")
plt.savefig(figname, dpi=300, bbox_inches='tight')
plt.show()

#%% STEP 3 - Generate and visualize an noissless grid to found the grund true

n_per_dim=30 # number of sample per dimension of the input space
n_grid = n_per_dim ** 3 # total number of points
x1 = np.linspace(-LL, LL, n_per_dim) # stractured list for each dimension
x2 = np.linspace(-LL, LL, n_per_dim)
x3 = np.linspace(-LL, LL, n_per_dim)

X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing="ij")
# Generate the input matrix of shape (N, d)
X_grid = np.column_stack([
    X1.ravel(),
    X2.ravel(),
    X3.ravel(),
])

# Evalute each point
y_grid = np.asarray([func(X_grid[ii, :], noise_level=0) for ii in range(n_grid)], dtype=float)

# Find teh minimun value of position
min_loc_true = np.argmin(y_grid)
x_min_true = X_grid[min_loc_true]
y_min_true = y_grid[min_loc_true]

# Plot the random points
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    X_grid[:, 0],
    X_grid[:, 1],
    X_grid[:, 2],
    c=y_grid,                 # color by output y
    cmap='viridis',      
    s=30,
    alpha=0.5
)
ax.scatter(x_min_true[0], x_min_true[1], x_min_true[2], s=100, c='r')
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$x_3$")
ax.set_title("3D input space colored by output $y$ \n" 
             f"- n={n_grid} gridded samples with Latin Hypercube -")
# Colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.1)
cbar.set_label(r"$y$")
figname = os.path.join(OUT_PATH, f"grid_n{n_grid}.png")
plt.savefig(figname, dpi=300, bbox_inches='tight')
plt.show()

#%% STEP 4 - Run Bayesian Optimization

n_init = 50
n_iter = 200 

res = bo.bayesian_optimization(
    f=func,
    bounds=bounds,
    bo_cfg=bo.BOConfig(
        n_init=n_init,
        n_iter=n_iter,
        random_state=1234
    ),
    gp_cfg=bo.GPConfig(),
    acq_cfg=bo.AcqConfig(kind="EI"),
    optim_cfg=bo.OptimConfig(method="refined"),
    save_cfg=bo.SaveConfig(
        plt_conv_enabled=True,
        plt_hist_enabled=True,
        save_path=OUT_PATH
    ),
)


print(f" ---- Comparison for n_init={n_init}, n_iter={n_iter} ---- ")
print("True function without noise")
print("  > Best x:", x_min_true)
print("  > Best y:", y_min_true)


print("BO")
print("  > Best x:", res.best_x)
print("  > Best y:", res.best_y)

#%% STEP 5 - Plotting GIFs for The 3D map


x_next_arr = []
y_next_arr = []

maps_path = os.path.join(GIF_PATH, "3D_plt")
os.makedirs(maps_path, exist_ok=True)

for ii, hist_iter in enumerate(res.history):

    x_next_arr.append(hist_iter["x_next"])
    y_next_arr.append(hist_iter["y_next"])

    # Convert to arrays ONCE per frame
    X = np.asarray(x_next_arr)
    Y = np.asarray(y_next_arr)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        X_train[:, 2],
        c=y_train,
        cmap="viridis",
        s=30,
        alpha=0.8, 

    )

    sc = ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=Y,
        cmap="viridis",
        s=30,
        alpha=0.8,
        edgecolor="r", 
        linewidths = 0.6
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$x_3$")
    ax.set_title(f"Exploration after $n_{{iter}}$ = {ii+1}", fontsize=16)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.1)
    cbar.set_label(r"$y$")

    figname = os.path.join(maps_path, f"it{ii:03d}.png")
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.close(fig)


maps_gif_path = os.path.join(GIF_PATH, "BO_3D_exploration.gif")

frames = []

png_files = sorted(
    f for f in os.listdir(maps_path) if f.endswith(".png")
)

for fname in png_files:
    frame = imageio.imread(os.path.join(maps_path, fname))
    frames.append(frame)

imageio.mimsave(
    maps_gif_path,
    frames,
    duration=0.8,   # seconds per frame (adjust)
    loop=0          # infinite loop
)


#%% STEP 6 - Plotting GIFs for the convergence

conv_list = []
hist_list = []

conv_path = os.path.join(GIF_PATH, "conv_plt")
os.makedirs(conv_path, exist_ok=True)

# Precompute y-limits ONCE (important for stable GIF)
best_y_min = min(h["best_y"] for h in res.history)
best_y_max = max(h["best_y"] for h in res.history)

y_next_min = min(h["y_next"] for h in res.history)
y_next_max = max(h["y_next"] for h in res.history)

# Optional padding (recommended for aesthetics)
pad_best = 0.05 * (best_y_max - best_y_min + 1e-12)
pad_next = 0.05 * (y_next_max - y_next_min + 1e-12)


for ii, hist_iter in enumerate(res.history):

    conv_list.append(hist_iter["best_y"])
    hist_list.append(hist_iter["y_next"])

    conv_array = np.asarray(conv_list)
    hist_array = np.asarray(hist_list)

    x_conv = np.arange(1, len(conv_array) + 1)

    # ---- Create figure
    fig, axs = plt.subplots(
        2, 1,
        figsize=(5, 4),
        sharex=True,
        constrained_layout=True
    )
    # Convergence
    axs[0].plot(
        x_conv,
        conv_array,
        "b-o",
        clip_on=False,
        zorder=2
    )

    axs[0].set_ylabel(r"$\min f(x)$")
    axs[0].grid(True)
    axs[0].set_xlim(1, n_iter+1)
    axs[0].set_ylim(best_y_min - pad_best, best_y_max + pad_best)

    # Highlight last point (above frame)
    axs[0].scatter(
        x_conv[-1],
        conv_array[-1],
        c="red",
        s=80,
        zorder=10,
        clip_on=False
    )

    # Explorations
    axs[1].plot(
        x_conv,
        hist_array,
        "k.--",
        alpha=0.7
    )

    axs[1].set_xlabel("Calls $n$")
    axs[1].set_ylabel(r"$f(x_n)$")
    axs[1].grid(True)
    axs[1].set_ylim(y_next_min - pad_next, y_next_max + pad_next)

    # ---- Save frame
    figname = os.path.join(conv_path, f"it{ii:03d}.png")
    fig.savefig(figname, dpi=250, bbox_inches="tight")
    plt.close(fig)


conv_gif_path = os.path.join(GIF_PATH, "conv.gif")

frames = []

png_files = sorted(
    f for f in os.listdir(conv_path) if f.endswith(".png")
)

for fname in png_files:
    frame = imageio.imread(os.path.join(conv_path, fname))
    frames.append(frame)

imageio.mimsave(
    conv_gif_path,
    frames,
    duration=0.8,   # seconds per frame (adjust)
    loop=0          # infinite loop
)

shapes = {}
for fname in png_files:
    frame = imageio.imread(os.path.join(conv_path, fname))
    shapes[fname] = frame.shape

