import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1) Define a 2D "EI-like" acquisition surface
# --------------------------------------------------
def acquisition_1(x1, x2):
    peak = np.exp(-((x1 - 0.65)**2 + (x2 - 0.35)**2) / 0.015)
    ripples = 0.12 * np.sin(6 * x1) * np.cos(5 * x2)
    bowl = -0.15 * ((x1 - 0.5)**2 + (x2 - 0.5)**2)
    return peak + ripples + bowl


def acquisition_2(x1, x2):
    """
    Multi-peak EI-like surface:
    - One global maximum
    - Several strong local maxima
    """
    # Main (global) peak — narrow and strong
    peak_main = 1.2 * np.exp(
        -((x1 - 0.62)**2 + (x2 - 0.38)**2) / 0.008
    )

    # Secondary local peaks
    peak_1 = 0.9 * np.exp(
        -((x1 - 0.25)**2 + (x2 - 0.75)**2) / 0.015
    )
    peak_2 = 0.85 * np.exp(
        -((x1 - 0.80)**2 + (x2 - 0.20)**2) / 0.020
    )
    peak_3 = 0.7 * np.exp(
        -((x1 - 0.35)**2 + (x2 - 0.35)**2) / 0.010
    )

    # Oscillatory structure (EI artifacts)
    ripples = 0.15 * np.sin(7 * x1) * np.cos(6 * x2)

    # Weak global trend
    bowl = -0.10 * ((x1 - 0.5)**2 + (x2 - 0.5)**2)

    return peak_main + peak_1 + peak_2 + peak_3 + ripples + bowl


# --------------------------------------------------
# 2) Domain
# --------------------------------------------------
bounds = [(0.0, 1.0), (0.0, 1.0)]

xx, yy = np.meshgrid(
    np.linspace(*bounds[0], 400),
    np.linspace(*bounds[1], 400),
)
ZZ = acquisition_2(xx, yy)


# --------------------------------------------------
# 3) Parameters (match BO logic)
# --------------------------------------------------
rng = np.random.default_rng(0)

n_random = 625        # many random samples
grid_n = 25           # dense grid
n_starts = 12         # number of refinement starts


# --------------------------------------------------
# 4) Random candidates
# --------------------------------------------------
Xrand = rng.uniform(0, 1, size=(n_random, 2))
Arand = acquisition_2(Xrand[:, 0], Xrand[:, 1])

top_idx_rand = np.argpartition(-Arand, n_starts - 1)[:n_starts]

best_rand = Xrand[np.argmax(Arand)]


# --------------------------------------------------
# 5) Grid candidates
# --------------------------------------------------
g = np.linspace(0, 1, grid_n)
Xg1, Xg2 = np.meshgrid(g, g)
Xgrid = np.column_stack([Xg1.ravel(), Xg2.ravel()])
Agrid = acquisition_2(Xgrid[:, 0], Xgrid[:, 1])

top_idx_grid = np.argpartition(-Agrid, n_starts - 1)[:n_starts]

best_grid = Xgrid[np.argmax(Agrid)]


# --------------------------------------------------
# 6) Plot
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

titles = [
    "Random",
    "Grid",
]

for ax, title in zip(axes, titles):
    im = ax.contourf(xx, yy, ZZ, levels=40, cmap="viridis")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title, fontsize=16)

# ---- LEFT: random
h1 = axes[0].scatter(
    Xrand[:, 0],
    Xrand[:, 1],
    s=15,
    c="r",
    edgecolors="black",
    linewidths=0.4,
    alpha=0.4,
    label="Global candidates",
)

h2 = axes[0].scatter(
    Xrand[top_idx_rand, 0],
    Xrand[top_idx_rand, 1],
    s=30,
    c="r",
    edgecolors="black",
    linewidths=1.1,
    label="$n_{starts}$",
)

h3 = axes[0].scatter(
    *best_rand,
    s=150,
    c="red",
    marker="*",
    edgecolor="k",
    label="Best EI (argmax)",
)

# ---- RIGHT: grid
axes[1].scatter(
    Xgrid[:, 0],
    Xgrid[:, 1],
    s=15,
    c="r",
    edgecolors="black",
    linewidths=0.4,
    alpha=0.4
)

axes[1].scatter(
    Xgrid[top_idx_grid, 0],
    Xgrid[top_idx_grid, 1],
    s=30,
    c="r",
    edgecolors="black",
    linewidths=1.1,
)

axes[1].scatter(
    *best_grid,
    s=150,
    c="red",
    marker="*",
    edgecolor="k"
)

# ---- Shared legend (figure-level)
fig.legend(
    handles=[h1, h2, h3],
    loc="lower center",
    ncol=3,
    frameon=True,
    fontsize=13,
    bbox_to_anchor=(0.5, -0.12),
)

# ---- Colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
cbar.set_label("Acquisition function", fontsize=16)
plt.savefig("EI_opti.png", dpi=350, bbox_inches='tight')
plt.show()
