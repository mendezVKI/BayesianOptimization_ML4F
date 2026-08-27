# -*- coding: utf-8 -*-
"""
Multi-objective BO on the Binh & Korn function -- the standard 2D
bi-objective benchmark.

    f1(x) = 4*x1^2 + 4*x2^2
    f2(x) = (x1 - 5)^2 + (x2 - 5)^2      x1 in [0,5], x2 in [0,3]

Both are minimized. f1 pulls the design towards the origin and f2 towards
(5,5), so the observed objective values are strongly anti-correlated
(corr ~ -0.88 over the box) and the Pareto set is the segment between the
two attractors.

The run also prints two correlations that are easy to conflate, and the gap
between them is the point:

  - the EMPIRICAL correlation of the observed f1, f2;
  - the LATENT correlation carried by the coregionalization matrix B.

B does not measure the first one. Under the ICM model both outputs are
already smooth functions of the SAME x, and that shared input alone induces
the observed anti-correlation. B_12 only captures whatever correlation is
left on top of that. Binh & Korn is deterministic and each objective is an
elementary quadratic, so the GP models each one almost exactly on its own
and essentially nothing is left over: the fitted B_12 comes out near zero
even though the data is strongly anti-correlated. That is the correct answer
for this problem, not a failure to learn.

A near-zero B_12 does not make the joint model pointless here -- the shared
ARD kernel, the common design, and above all the joint EHVI (which needs the
full 2x2 posterior covariance at each candidate) are what the single ICM GP
buys. B_12 becomes informative instead when the objectives share structure
the inputs do not explain, e.g. correlated measurement noise or a common
unmodelled latent effect.

@author: Yannick Lecomte
"""

#%% Initialization

import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Make the `examples` package importable, so this file runs both as
#     python -m examples.mobo.main_mobo_binh_korn
# and directly (Spyder / VS Code "Run file"), from any working directory.
# ONLY the repository root is added -- never examples/ itself, whose sbo/,
# mfbo/ and mobo/ folders would shadow the libraries of the same name.
# Importing `examples` is what puts src/ on the path, so it has to come
# before the library import below. See examples/README.md.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples import _common  # noqa: E402
from examples.benchmarks import binh_korn  # noqa: E402
import mobo  # noqa: E402

_common.use_paper_style()

# Every example writes here, whatever the working directory (gitignored).
OUT_PATH = _common.output_dir("mobo_binh_korn")


noise_level = 0.0  # deterministic benchmark, no observation noise

bounds = [(0.0, 5.0), (0.0, 3.0)]

#%% Visualize the two objectives over the design space

n_grid = 120
x1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
x2 = np.linspace(bounds[1][0], bounds[1][1], n_grid)
X1, X2 = np.meshgrid(x1, x2)
Xgrid = np.column_stack([X1.ravel(), X2.ravel()])
Ygrid = binh_korn(Xgrid, noise_level=0.0)

# Reference front, obtained by brute force on the dense grid
mask_true = mobo.nondominated_mask(-Ygrid)
front_true = Ygrid[mask_true]
front_true = front_true[np.argsort(front_true[:, 0])]

fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)

for m, ax in enumerate(axs[:2]):
    cs = ax.contourf(X1, X2, Ygrid[:, m].reshape(n_grid, n_grid), levels=40, cmap="viridis")
    fig.colorbar(cs, ax=ax)
    ax.scatter(Xgrid[mask_true, 0], Xgrid[mask_true, 1], c="red", s=3, label="True Pareto set")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"$f_{m + 1}(x)$")
axs[0].legend(fontsize=8)

axs[2].scatter(Ygrid[:, 0], Ygrid[:, 1], c="grey", s=3, alpha=0.4, label="Dominated")
axs[2].plot(front_true[:, 0], front_true[:, 1], "k-", lw=2, label="True Pareto front")
axs[2].set_xlabel("$f_1$")
axs[2].set_ylabel("$f_2$")
axs[2].legend(fontsize=8)

figname = os.path.join(OUT_PATH, "true_objectives.png")
plt.savefig(figname, dpi=250, bbox_inches="tight")
plt.show()

#%% Run

f = lambda x: binh_korn(x, noise_level=noise_level)

start_time = time.time()

res = mobo.multi_objective_bayesian_optimization(
    f=f,
    bounds=bounds,
    mobo_cfg=mobo.MOBOConfig(
        n_init=10,
        n_iter=25,
        init_sampling="lhs",
        random_state=7,
    ),
    gp_cfg=mobo.GPConfig(
        length_scales=0.3,
        L_B_diag=1.0,
        L_B_offdiag=0.0,
        sigma_n=0.02,
        optimize_hyperparams=True,
        hpo_every=1,
    ),
    acq_cfg=mobo.AcqConfig(
        n_mc=512,
        maximize=False,   # both objectives are minimized
        ref_margin=0.10,
    ),
    optim_cfg=mobo.OptimConfig(
        method="refined",
        global_method="random",
        n_raw_samples=1500,
        n_restarts=10,
    ),
    save_cfg=mobo.SaveConfig(
        out_path=OUT_PATH,
        plt_all=True,
        plot_every=5,
    ),
)

run_time = time.time() - start_time

#%% Final front against the brute-force reference

P = res.pareto_Y[np.argsort(res.pareto_Y[:, 0])]

plt.figure(figsize=(5, 4))
plt.plot(front_true[:, 0], front_true[:, 1], "k-", lw=2, label="True Pareto front")
plt.scatter(res.Y[:, 0], res.Y[:, 1], c="grey", edgecolor="k", s=25, label="Observations")
plt.plot(P[:, 0], P[:, 1], "C0*--", ms=12, label="MOBO Pareto front")
plt.xlabel("$f_1$")
plt.ylabel("$f_2$")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
figname = os.path.join(OUT_PATH, "final_front.png")
plt.savefig(figname, dpi=250, bbox_inches="tight")
plt.show()

B = res.gp.B
corr_latent = B[0, 1] / np.sqrt(B[0, 0] * B[1, 1])
corr_observed = np.corrcoef(res.Y.T)[0, 1]

print("------------------------------------")
print("Home-Made Multi-Objective BO (ICM GP + Monte-Carlo EHVI)")
print(f"  > Elapsed time = {run_time:.4f} s")
print(f"  > Evaluations: {res.X.shape[0]} ({res.pareto_X.shape[0]} of them Pareto-optimal)")
print(f"  > Reference point = {res.ref_point}")
print(f"  > Final hypervolume = {res.hypervolume:.4f}")
print(f"  > Observed corr(f1, f2)                      = {corr_observed:+.4f}")
print(f"  > Latent corr B_12 / sqrt(B_11 B_22)         = {corr_latent:+.4f}")
print("    (near zero, as expected: the shared input x already explains the")
print("     observed anti-correlation -- see the module docstring)")
