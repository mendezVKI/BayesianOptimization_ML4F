# -*- coding: utf-8 -*-
"""
Multi-objective BO on the Schaffer N.1 function -- the standard 1D
bi-objective benchmark, and the one case where the whole MOBO state fits in
a single picture.

    f1(x) = x^2
    f2(x) = (x - 2)^2                    x in [-4, 4]

Both are minimized. They pull in opposite directions, so no single x is
best: the Pareto SET is the entire interval x in [0, 2], and the Pareto
FRONT is the convex curve it traces in the (f1, f2) plane. A good run
spreads its evaluations along that interval instead of converging to a
point, and its dominated hypervolume climbs towards the area under the
front.

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
#     python -m examples.mobo.main_mobo_schaffer
# and directly (Spyder / VS Code "Run file"), from any working directory.
# ONLY the repository root is added -- never examples/ itself -- so shared
# code is reached as `examples.benchmarks`, never as a bare `benchmarks`.
# Importing `examples` is what puts src/ on the path, so it has to come
# before the library import below. See examples/README.md.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples import _common  # noqa: E402
from examples.benchmarks import schaffer_n1  # noqa: E402
from pyRAMBO import mobo  # noqa: E402

_common.use_paper_style()

# Every example writes here, whatever the working directory (gitignored).
OUT_PATH = _common.output_dir("mobo_schaffer")


noise_level = 0.0  # deterministic benchmark, no observation noise

#%% Visualize the two objectives and the true Pareto front

n_real = 400
Xplot = np.linspace(-4.0, 4.0, n_real).reshape(-1, 1)
Yplot = schaffer_n1(Xplot, noise_level=0.0)

# The true Pareto set is x in [0, 2]
mask_true = (Xplot[:, 0] >= 0.0) & (Xplot[:, 0] <= 2.0)

fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

axs[0].plot(Xplot[:, 0], Yplot[:, 0], "C0-", lw=2, label="$f_1(x) = x^2$")
axs[0].plot(Xplot[:, 0], Yplot[:, 1], "C1-", lw=2, label="$f_2(x) = (x-2)^2$")
axs[0].axvspan(0.0, 2.0, color="k", alpha=0.12, label="True Pareto set")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$f(x)$")
axs[0].legend()

axs[1].plot(Yplot[mask_true, 0], Yplot[mask_true, 1], "k-", lw=2, label="True Pareto front")
axs[1].scatter(Yplot[~mask_true, 0], Yplot[~mask_true, 1], c="grey", s=8, alpha=0.5, label="Dominated")
axs[1].set_xlabel("$f_1$")
axs[1].set_ylabel("$f_2$")
axs[1].legend()

figname = os.path.join(OUT_PATH, "true_objectives.png")
plt.savefig(figname, dpi=250, bbox_inches="tight")
plt.show()

#%% Run

bounds = [(-4.0, 4.0)]

f = lambda x: schaffer_n1(x, noise_level=noise_level)
f_true = lambda X: schaffer_n1(X, noise_level=0.0)

start_time = time.time()

res = mobo.multi_objective_bayesian_optimization(
    f=f,
    bounds=bounds,
    mobo_cfg=mobo.MOBOConfig(
        n_init=6,
        n_iter=20,
        init_sampling="lhs",
        random_state=1234,
    ),
    gp_cfg=mobo.GPConfig(
        length_scales=0.2,
        L_B_diag=1.0,
        L_B_offdiag=0.0,
        sigma_n=0.02,
        optimize_hyperparams=True,
        hpo_every=1,
    ),
    acq_cfg=mobo.AcqConfig(
        n_mc=512,
        maximize=False,   # both objectives are minimized
        ref_margin=0.10,  # reference point placed 10% below the worst initial values
    ),
    optim_cfg=mobo.OptimConfig(
        method="refined",
        global_method="random",
        n_raw_samples=1000,
        n_restarts=10,
    ),
    save_cfg=mobo.SaveConfig(
        out_path=OUT_PATH,
        plt_all=True,
        plot_every=1,
    ),
    f_true=f_true,
)

run_time = time.time() - start_time

pareto_x = np.sort(res.pareto_X[:, 0])

print("------------------------------------")
print("Home-Made Multi-Objective BO (ICM GP + Monte-Carlo EHVI)")
print(f"  > Elapsed time = {run_time:.4f} s")
print(f"  > Evaluations: {res.X.shape[0]} ({res.pareto_X.shape[0]} of them Pareto-optimal)")
print(f"  > Reference point = {res.ref_point}")
print(f"  > Final hypervolume = {res.hypervolume:.4f}")
print(f"  > Pareto set spans x in [{pareto_x[0]:.4f}, {pareto_x[-1]:.4f}] (true: [0, 2])")
print(f"  > Points outside the true Pareto set: {np.sum((pareto_x < -1e-2) | (pareto_x > 2.01))}")
