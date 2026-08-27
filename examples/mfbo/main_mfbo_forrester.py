# -*- coding: utf-8 -*-
"""
Multi-fidelity BO on the Forrester function (Forrester, Sobester & Keane,
2008) -- the standard synthetic benchmark for two-fidelity BO.

    f_high(x) = (6x-2)^2 * sin(12x-4)                       x in [0,1]
    f_low(x)  = 0.5*f_high(x) + 10*(x-0.5) - 5               (cheap, biased)

f_low is a systematically biased but strongly correlated cheap surrogate for
f_high -- exactly the setting the AR1 (Kennedy & O'Hagan) model in mfbo.core
is built for. The true high-fidelity minimum is at x* ~= 0.7572,
f_high(x*) ~= -6.0207.

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
#     python -m examples.mfbo.main_mfbo_forrester
# and directly (Spyder / VS Code "Run file"), from any working directory.
# ONLY the repository root is added -- never examples/ itself, whose sbo/,
# mfbo/ and mobo/ folders would shadow the libraries of the same name.
# Importing `examples` is what puts src/ on the path, so it has to come
# before the library import below. See examples/README.md.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples import _common  # noqa: E402
from examples.benchmarks import forrester_high, forrester_low  # noqa: E402
import mfbo  # noqa: E402

_common.use_paper_style()

# Every example writes here, whatever the working directory (gitignored).
OUT_PATH = _common.output_dir("mfbo_forrester")


func_rng = 47
rng = np.random.default_rng(func_rng)
noise_level = 0.0  # deterministic benchmark, no observation noise

#%% Visualize the two fidelities

n_real = 300
Xplot = np.linspace(0.0, 1.0, n_real).reshape(-1, 1)
y_high = forrester_high(Xplot, noise_level=0.0)
y_low = forrester_low(Xplot, noise_level=0.0)

x_min_true = Xplot[np.argmin(y_high), 0]
y_min_true = np.min(y_high)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Xplot[:, 0], y_high, "k-", lw=2, label="$f_H$ (high fidelity)")
ax.plot(Xplot[:, 0], y_low, "r--", lw=1.5, label="$f_L$ (low fidelity)")
ax.scatter(x_min_true, y_min_true, c="k", marker="*", s=120, zorder=10, label="True minimum")
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.legend()
figname = os.path.join(OUT_PATH, "true_functions.png")
plt.savefig(figname, dpi=250, bbox_inches="tight")
plt.show()

#%% Run

bounds = [(0.0, 1.0)]

f_low = lambda x: forrester_low(x, noise_level=noise_level, rng=rng)
f_high = lambda x: forrester_high(x, noise_level=noise_level, rng=rng)
f_low_true = lambda x: forrester_low(x, noise_level=0.0)
f_high_true = lambda x: forrester_high(x, noise_level=0.0)

start_time = time.time()

res = mfbo.multi_fidelity_bayesian_optimization(
    f_low=f_low,
    f_high=f_high,
    bounds=bounds,
    mfbo_cfg=mfbo.MFBOConfig(
        n_init_L=8,
        n_init_H=3,
        n_iter=20,
        init_sampling="lhs",
        random_state=1234,
    ),
    gp_cfg=mfbo.GPConfig(
        l_lf=0.2, sigma_lf=5.0,
        l_delta=0.2, sigma_delta=3.0,
        rho=1.0, sigma_L=0.05, sigma_H=0.02,
        optimize_hyperparams=True,
        hpo_every=1,
    ),
    acq_cfg=mfbo.AcqConfig(xi=0.01),
    optim_cfg=mfbo.OptimConfig(
        method="refined",
        global_method="random",
        n_raw_samples=1000,
        n_restarts=10,
    ),
    fidelity_cfg=mfbo.FidelityConfig(cost_low=1.0, cost_high=10.0),
    save_cfg=mfbo.SaveConfig(
        out_path=OUT_PATH,
        plt_all=True,
        plot_every=1,
    ),
    f_low_true=f_low_true,
    f_high_true=f_high_true,
)

run_time = time.time() - start_time

n_L = len(res.y_L)
n_H = len(res.y_H)
total_cost = res.trace.cumulative_cost[-1] if res.trace is not None and len(res.trace) > 0 else float("nan")

print("------------------------------------")
print("Home-Made Multi-Fidelity BO (AR1 / Kennedy & O'Hagan)")
print(f"  > Elapsed time = {run_time:.4f} s")
print(f"  > Evaluations: {n_L} low-fidelity, {n_H} high-fidelity, total cost = {total_cost:.1f}")
print(f"  > Best x = {res.best_x[0]:.4f} (true x* = {x_min_true:.4f})")
print(f"  > Best y = {res.best_y:.4f} (true f(x*) = {y_min_true:.4f})")
