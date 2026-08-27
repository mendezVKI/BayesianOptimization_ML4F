# -*- coding: utf-8 -*-
"""
Demonstrates the two-tier persistence layer of bo_ml4f.

Part 1 -- "quick run": default SaveConfig(). Tier 1 (the lightweight
per-iteration trace) is always on and stays in memory / gets written as a
tiny trace.npz; no GP snapshots are ever built.

Part 2 -- "full run": Tier 2 enabled. Every iteration's full GP state
(design set, hyperparameters, kernel id, normalization) is flushed to disk
as it goes. We then reload the run from disk ALONE (no reference to the live
`res` object) and re-plot the posterior at a chosen iteration, to show that
Tier 2 is enough to reconstruct predictions without re-running the objective.

@author: Yannick Lecomte
"""

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

import bo_ml4f as bo
from bo_ml4f import persistence
from benchmarks import quadratic_1d

OUT_PATH = "./persistence_demo_out"
if os.path.exists(OUT_PATH):
    shutil.rmtree(OUT_PATH)
os.makedirs(OUT_PATH)

bounds = [(-2.0, 2.0)]
n_init, n_iter = 4, 12

#%% Part 1 -- quick run: Tier 1 only (default SaveConfig)

quick_res = bo.bayesian_optimization(
    f=quadratic_1d,
    bounds=bounds,
    bo_cfg=bo.BOConfig(n_init=n_init, n_iter=n_iter, random_state=0),
    gp_cfg=bo.GPConfig(),
    acq_cfg=bo.AcqConfig(),
    optim_cfg=bo.OptimConfig(n_raw_samples=300),
    save_cfg=bo.SaveConfig(),  # no out_path -> nothing touches disk
)

print("Quick run (Tier 1 only)")
print(f"  best_y = {quick_res.best_y:.4f}")
print(f"  trace length = {len(quick_res.trace)} (fields: it, x_next, y_next, x_best, y_best, wall_time, acq_value)")
print(f"  Tier 2 snapshots allocated? {quick_res.snapshots is not None}")  # False: never built

#%% Part 2 -- full run: Tier 2 enabled, flushed to disk as it goes

full_out = os.path.join(OUT_PATH, "full_run")

full_res = bo.bayesian_optimization(
    f=quadratic_1d,
    bounds=bounds,
    bo_cfg=bo.BOConfig(n_init=n_init, n_iter=n_iter, random_state=0),
    gp_cfg=bo.GPConfig(),
    acq_cfg=bo.AcqConfig(),
    optim_cfg=bo.OptimConfig(n_raw_samples=300),
    save_cfg=bo.SaveConfig(
        out_path=full_out,
        create_timestamp=False,
        log_enabled=False,
        snapshot_enabled=True,
        snapshot_every=1,
        snapshot_flush_every=5,  # write to disk and drop from memory every 5 snapshots
    ),
)

print("\nFull run (Tier 2 enabled)")
print(f"  best_y = {full_res.best_y:.4f}")
print(f"  snapshots left in memory (unflushed tail) = {len(full_res.snapshots)}")
print(f"  files on disk = {sorted(os.listdir(os.path.join(full_out, 'snapshots')))}")

#%% Reload from disk only, and re-plot the posterior at a past iteration

run = persistence.load_run(full_out)
print(f"\nReloaded run: available Tier-2 iterations = {run.available_iterations}")

it_to_plot = run.available_iterations[-1]
posterior = run.posterior(it_to_plot)

Xgrid = np.linspace(bounds[0][0], bounds[0][1], 400).reshape(-1, 1)
mu, var = posterior.predict(Xgrid)
std = np.sqrt(var)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Xgrid[:, 0], mu, "C0", lw=2, label=r"$\mu_{GP}$ (reconstructed)")
ax.fill_between(Xgrid[:, 0], mu - 2 * std, mu + 2 * std, color="C0", alpha=0.25, label=r"$\pm 2\sigma$")
ax.scatter(posterior.gp.Xs[:, 0], posterior.gp.ys, c="k", s=20, zorder=10, label="Observations")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title(f"Posterior reconstructed from disk (iteration {it_to_plot})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_PATH, "reconstructed_posterior.png"), dpi=200)
plt.show()

# Sanity check: the reconstruction must match the live model bit-for-bit.
live_gp = full_res.states[it_to_plot].gp
mu_live, var_live = bo.gp_predict(
    Xgrid, live_gp.Xs, live_gp.alpha, live_gp.L,
    l_c=live_gp.l_c, sigma_f=live_gp.sigma_f, return_cov=False, gp=live_gp,
)
print(f"  max |mu_reconstructed - mu_live| = {np.max(np.abs(mu - mu_live)):.3e}")
print(f"  max |var_reconstructed - var_live| = {np.max(np.abs(var - var_live)):.3e}")
