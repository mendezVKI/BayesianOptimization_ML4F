"""
Plotting for the mfbo Multi-Fidelity Bayesian Optimization package.

Kept separate from core.py so that the compute core (config/GP/acquisition/
BO loop) has no dependency on matplotlib. Imported lazily by
core.multi_fidelity_bayesian_optimization(), so this module is only loaded
(and matplotlib only configured) if plotting is actually used.

Only the 1D state plot is implemented (the shipped example is 1D Forrester);
plt_conv / plt_hist work in any dimension since they only plot against
iteration / cumulative cost. Extending plt_state to 2D would follow sbo's
plt_state_2D contourf pattern (three panels: mean, std, acquisition) with an
extra dimension for which fidelity plot to show.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
from typing import Callable, Dict, Any, Optional, List

import numpy as np
import os
import matplotlib.pyplot as plt

from .core import (
    Array, Bounds, MFGPModel, SaveConfig, AcqConfig, MFBOConfig, mf_gp_predict,
)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)


def setup_plotting_toggle(save_cfg: SaveConfig):
    if save_cfg.plt_all:
        save_cfg.plt_state_enabled = True
        save_cfg.plt_conv_enabled = True
        save_cfg.plt_hist_enabled = True
        save_cfg.plt_MLE_conv_enbable = True


def plt_state(
    gp: MFGPModel,
    acq_res,
    bounds: Bounds,
    it: int,
    x_best: Array,
    y_best: float,
    save_cfg: SaveConfig,
    acq_cfg: AcqConfig,
    mfbo_cfg: MFBOConfig,
    value: Callable[[Array, str], Array],
    n_plot: int = 400,
    f_low_true: Optional[Callable[[Array], Array]] = None,
    f_high_true: Optional[Callable[[Array], Array]] = None,
):
    """Dispatcher for plotting depending on input dimension."""
    if gp.X_L is None or gp.X_H is None:
        return

    d = len(bounds)
    if d == 1:
        plt_state_1D(
            gp, acq_res, bounds, it, x_best, y_best, save_cfg,
            acq_cfg, mfbo_cfg, value=value, n_plot=n_plot,
            f_low_true=f_low_true, f_high_true=f_high_true,
        )
    else:
        print(f"[plt_state] Plot not supported for dimension d={d}. Skipping.")


def plt_state_1D(
    gp: MFGPModel,
    acq_res,
    bounds: Bounds,
    it: int,
    x_best: Array,
    y_best: float,
    save_cfg: SaveConfig,
    acq_cfg: AcqConfig,
    mfbo_cfg: MFBOConfig,
    value: Callable[[Array, str], Array],
    n_plot: int = 400,
    f_low_true: Optional[Callable[[Array], Array]] = None,
    f_high_true: Optional[Callable[[Array], Array]] = None,
):
    """
    Plot current MFBO state (1D only).

    Top:
      - true low-/high-fidelity functions (if given)
      - low-/high-fidelity observations
      - GP posterior mean + 95% CI of the HIGH-fidelity (target) function

    Bottom:
      - cost-aware acquisition value for each fidelity level, with the next
        query point marked and colored by the fidelity it was assigned.
    """
    if gp.X_L is None or gp.X_H is None:
        return
    if gp.X_L.shape[1] != 1:
        return

    x_min, x_max = bounds[0]
    Xplot = np.linspace(x_min, x_max, n_plot).reshape(-1, 1)

    mu, var = mf_gp_predict(Xplot, gp, gp.X_L_norm, gp.X_H_norm, gp.alpha, gp.L, return_cov=False)
    std = np.sqrt(var)

    a_L = np.asarray(value(Xplot, "L")).reshape(-1)
    a_H = np.asarray(value(Xplot, "H")).reshape(-1)

    fig, axs = plt.subplots(
        2, 1, figsize=(6, 5.5), constrained_layout=True, sharex=True,
        gridspec_kw=dict(height_ratios=[1.4, 1]),
    )

    if f_low_true is not None:
        axs[0].plot(Xplot[:, 0], np.asarray(f_low_true(Xplot)).reshape(-1), "r--", lw=1.0, label="$f_L$ (true)")
    if f_high_true is not None:
        axs[0].plot(Xplot[:, 0], np.asarray(f_high_true(Xplot)).reshape(-1), "k--", lw=1.2, label="$f_H$ (true)")

    axs[0].plot(Xplot[:, 0], mu, "C0", lw=2, label=r"$\mu_{H}$")
    axs[0].fill_between(
        Xplot[:, 0], mu - 2 * std, mu + 2 * std,
        color="C0", alpha=0.25, label=r"$\mu_H \pm 2\sigma_H$",
    )

    axs[0].scatter(gp.X_L[:, 0], gp.y_L, c="red", edgecolor="k", s=25, zorder=10, label="Low-fidelity obs.")
    axs[0].scatter(gp.X_H[:, 0], gp.y_H, c="green", edgecolor="k", s=35, zorder=10, label="High-fidelity obs.")
    axs[0].scatter(x_best, y_best, marker="*", s=140, c="white", edgecolor="k", zorder=11, label="Best (high-fid.)")

    axs[0].set_ylabel("f(x)")
    axs[0].set_title(f"Iteration {it}/{mfbo_cfg.n_iter}")
    axs[0].legend(fontsize=7, loc='upper right', ncol=2)

    axs[1].plot(Xplot[:, 0], a_L, "r-", lw=1.5, label="acq (L)")
    axs[1].plot(Xplot[:, 0], a_H, "g-", lw=1.5, label="acq (H)")
    next_color = "red" if acq_res.level_next == "L" else "green"
    axs[1].scatter(
        acq_res.x_next, acq_res.a_best, c=next_color, edgecolor="k", s=80,
        zorder=10, label=f"Next ({acq_res.level_next})",
    )
    axs[1].legend(fontsize=8, loc='upper right')
    axs[1].set_ylabel("acq(x)")
    axs[1].set_xlabel("x")

    if save_cfg.out_path:
        fig_path = os.path.join(save_cfg.out_path, "GIF")
        os.makedirs(fig_path, exist_ok=True)
        figname = os.path.join(fig_path, f"it_{it:03d}.png")
        plt.savefig(figname, dpi=save_cfg.dpi, bbox_inches='tight')
    plt.show()
    plt.close()


def plt_conv(hist: List[Dict[str, Any]], save_cfg: SaveConfig):
    """Best (high-fidelity) value so far, against cumulative cost spent."""
    cum_cost = [h["cumulative_cost"] for h in hist]
    best_y = [h["best_y"] for h in hist]

    plt.figure(figsize=(5, 3))
    plt.plot(cum_cost, best_y, "b-o")
    plt.xlabel("Cumulative cost")
    plt.ylabel(r"min $f_H(x)$ so far")
    plt.grid(True)
    if save_cfg.out_path:
        os.makedirs(save_cfg.out_path, exist_ok=True)
        figname = os.path.join(save_cfg.out_path, "conv.png")
        plt.savefig(figname, dpi=save_cfg.dpi)
    plt.show()


def plt_hist(hist: List[Dict[str, Any]], save_cfg: SaveConfig):
    """Value observed at each call, colored by which fidelity was queried."""
    its = [h["it"] for h in hist]
    y_next = [h["y_next"] for h in hist]
    colors = ["red" if h["level_next"] == "L" else "green" for h in hist]

    plt.figure(figsize=(5, 3))
    plt.scatter(its, y_next, c=colors, edgecolor="k", s=30)
    plt.xlabel("Iteration")
    plt.ylabel("f(x) at each call")
    plt.grid(True)
    if save_cfg.out_path:
        os.makedirs(save_cfg.out_path, exist_ok=True)
        figname = os.path.join(save_cfg.out_path, "hist.png")
        plt.savefig(figname, dpi=save_cfg.dpi)
    plt.show()
