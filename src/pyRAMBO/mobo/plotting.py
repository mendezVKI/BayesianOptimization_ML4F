"""
Plotting for the mobo Multi-Objective Bayesian Optimization package.

Kept separate from core.py so that the compute core (config/GP/acquisition/
BO loop) has no dependency on matplotlib. Imported lazily by
core.multi_objective_bayesian_optimization(), so this module is only loaded
(and matplotlib only configured) if plotting is actually used.

A multi-objective run has two natural views and both are provided:

  - DESIGN space (``plt_state``): the GP posterior of every objective and
    the EHVI surface over x. Implemented for d = 1 and d = 2, following the
    same split as sbo (line plots, then contourf maps).
  - OBJECTIVE space (``plt_pareto``): where the Pareto front actually lives,
    with the attainment staircase, the reference point, and the dominated
    region whose area is the hypervolume being maximized.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
from typing import Callable, Dict, Any, Optional, List

import numpy as np
import os
import matplotlib.pyplot as plt

from .core import (
    Array, Bounds, ICMGPModel, SaveConfig, AcqConfig, MOBOConfig,
    icm_gp_predict,
)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)


def setup_plotting_toggle(save_cfg: SaveConfig):
    if save_cfg.plt_all:
        save_cfg.plt_state_enabled = True
        save_cfg.plt_pareto_enabled = True
        save_cfg.plt_conv_enabled = True
        save_cfg.plt_hist_enabled = True


def _save_frame(save_cfg: SaveConfig, subdir: str, it: int) -> None:
    if not save_cfg.out_path:
        return
    fig_path = os.path.join(save_cfg.out_path, "plots", subdir)
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, f"it_{it:03d}.png"), dpi=save_cfg.dpi, bbox_inches='tight')


def plt_state(
    gp: ICMGPModel,
    acq_res,
    bounds: Bounds,
    it: int,
    pareto_X: Array,
    pareto_Y: Array,
    save_cfg: SaveConfig,
    acq_cfg: AcqConfig,
    mobo_cfg: MOBOConfig,
    acq: Callable[[Array], Array],
    n_plot: int = 400,
    f_true: Optional[Callable[[Array], Array]] = None,
    y_new: Optional[Array] = None,
):
    """Dispatcher for the design-space plot, depending on input dimension.

    ``f_true``, when given, must accept an (n, d) array and return the (n, M)
    true objective values, so the posterior can be compared against ground
    truth on a synthetic benchmark.

    gp / acq_res / acq describe the state the acquisition was computed from
    (pre-update). ``y_new`` (the (M,) objective vector obtained at
    acq_res.x_next) is an optional 1D overlay; pareto_X / pareto_Y may be the
    post-update Pareto set. The 2D plot already marks x_next."""

    if gp.X is None or gp.Y is None:
        return

    d = len(bounds)
    if d == 1:
        plt_state_1D(
            gp, acq_res, bounds, it, pareto_X, pareto_Y, save_cfg,
            acq_cfg, mobo_cfg, acq=acq, n_plot=n_plot, f_true=f_true, y_new=y_new,
        )
    elif d == 2:
        plt_state_2D(
            gp, acq_res, bounds, it, pareto_X, pareto_Y, save_cfg,
            acq_cfg, mobo_cfg, acq=acq, n_plot=min(n_plot, 60),
        )
    else:
        print(f"[plt_state] Plot not supported for dimension d={d}. Skipping.")


def plt_state_1D(
    gp: ICMGPModel,
    acq_res,
    bounds: Bounds,
    it: int,
    pareto_X: Array,
    pareto_Y: Array,
    save_cfg: SaveConfig,
    acq_cfg: AcqConfig,
    mobo_cfg: MOBOConfig,
    acq: Callable[[Array], Array],
    n_plot: int = 400,
    f_true: Optional[Callable[[Array], Array]] = None,
    y_new: Optional[Array] = None,
):
    """
    Plot the current MOBO state in design space (1D only).

    One panel per objective (posterior mean, 95% band, observations, and the
    Pareto-optimal designs highlighted), then one panel with the EHVI and
    the next query point.
    """

    if gp.X is None or gp.Y is None or gp.X.shape[1] != 1:
        return

    M = gp.n_obj
    x_min, x_max = bounds[0]
    Xplot = np.linspace(x_min, x_max, n_plot).reshape(-1, 1)

    mu, var = icm_gp_predict(Xplot, gp, return_cov=False)
    std = np.sqrt(np.maximum(var, 0.0))
    a = np.asarray(acq(Xplot)).reshape(-1)

    Y_true = None
    if f_true is not None:
        Y_true = np.atleast_2d(np.asarray(f_true(Xplot), dtype=float))

    fig, axs = plt.subplots(
        M + 1, 1, figsize=(6, 2.2 * (M + 1)), constrained_layout=True, sharex=True,
    )

    for m in range(M):
        ax = axs[m]
        if Y_true is not None:
            ax.plot(Xplot[:, 0], Y_true[:, m], "k--", lw=1.2, label=f"$f_{m + 1}$ (true)")

        ax.plot(Xplot[:, 0], mu[:, m], "C0", lw=2, label=rf"$\mu_{m + 1}$")
        ax.fill_between(
            Xplot[:, 0], mu[:, m] - 2 * std[:, m], mu[:, m] + 2 * std[:, m],
            color="C0", alpha=0.25, label=rf"$\mu_{m + 1} \pm 2\sigma_{m + 1}$",
        )

        ax.scatter(gp.X[:, 0], gp.Y[:, m], c="grey", edgecolor="k", s=25, zorder=10, label="Observations")
        if y_new is not None:
            ax.scatter(
                np.asarray(acq_res.x_next).reshape(-1), [float(np.asarray(y_new).reshape(-1)[m])],
                c="red", edgecolor="k", s=45, zorder=12, label="Evaluated this iter",
            )
        if pareto_X.shape[0] > 0:
            ax.scatter(
                pareto_X[:, 0], pareto_Y[:, m], marker="*", s=120, c="white",
                edgecolor="k", zorder=11, label="Pareto set",
            )

        ax.set_ylabel(f"$f_{m + 1}(x)$")
        if m == 0:
            ax.set_title(f"Iteration {it}/{mobo_cfg.n_iter}")
            ax.legend(fontsize=7, loc='upper right', ncol=2)

    axs[-1].plot(Xplot[:, 0], a, "C3-", lw=1.5, label="EHVI")
    axs[-1].scatter(
        acq_res.x_next, acq_res.a_best, c="C3", edgecolor="k", s=80, zorder=10, label="Next",
    )
    axs[-1].legend(fontsize=8, loc='upper right')
    axs[-1].set_ylabel("EHVI$(x)$")
    axs[-1].set_xlabel("$x$")

    _save_frame(save_cfg, "GIF", it)
    plt.show()
    plt.close()


def plt_state_2D(
    gp: ICMGPModel,
    acq_res,
    bounds: Bounds,
    it: int,
    pareto_X: Array,
    pareto_Y: Array,
    save_cfg: SaveConfig,
    acq_cfg: AcqConfig,
    mobo_cfg: MOBOConfig,
    acq: Callable[[Array], Array],
    n_plot: int = 60,
):
    """
    Plot the current MOBO state in design space (2D only).

    One contourf panel per objective posterior mean, then one for the EHVI,
    all overlaid with the observations, the Pareto set and the next query.
    """

    if gp.X is None or gp.Y is None or gp.X.shape[1] != 2:
        return

    M = gp.n_obj

    x1 = np.linspace(bounds[0][0], bounds[0][1], n_plot)
    x2 = np.linspace(bounds[1][0], bounds[1][1], n_plot)
    X1, X2 = np.meshgrid(x1, x2)
    Xplot = np.column_stack([X1.ravel(), X2.ravel()])

    mu, _ = icm_gp_predict(Xplot, gp, return_cov=False)
    a = np.asarray(acq(Xplot)).reshape(n_plot, n_plot)

    fig, axs = plt.subplots(1, M + 1, figsize=(4.2 * (M + 1), 3.6), constrained_layout=True)

    fields = [mu[:, m].reshape(n_plot, n_plot) for m in range(M)]
    titles = [rf"$\mu_{m + 1}(x)$" for m in range(M)]
    fields.append(a)
    titles.append("EHVI$(x)$")

    x_next = np.asarray(acq_res.x_next).reshape(-1)

    for ax, field, title in zip(axs, fields, titles):
        cs = ax.contourf(X1, X2, field, levels=40, cmap="viridis")
        fig.colorbar(cs, ax=ax)

        ax.scatter(gp.X[:, 0], gp.X[:, 1], c="grey", edgecolor="k", s=18, zorder=10, label="Observations")
        if pareto_X.shape[0] > 0:
            ax.scatter(
                pareto_X[:, 0], pareto_X[:, 1], marker="*", s=90, c="white",
                edgecolor="k", zorder=11, label="Pareto set",
            )
        ax.scatter(x_next[0], x_next[1], marker="X", s=80, c="red", edgecolor="k", zorder=12, label="Next")

        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    axs[0].legend(fontsize=7, loc='upper right')
    fig.suptitle(f"Iteration {it}/{mobo_cfg.n_iter}")

    _save_frame(save_cfg, "GIF", it)
    plt.show()
    plt.close()


def plt_pareto(
    Y: Array,
    pareto_Y: Array,
    y_next: Array,
    ref_point: Array,
    it: int,
    hv: float,
    save_cfg: SaveConfig,
    mobo_cfg: MOBOConfig,
):
    """
    Plot the current state in OBJECTIVE space (2 objectives).

    All observations, the current Pareto front drawn as the attainment
    staircase, the reference point, and the point that was just evaluated.
    The staircase is exactly the boundary of the region whose area is the
    hypervolume reported in the title, so the picture and the number in the
    trace always tell the same story.
    """

    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if Y.shape[1] != 2:
        print(f"[plt_pareto] Objective-space plot needs 2 objectives, got {Y.shape[1]}. Skipping.")
        return

    ref_point = np.asarray(ref_point, dtype=float).reshape(-1)
    y_next = np.asarray(y_next, dtype=float).reshape(-1)

    plt.figure(figsize=(5, 4))
    plt.scatter(Y[:, 0], Y[:, 1], c="grey", edgecolor="k", s=25, zorder=5, label="Observations")

    if pareto_Y.shape[0] > 0:
        # Order the front the way it is actually attained: sorted along the
        # first objective, with a step between consecutive points.
        P = np.asarray(pareto_Y, dtype=float)
        P = P[np.argsort(P[:, 0])]
        plt.step(P[:, 0], P[:, 1], where="post", color="C0", lw=1.5, zorder=6)
        plt.scatter(
            P[:, 0], P[:, 1], marker="*", s=140, c="white", edgecolor="C0",
            linewidth=1.5, zorder=7, label="Pareto front",
        )

    plt.scatter(y_next[0], y_next[1], marker="X", s=90, c="red", edgecolor="k", zorder=8, label="Just evaluated")
    plt.scatter(
        ref_point[0], ref_point[1], marker="s", s=50, c="k", zorder=8, label="Reference point",
    )

    plt.xlabel("$f_1$")
    plt.ylabel("$f_2$")
    plt.title(f"Iteration {it}/{mobo_cfg.n_iter} -- hypervolume = {hv:.4g}")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    _save_frame(save_cfg, "GIF_pareto", it)
    plt.show()
    plt.close()


def plt_conv(hist: List[Dict[str, Any]], save_cfg: SaveConfig):
    """Dominated hypervolume against iteration -- the multi-objective
    counterpart of sbo's best-so-far convergence curve. It is monotonically
    non-decreasing by construction, since the reference point is fixed."""

    its = [h["it"] for h in hist]
    hv = [h["hypervolume"] for h in hist]

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(its, hv, "b-o", label="Hypervolume")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Dominated hypervolume")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(its, [h["n_pareto"] for h in hist], "r--s", ms=3, label="Front size")
    ax2.set_ylabel("Pareto front size", color="r")
    ax2.tick_params(axis="y", colors="r")

    fig.tight_layout()
    if save_cfg.out_path:
        plots_path = os.path.join(save_cfg.out_path, "plots")
        os.makedirs(plots_path, exist_ok=True)
        plt.savefig(os.path.join(plots_path, "conv.png"), dpi=save_cfg.dpi)
    plt.show()


def plt_hist(hist: List[Dict[str, Any]], save_cfg: SaveConfig):
    """Every objective value observed at each call, one series per objective."""

    its = [h["it"] for h in hist]
    Y = np.asarray([h["y_next"] for h in hist], dtype=float)

    fig, axs = plt.subplots(Y.shape[1], 1, figsize=(5, 2.2 * Y.shape[1]), sharex=True, constrained_layout=True)
    axs = np.atleast_1d(axs)

    for m, ax in enumerate(axs):
        ax.scatter(its, Y[:, m], c=f"C{m}", edgecolor="k", s=30)
        ax.set_ylabel(f"$f_{m + 1}$ at each call")
        ax.grid(True)

    axs[-1].set_xlabel("Iteration")

    if save_cfg.out_path:
        plots_path = os.path.join(save_cfg.out_path, "plots")
        os.makedirs(plots_path, exist_ok=True)
        plt.savefig(os.path.join(plots_path, "hist.png"), dpi=save_cfg.dpi)
    plt.show()
