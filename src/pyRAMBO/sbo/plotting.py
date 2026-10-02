"""
Plotting for the sbo Bayesian Optimization package.

Kept separate from core.py so that the compute core (config/GP/acquisition/
BO loop) has no dependency on matplotlib. Imported lazily by
core.bayesian_optimization(), so this module is only loaded (and matplotlib
only configured) if plotting is actually used.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
from typing import Callable, Dict, Any, Optional, List

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from .core import (
    Array, Bounds, GPModel, SaveConfig, AcqConfig, BOConfig, gp_predict,
)

#Customization of the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=12)


def setup_plotting_toggle(save_cfg:SaveConfig):

    if save_cfg.plt_all:
        save_cfg.plt_state_enabled     = True
        save_cfg.plt_conv_enabled      = True
        save_cfg.plt_hist_enabled      = True
        save_cfg.plt_MLE_conv_enbable  = True


def _finish_figure(save_cfg: SaveConfig) -> None:
    """Show the current figure only if save_cfg.show_plots, then close it."""
    if save_cfg.show_plots:
        plt.show()
    plt.close()


def plt_state(
    gp: GPModel,
    acq_res,
    f: Callable[[Array], Array],
    bounds: Bounds,
    it: int,
    x_best: Array,
    y_best: float,
    save_cfg: SaveConfig,
    acq_cfg: AcqConfig,
    bo_cfg: BOConfig,
    acq: Callable[[Array], Array],
    n_plot: int = 400,
    f_true: Optional[Callable[[Array], Array]] = None,
    x_new: Optional[Array] = None,
    y_new: Optional[Array] = None,
    x_refined: Optional[Array] = None,
    y_refined: Optional[Array] = None,
):
    """
    Dispatcher for plotting depending on input dimension.

    gp / acq_res / acq describe the state the acquisition was computed from
    (pre-update). x_new, y_new (every row added to the dataset this
    iteration) and x_refined (ADAM endpoints, NaN rows if not refined) are
    optional overlays of what was then evaluated; x_best / y_best may be the
    post-update best.
    """

    # Safety checks
    if gp.Xs is None or gp.ys is None:
        return

    d = gp.Xs.shape[1]

    if d == 1:
        plt_state_1D(
            gp, acq_res, f, bounds, it,
            x_best, y_best, save_cfg,
            acq_cfg, bo_cfg, acq=acq,
            n_plot=n_plot, f_true=f_true,
            x_new=x_new, y_new=y_new, x_refined=x_refined, y_refined=y_refined
        )

    elif d == 2:
        plt_state_2D(
            gp, acq_res, f, bounds, it,
            x_best, y_best, save_cfg,
            acq_cfg, bo_cfg, acq=acq,
            n_plot=n_plot, f_true=f_true,
            x_new=x_new, y_new=y_new, x_refined=x_refined, y_refined=y_refined
        )

    else:
        # Graceful fallback
        print(f"[plt_state] Plot not supported for dimension d={d}. Skipping.")



def plt_state_1D(
    gp: GPModel,
    acq_res,
    f: Callable[[Array], Array],
    bounds: Bounds,
    it: int,
    x_best: Array,
    y_best: float,
    save_cfg: SaveConfig,
    acq_cfg: AcqConfig,
    bo_cfg: BOConfig,
    acq: Callable[[Array], Array],
    n_plot: int = 400,
    f_true: Optional[Callable[[Array], Array]] = None,
    x_new: Optional[Array] = None,
    y_new: Optional[Array] = None,
    x_refined: Optional[Array] = None,
    y_refined: Optional[Array] = None,
):
    """
    Plot current BO state (1D only).

    Top:
      - noisy true function (single realization)
      - observations
      - GP posterior mean + 95% CI

    Bottom:
      - acquisition values on candidate set
    """
    # Safety checks
    if gp.Xs is None or gp.ys is None:
        return

    if gp.Xs.shape[1] != 1:
        # plotting only supported in 1D
        return

    # Build dense grid
    x_min, x_max = bounds[0]
    Xplot = np.linspace(x_min, x_max, n_plot).reshape(-1, 1)

    # True (noisy) function
    if f_true is not None:
        y_true = np.asarray(f_true(Xplot)).reshape(-1)
    else:
        y_true = None

    # GP posterior
    mu, var = gp_predict(
        Xplot, gp.Xs, gp.alpha, gp.L,
        l_c=gp.l_c, sigma_f=gp.sigma_f,
        return_cov=False,
        gp=gp
    )

    mu = mu.reshape(-1)
    std = np.sqrt(var.reshape(-1))

    # Observations
    Xs = gp.Xs.reshape(-1)
    ys = gp.ys.reshape(-1)

    # Acquisition values
    Xcand = acq_res.Xcand.reshape(-1)
    a = acq_res.a.reshape(-1)

    idx = np.argsort(Xcand)
    Xcand = Xcand[idx]
    a = a[idx]

    # Plot
    fig, axs = plt.subplots(
        2, 1, figsize=(6, 5),
        constrained_layout=True,
        sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1])
    )

    # Top: function + GP
    axs[0].plot(Xplot[:, 0], y_true, "k--", lw=1.0, label="True (unknown)")
    axs[0].plot(Xplot[:, 0], mu, "C0", lw=2, label="$\\mu_{\\mathcal{GP}}$")
    axs[0].fill_between(
        Xplot[:, 0],
        mu - 2 * std,
        mu + 2 * std,
        color="C0",
        alpha=0.25,
        label="$\\mu_{\\mathcal{GP}} \\pm 2 \\sigma$",
    )
    axs[0].scatter(Xs, ys, c="k", s=20, zorder=10, label="Observations")
    if x_new is not None and y_new is not None and len(y_new) > 0:
        axs[0].scatter(
            np.asarray(x_new).reshape(-1), np.asarray(y_new).reshape(-1),
            c="C3", s=25, zorder=11, label="Evaluated this iter",
        )
    if x_refined is not None and y_refined is not None:
        xr = np.asarray(x_refined, dtype=float).reshape(-1)
        yr = np.asarray(y_refined, dtype=float).reshape(-1)
        ok = np.isfinite(xr) & np.isfinite(yr)
        if ok.any():
            axs[0].scatter(
                xr[ok], yr[ok], marker="^", s=45, facecolors="none",
                edgecolors="C3", zorder=11, label="Refined (ADAM)",
            )
    axs[0].scatter(x_best, y_best, marker="*", s=20, zorder=12, label="Best so far")

    axs[0].set_ylabel("f(x)")
    axs[0].set_title(f"Iteration {it}/{bo_cfg.n_iter}")
    axs[0].legend(fontsize=8, loc='upper right')

    #  Bottom: acquisition
    axs[1].plot(Xcand, a, "C1", lw=1.5, label=acq_cfg.kind)
    axs[1].fill_between(
        x= Xcand,
        y1= a,
        color= "C1",
        alpha= 0.2
    )
    axs[1].scatter(acq_res.x_next, acq_res.a_best, c="C1", label='Next query point')
    axs[1].legend(fontsize=8, loc='lower right')

    axs[1].set_ylabel("acq(x)")
    axs[1].set_xlabel("x")
    if save_cfg.out_path:
        fig_path = os.path.join(save_cfg.out_path, "plots", "GIF")
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        figname = os.path.join(fig_path, f"it_{it:03d}.png")
        plt.savefig(figname, dpi=250, bbox_inches="tight")
    _finish_figure(save_cfg)


def plt_state_2D(
    gp: GPModel,
    acq_res,
    f: Callable[[Array], Array],
    bounds: Bounds,
    it: int,
    x_best: Array,
    y_best: float,
    save_cfg: SaveConfig,
    acq_cfg: AcqConfig,
    bo_cfg: BOConfig,
    acq: Callable[[Array], Array],
    n_plot: int = 50,
    f_true: Optional[Callable[[Array], Array]] = None,
    n_init: Optional[int] = None,
    x_new: Optional[Array] = None,
    y_new: Optional[Array] = None,
    x_refined: Optional[Array] = None,
    y_refined: Optional[Array] = None,
):
    """
    Plot current BO state (2D only).

    Subplots:
      1. GP posterior mean
      2. GP posterior standard deviation
      3. Acquisition map evaluated on the same dense grid

    Overlaid on each subplot:
      - initial dataset points
      - BO-explored points
      - best observed point
      - next query point
    """
    # ------------------------------------------------------------------
    # Safety checks
    if gp.Xs is None or gp.ys is None:
        return

    if gp.Xs.shape[1] != 2:
        return

    # ------------------------------------------------------------------
    # Infer n_init if not provided
    if n_init is None:
        n_init = getattr(bo_cfg, "n_init", None)

    if n_init is None:
        raise ValueError(
            "n_init is required to distinguish initial dataset points "
            "from BO-explored points."
        )

    # ------------------------------------------------------------------
    # Dense grid
    x1_min, x1_max = bounds[0]
    x2_min, x2_max = bounds[1]

    x1 = np.linspace(x1_min, x1_max, n_plot)
    x2 = np.linspace(x2_min, x2_max, n_plot)
    X1, X2 = np.meshgrid(x1, x2)
    Xplot = np.column_stack([X1.ravel(), X2.ravel()])

    # ------------------------------------------------------------------
    # GP posterior on dense grid
    mu, var = gp_predict(
        Xplot, gp.Xs, gp.alpha, gp.L,
        l_c=gp.l_c, sigma_f=gp.sigma_f,
        return_cov=False,
        gp=gp
    )



    mu = mu.reshape(n_plot, n_plot)
    std = np.sqrt(var.reshape(n_plot, n_plot))

    # ------------------------------------------------------------------
    # Acquisition on the same dense grid
    # The cleanest setup is that `acq` is already a closure/lambda with all
    # needed BO parameters baked in, so it only needs X as input.
    a = np.asarray(acq(Xplot)).reshape(-1)
    a_grid = a.reshape(n_plot, n_plot)

    # ------------------------------------------------------------------
    # Observations
    Xs = np.asarray(gp.Xs)
    n_init = min(n_init, Xs.shape[0])

    X_init = Xs[:n_init]
    X_bo = Xs[n_init:]

    x_best = np.asarray(x_best).reshape(-1)
    x_next = np.asarray(acq_res.x_next).reshape(-1)

    # ------------------------------------------------------------------
    # Plot settings
    titles = [
        r"Posterior mean $\mu(x_1, x_2)$",
        r"Posterior uncertainty $\sigma(x_1, x_2)$",
        "Expected Improvement" if acq_cfg.kind.lower() == "ei" else f"Acquisition ({acq_cfg.kind})",
    ]
    cmaps = ["viridis", "plasma", "cividis"]

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    fig.subplots_adjust(bottom=0.24, wspace=0.38)

    def _overlay_points(ax):
        if len(X_init) > 0:
            ax.scatter(
                X_init[:, 0], X_init[:, 1],
                c="green", edgecolor="k", s=50,
                zorder=10, clip_on=False,
                label="Initial dataset"
            )

        if len(X_bo) > 0:
            ax.scatter(
                X_bo[:, 0], X_bo[:, 1],
                c="blue", edgecolor="k", s=50,
                zorder=10, clip_on=False,
                label="BO explored"
            )

        if x_new is not None and len(x_new) > 0:
            xn = np.asarray(x_new, dtype=float).reshape(-1, 2)
            ax.scatter(
                xn[:, 0], xn[:, 1],
                c="red", edgecolor="k", s=40,
                zorder=10, clip_on=False,
                label="Evaluated this iter"
            )

        if x_refined is not None:
            xr = np.asarray(x_refined, dtype=float).reshape(-1, 2)
            xr = xr[np.all(np.isfinite(xr), axis=1)]
            if len(xr) > 0:
                ax.scatter(
                    xr[:, 0], xr[:, 1],
                    facecolors="none", edgecolor="red",
                    marker="^", s=70,
                    zorder=11, clip_on=False,
                    label="Refined (ADAM)"
                )

        ax.scatter(
            x_best[0], x_best[1],
            c="white", edgecolor="k",
            marker="D", s=60,
            zorder=11, clip_on=False,
            label="Best so far"
        )

        ax.scatter(
            x_next[0], x_next[1],
            c="white", edgecolor="k",
            marker="*", s=180,
            zorder=12, clip_on=False,
            label="Next query point"
        )


        ax.set_aspect("equal")

    # ------------------------------------------------------------------
    # 1) Mean
    c0 = axs[0].contourf(X1, X2, mu, levels=100, cmap=cmaps[0])
    axs[0].set_title(titles[0])
    _overlay_points(axs[0])
    cb0 = fig.colorbar(c0, ax=axs[0], fraction=0.046, pad=0.04)
    cb0.set_label(r"$\mu_{\mathcal{GP}}$")
    axs[0].set_xlabel(r"$x_1$")
    axs[0].set_ylabel(r"$x_2$")

    # ------------------------------------------------------------------
    # 2) Std
    c1 = axs[1].contourf(X1, X2, std, levels=100, cmap=cmaps[1])
    axs[1].set_title(titles[1])
    _overlay_points(axs[1])
    cb1 = fig.colorbar(c1, ax=axs[1], fraction=0.046, pad=0.04)
    cb1.set_label(r"$\sigma_{\mathcal{GP}}$")
    axs[1].set_xlabel(r"$x_1$")

    # ------------------------------------------------------------------
    # 3) Acquisition
    c2 = axs[2].contourf(X1, X2, a_grid, levels=100, cmap=cmaps[2])
    axs[2].set_title(titles[2])
    _overlay_points(axs[2])
    cb2 = fig.colorbar(c2, ax=axs[2], fraction=0.046, pad=0.04)
    cb2.set_label(acq_cfg.kind.upper())
    cb2.formatter = FormatStrFormatter("%.2e")   # always scientific notation, 2 decimals
    cb2.update_ticks()
    axs[2].set_xlabel(r"$x_1$")

    # if acq_cfg.kind.lower() == "ei":
    #     cb2.formatter.set_powerlimits((0, 0))
    #     cb2.update_ticks()

    # ------------------------------------------------------------------
    # Shared legend
    handles = [
        plt.Line2D(
            [0], [0],
            marker="o", linestyle="None",
            markerfacecolor="green", markeredgecolor="k",
            markersize=8, label="Initial dataset"
        ),
        plt.Line2D(
            [0], [0],
            marker="o", linestyle="None",
            markerfacecolor="blue", markeredgecolor="k",
            markersize=8, label="BO explored"
        ),
        plt.Line2D(
            [0], [0],
            marker="D", linestyle="None",
            markerfacecolor="white", markeredgecolor="k",
            markersize=8, label="Best so far"
        ),
        plt.Line2D(
            [0], [0],
            marker="*", linestyle="None",
            markerfacecolor="white", markeredgecolor="k",
            markersize=12, label="Next query point"
        ),
    ]
    if x_new is not None and len(x_new) > 0:
        handles.append(plt.Line2D(
            [0], [0],
            marker="o", linestyle="None",
            markerfacecolor="red", markeredgecolor="k",
            markersize=8, label="Evaluated this iter"
        ))
    if x_refined is not None and np.isfinite(np.asarray(x_refined, dtype=float)).all(axis=-1).any():
        handles.append(plt.Line2D(
            [0], [0],
            marker="^", linestyle="None",
            markerfacecolor="none", markeredgecolor="red",
            markersize=8, label="Refined (ADAM)"
        ))

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3 if len(handles) > 4 else 4,
        bbox_to_anchor=(0.5, 0.04),
        frameon=True,
        fontsize=10
    )

    fig.suptitle(f"Iteration {it}/{bo_cfg.n_iter}", fontsize=14, y=0.98)

    # ------------------------------------------------------------------
    # Save
    if save_cfg.out_path:
        fig_path = os.path.join(save_cfg.out_path, "plots", "GIF")
        os.makedirs(fig_path, exist_ok=True)
        figname = os.path.join(fig_path, f"it_{it:03d}.png")
        plt.savefig(figname, dpi=250, bbox_inches="tight")

    _finish_figure(save_cfg)

def plt_conv(hist: List[Dict[str, Any]], save_cfg: SaveConfig):

    # Extract the values fromt eh list of dictionaries
    its      = [h["it"] for h in hist]
    best_y  = [h["best_y"] for h in hist]

    # Plot
    plt.figure(figsize=(5,3))
    plt.plot(its, best_y, "b-o")
    plt.xlabel("Calls $n$")
    plt.ylabel("min $f(x)$ after $n$ calls")
    plt.grid(True)
    plt.tight_layout()   # keep the axis labels inside the canvas
    if save_cfg.out_path:
        # Ensure the saving path exists
        plots_path = os.path.join(save_cfg.out_path, "plots")
        os.makedirs(plots_path, exist_ok=True)

        figname = os.path.join(plots_path, "conv.png")
        plt.savefig(figname, dpi=save_cfg.dpi, bbox_inches="tight")
    _finish_figure(save_cfg)


def plt_hist(hist: List[Dict[str, Any]], save_cfg: SaveConfig):

    # Extract the values fromt eh list of dictionaries
    its      = [h["it"] for h in hist]
    y_next  = [h["y_next"] for h in hist]

    # Plot
    plt.figure(figsize=(5,3))
    plt.plot(its, y_next, "b-o")
    plt.xlabel("Calls $n$")
    plt.ylabel("$f(x)$ at each call")
    plt.grid(True)
    plt.tight_layout()   # keep the axis labels inside the canvas
    if save_cfg.out_path:
        # Ensure the saving path exists
        plots_path = os.path.join(save_cfg.out_path, "plots")
        os.makedirs(plots_path, exist_ok=True)

        figname = os.path.join(plots_path, "hist.png")
        plt.savefig(figname, dpi=save_cfg.dpi, bbox_inches="tight")
    _finish_figure(save_cfg)
