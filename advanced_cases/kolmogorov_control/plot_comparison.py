"""
Figures comparing sbo, sbo with the rank-one update and scikit-optimize over
repeated runs.

plot_comparison  (mean +/- std over the runs)
    (a) best-so-far cost vs number of evaluations
    (b) final best cost
    (c) total time, split into objective (simulations), GP update and acquisition search
plot_overhead    (mean over the runs, against the number of points in the GP)
    (a) time of the GP update of every iteration
    (b) time of the acquisition search of every iteration
    (c) cumulative optimizer time
plot_per_run     (one entry per run; the methods of a run share the seed)
    (a) final best cost of every run
    (b) total time of every run

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from solvers import RunResult

# One colour (colour-blind safe) and one line style per method; identical in all panels and figures.
COLORS = {"sbo": "#0072B2", "sbo (rank-one)": "#D55E00", "scikit-optimize": "#009E73"}
STYLES = {"sbo": "-", "sbo (rank-one)": "--", "scikit-optimize": "-."}
LIGHT = "lightgrey"      # objective (simulations) part of the time bars
ACQ_ALPHA = 0.4          # acquisition part of the time bars: method colour, lighter
# Same look whether or not sbo was run in the session (importing its plotting module switches LaTeX on).
RC = {"text.usetex": False, "font.family": "sans-serif", "axes.labelsize": 10,
      "xtick.labelsize": 9, "ytick.labelsize": 9}


def _stack(runs: List[RunResult], attr: str) -> np.ndarray:
    return np.array([getattr(r, attr) for r in runs], dtype=float)


def _std(a):
    """Sample std over the runs (axis 0); 0 if there is a single run."""
    return np.std(a, axis=0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1:])


def _running_median(a: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(a, half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=-1)


def _save(fig, out_path: Path, show: bool) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def _title(fig, results) -> None:
    first = results[list(results)[0]]
    n_init, n_new = first[0].n_init, len(first[0].y) - first[0].n_init
    fig.suptitle(f"Kolmogorov flow control, {first[0].X.shape[1]} unknowns: {n_init} shared initial points + "
                 f"{n_new} evaluations, {len(first)} run(s)", fontsize=12)


def _time_parts(runs: List[RunResult]):
    """Per run: time in the objective, in the GP update and in the acquisition search."""
    obj = _stack(runs, "objective_time")
    fit = np.array([r.fit_time.sum() for r in runs])
    return obj, fit, _stack(runs, "total_time") - obj - fit


def _time_legend(ax) -> None:
    ax.legend(handles=[Patch(color=LIGHT, label="objective (simulations)"),
                       Patch(color="0.25", label="GP update"),
                       Patch(color="0.25", alpha=ACQ_ALPHA, label="acquisition search")],
              fontsize=8, loc="upper left")


# ---------------------------------------------------------------------------
def plot_comparison(results: Dict[str, List[RunResult]], out_path: Path,
                    baseline_cost: Optional[float] = None, relative: bool = False,
                    show: bool = False) -> None:
    """relative=False: costs as they are, log scale. relative=True: costs minus
    the uncontrolled one, linear scale (for small changes of a large cost)."""
    methods = list(results)
    n_init = results[methods[0]][0].n_init
    ref = baseline_cost if relative else 0.0
    rel = "cost $-$ cost without control" if relative else "cost"
    x = np.arange(len(methods))
    labels = [m.replace(" ", "\n", 1) for m in methods]
    colors = [COLORS[m] for m in methods]

    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True,
                                gridspec_kw=dict(width_ratios=[1.6, 0.8, 1.0]))

        # (a) convergence: mean +/- std
        ax = axs[0]
        for m in methods:
            curves = _stack(results[m], "best_so_far") - ref              # (runs, n_total)
            mean, std = curves.mean(axis=0), _std(curves)
            n = np.arange(1, curves.shape[1] + 1)
            ax.plot(n, mean, color=COLORS[m], ls=STYLES[m], lw=2, label=f"{m} (mean $\\pm$ std)")
            ax.fill_between(n, mean - std if relative else np.maximum(mean - std, 1e-9), mean + std,
                            color=COLORS[m], alpha=0.2, lw=0)
        if baseline_cost is not None:
            ax.axhline(baseline_cost - ref, color="k", ls=":", lw=1, label="no control (x = 0)")
        if not relative:
            ax.set_yscale("log")
        ax.axvline(n_init, color="grey", ls="--", lw=0.8)
        ax.text(n_init, 0.97, " end of shared\n initial design", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, color="grey")
        ax.set_xscale("log")
        ax.set_xlabel("number of evaluations")
        ax.set_ylabel(f"best {rel} so far (lower is better)")
        ax.set_title("Convergence")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3, which="both")

        # (b) final best: mean bar + std error bar, dots = runs
        ax = axs[1]
        finals = [_stack(results[m], "best_y") - ref for m in methods]
        means = np.array([f.mean() for f in finals])
        stds = np.array([float(_std(f)) for f in finals])
        ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.85, width=0.6,
               error_kw=dict(ecolor="k", lw=1.5))
        for i, f in enumerate(finals):
            ax.scatter(np.full(len(f), i), f, color="k", s=14, zorder=3)
            ax.annotate(f"{means[i]:.3g}\n$\\pm${stds[i]:.2g}", (i, means[i] / 2), ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylabel(f"final best {rel}")
        ax.set_title("Best obtained (mean $\\pm$ std)")
        ax.grid(alpha=0.3, axis="y")

        # (c) time: stacked mean bar, std error bar on the total
        ax = axs[2]
        obj, fit, acq = (np.array(v) for v in zip(*[[p.mean() for p in _time_parts(results[m])] for m in methods]))
        tot_std = np.array([float(_std(_stack(results[m], "total_time"))) for m in methods])
        ax.bar(x, obj, color=LIGHT, width=0.6)
        ax.bar(x, fit, bottom=obj, color=colors, width=0.6)
        ax.bar(x, acq, bottom=obj + fit, color=colors, alpha=ACQ_ALPHA, width=0.6)
        ax.errorbar(x, obj + fit + acq, yerr=tot_std, fmt="none", ecolor="k", lw=1.5, capsize=6)
        for i in range(len(methods)):
            ax.text(i, obj[i] + fit[i] + acq[i] + tot_std[i],
                    f"{(obj[i] + fit[i] + acq[i]) / 60:.1f} $\\pm$ {tot_std[i] / 60:.1f} min\n"
                    f"GP {fit[i] / 60:.1f} + acq. {acq[i] / 60:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylabel("wall time [s]")
        ax.set_title("Time taken (mean $\\pm$ std)")
        ax.set_ylim(top=(obj + fit + acq + tot_std).max() * 1.45)
        _time_legend(ax)
        ax.grid(alpha=0.3, axis="y")

        _title(fig, results)
        _save(fig, out_path, show)


# ---------------------------------------------------------------------------
def plot_overhead(results: Dict[str, List[RunResult]], out_path: Path, hpo_every: int,
                  rank_one_threshold: int, show: bool = False) -> None:
    methods = list(results)
    first = results[methods[0]][0]
    n_points = first.n_init + np.arange(len(first.iter_time))     # points in the GP at each iteration
    hpo = np.arange(len(n_points)) % hpo_every == 0               # iterations with hyperparameter optimization

    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

        for m in methods:
            fit = _stack(results[m], "fit_time").mean(axis=0)
            total = _stack(results[m], "iter_time").mean(axis=0)
            style = dict(color=COLORS[m], ls=STYLES[m], lw=2)
            # (a) GP update: line = iterations with fixed hyperparameters, dots = with optimization
            axs[0].plot(n_points[~hpo], _running_median(fit[~hpo], 9), label=m, **style)
            axs[0].plot(n_points[hpo], fit[hpo], "o", color=COLORS[m], ms=3, alpha=0.5)
            # (b) acquisition search
            axs[1].plot(n_points, _running_median(total - fit, 25), label=m, **style)
            # (c) cumulative optimizer time
            axs[2].plot(n_points, np.cumsum(total) / 60, label=m, **style)

        axs[0].set_title(f"GP update, per iteration\n(dots: with hyperparameter optimization, every {hpo_every})")
        axs[1].set_title("Acquisition search, per iteration\n(running median)")
        axs[2].set_title("Optimizer time, cumulative\n(GP update + acquisition search)")
        for ax, ylabel in zip(axs, ("time [s]", "time [s]", "time [min]")):
            if n_points[0] < rank_one_threshold < n_points[-1]:
                ax.axvline(rank_one_threshold, color="grey", ls="--", lw=0.8)
                ax.text(rank_one_threshold, 0.03, " rank-one\n threshold", transform=ax.get_xaxis_transform(),
                        va="bottom", fontsize=8, color="grey")
            ax.set_xlabel("number of points in the GP")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3, which="both")
        axs[0].set_yscale("log")
        axs[1].set_yscale("log")
        axs[0].legend(handles=[Line2D([], [], color=COLORS[m], ls=STYLES[m], lw=2, label=m) for m in methods],
                      fontsize=8, loc="upper left")

        _title(fig, results)
        _save(fig, out_path, show)


# ---------------------------------------------------------------------------
def plot_per_run(results: Dict[str, List[RunResult]], out_path: Path,
                 baseline_cost: Optional[float] = None, relative: bool = False,
                 show: bool = False) -> None:
    methods = list(results)
    n_runs = len(results[methods[0]])
    ref = baseline_cost if relative else 0.0
    idx = np.arange(n_runs)
    width = 0.8 / len(methods)
    offsets = {m: (k - (len(methods) - 1) / 2) * width for k, m in enumerate(methods)}

    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

        # (a) final best per run
        ax = axs[0]
        for m in methods:
            ax.bar(idx + offsets[m], _stack(results[m], "best_y") - ref, width=width, color=COLORS[m], label=m)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(idx, [f"run {i + 1}" for i in range(n_runs)], fontsize=8)
        ax.set_ylabel("final best cost" + (" $-$ cost without control" if relative else ""))
        ax.set_title("Best obtained, per run")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3, axis="y")

        # (b) time per run (objective in grey, GP update in colour, acquisition search lighter)
        ax = axs[1]
        for m in methods:
            obj, fit, acq = _time_parts(results[m])
            ax.bar(idx + offsets[m], obj, width=width, color=LIGHT)
            ax.bar(idx + offsets[m], fit, bottom=obj, width=width, color=COLORS[m])
            ax.bar(idx + offsets[m], acq, bottom=obj + fit, width=width, color=COLORS[m], alpha=ACQ_ALPHA)
        ax.set_xticks(idx, [f"run {i + 1}" for i in range(n_runs)], fontsize=8)
        ax.set_ylabel("wall time [s]")
        ax.set_title("Time taken, per run")
        ax.set_ylim(top=max((_stack(results[m], "total_time").max() for m in methods)) * 1.35)
        _time_legend(ax)
        ax.grid(alpha=0.3, axis="y")

        _title(fig, results)
        _save(fig, out_path, show)
