"""
Figures comparing sbo and scikit-optimize over repeated runs.

plot_comparison  (mean +/- std over the runs)
    (a) best-so-far cost vs number of evaluations
    (b) final best cost
    (c) total time, split into objective (PDE solves) and optimizer overhead
plot_per_run     (one entry per run; sbo and scikit-optimize of a run share the seed)
    (a) final best cost of every run
    (b) total time of every run

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from solvers import RunResult

# One colour per method, colour-blind safe; identical in all panels and figures.
COLORS = {"sbo": "#0072B2", "scikit-optimize": "#009E73"}
LIGHT = "lightgrey"      # objective (PDE solves) part of the time bars


def _stack(runs: List[RunResult], attr: str) -> np.ndarray:
    return np.array([getattr(r, attr) for r in runs], dtype=float)


def _std(a):
    """Sample std over the runs (axis 0); 0 if there is a single run."""
    return np.std(a, axis=0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1:])


def _save(fig, out_path: Path, show: bool) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def _convergence_decor(ax, n_init: int, baseline_cost: Optional[float]) -> None:
    if baseline_cost is not None:
        ax.axhline(baseline_cost, color="k", ls=":", lw=1, label="no control (w = 0)")
    ax.axvline(n_init, color="grey", ls="--", lw=0.8)
    ax.text(n_init, 0.45, " end of shared\n initial design", transform=ax.get_xaxis_transform(),
            va="bottom", fontsize=8, color="grey")
    ax.set_yscale("log")
    ax.set_xlabel("number of evaluations")
    ax.set_ylabel("best cost so far (lower is better)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")


def _title(fig, results) -> None:
    first = results[list(results)[0]]
    n_init, n_new = first[0].n_init, len(first[0].y) - first[0].n_init
    fig.suptitle(f"Burgers control, 3 weights: {n_init} shared initial points + {n_new} "
                 f"evaluations, {len(first)} run(s)", fontsize=12)


# ---------------------------------------------------------------------------
def plot_comparison(results: Dict[str, List[RunResult]], out_path: Path,
                    baseline_cost: Optional[float] = None, show: bool = False) -> None:
    methods = list(results)
    n_init = results[methods[0]][0].n_init
    x = np.arange(len(methods))
    labels = [m.replace(" ", "\n", 1) for m in methods]
    colors = [COLORS[m] for m in methods]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1.6, 0.8, 1.0]))

    # (a) convergence: mean +/- std
    ax = axs[0]
    for m in methods:
        curves = _stack(results[m], "best_so_far")                # (runs, n_total)
        mean, std = curves.mean(axis=0), _std(curves)
        n = np.arange(1, curves.shape[1] + 1)
        ax.plot(n, mean, color=COLORS[m], lw=2, label=f"{m} (mean $\\pm$ std)")
        ax.fill_between(n, np.maximum(mean - std, 1e-9), mean + std, color=COLORS[m], alpha=0.2)
    _convergence_decor(ax, n_init, baseline_cost)
    ax.set_title("Convergence")

    # (b) final best: mean bar + std error bar, dots = runs
    ax = axs[1]
    finals = [_stack(results[m], "best_y") for m in methods]
    means = np.array([f.mean() for f in finals])
    stds = np.array([float(_std(f)) for f in finals])
    ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.85, width=0.6,
           error_kw=dict(ecolor="k", lw=1.5))
    for i, f in enumerate(finals):
        ax.scatter(np.full(len(f), i), f, color="k", s=14, zorder=3)
        ax.text(i, means[i] * 0.04, f"{means[i]:.0f}\n$\\pm${stds[i]:.0f}", ha="center", va="bottom",
                fontsize=10, color="white", fontweight="bold")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("final best cost")
    ax.set_title("Best obtained (mean $\\pm$ std)")
    ax.grid(alpha=0.3, axis="y")

    # (c) time: stacked mean bar, std error bar on the total
    ax = axs[2]
    obj = np.array([_stack(results[m], "objective_time").mean() for m in methods])
    over = np.array([_stack(results[m], "overhead_time").mean() for m in methods])
    tot_std = np.array([float(_std(_stack(results[m], "total_time"))) for m in methods])
    ax.bar(x, obj, color=LIGHT, width=0.6, label="objective (PDE solves)")
    ax.bar(x, over, bottom=obj, color=colors, width=0.6, label="optimizer overhead")
    ax.errorbar(x, obj + over, yerr=tot_std, fmt="none", ecolor="k", lw=1.5, capsize=6)
    for i in range(len(methods)):
        ax.text(i, obj[i] + over[i] + tot_std[i], f"{obj[i] + over[i]:.1f} $\\pm$ {tot_std[i]:.1f} s\n"
                f"(optimizer {over[i]:.1f} s)", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("wall time [s]")
    ax.set_title("Time taken (mean $\\pm$ std)")
    ax.set_ylim(top=(obj + over + tot_std).max() * 1.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    _title(fig, results)
    _save(fig, out_path, show)


# ---------------------------------------------------------------------------
def plot_per_run(results: Dict[str, List[RunResult]], out_path: Path,
                 baseline_cost: Optional[float] = None, show: bool = False) -> None:
    methods = list(results)
    n_runs = len(results[methods[0]])
    idx = np.arange(n_runs)
    width = 0.38
    offsets = {m: (k - (len(methods) - 1) / 2) * width for k, m in enumerate(methods)}

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)

    # (a) final best per run
    ax = axs[0]
    for m in methods:
        ax.bar(idx + offsets[m], _stack(results[m], "best_y"), width=width, color=COLORS[m], label=m)
    ax.set_xticks(idx, [f"run {i + 1}" for i in range(n_runs)], fontsize=8)
    ax.set_ylabel("final best cost")
    ax.set_title("Best obtained, per run")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    # (b) time per run (objective part in grey, optimizer overhead in colour)
    ax = axs[1]
    for k, m in enumerate(methods):
        obj, over = _stack(results[m], "objective_time"), _stack(results[m], "overhead_time")
        ax.bar(idx + offsets[m], obj, width=width, color=LIGHT, label="objective (PDE solves)" if k == 0 else None)
        ax.bar(idx + offsets[m], over, bottom=obj, width=width, color=COLORS[m], label=f"optimizer overhead, {m}")
    ax.set_xticks(idx, [f"run {i + 1}" for i in range(n_runs)], fontsize=8)
    ax.set_ylabel("wall time [s]")
    ax.set_title("Time taken, per run")
    ax.set_ylim(top=max((_stack(results[m], "total_time").max() for m in methods)) * 1.35)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    _title(fig, results)
    _save(fig, out_path, show)
