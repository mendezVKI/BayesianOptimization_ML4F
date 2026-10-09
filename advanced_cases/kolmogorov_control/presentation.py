"""
Presentation material for the Kolmogorov control case: the vorticity field at
the end of the episode with the best schedule found, the flow it is compared
with (the target flow, or the uncontrolled one for the energy objective),
their difference, and the schedule itself.

    python presentation.py                      # schedule = best found by run_comparison.py
    python presentation.py --results-dir results/<case folder>

Output: assets/fields.png

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kolmogorov_case import HERE, ACTION_BOUND, KolmogorovControl

ASSETS = HERE / "assets"
RESULTS = HERE / "results"


def best_from_results(results_dir: Path | None = None):
    """Best schedule of `results_dir` (default: the most recent study in results/)
    and the settings of that study."""
    if results_dir is None:
        results_dir = max(RESULTS.glob("*/study.json"), key=lambda p: p.stat().st_mtime).parent
    print(f"Reading {results_dir}")
    best, x = np.inf, None
    for path in (results_dir / "runs").glob("*.npz"):
        data = np.load(path)
        i = int(np.argmin(data["y"]))
        if data["y"][i] < best:
            best, x = data["y"][i], data["X"][i]
    return x, json.loads((results_dir / "study.json").read_text())["settings"]


def _plot_schedule(fig, ax, problem: KolmogorovControl, x: np.ndarray, title: str) -> None:
    """One row per segment, one column per forcing mode."""
    sched = x.reshape(problem.n_segments, problem.n_modes)
    im = ax.imshow(sched, cmap="RdBu_r", vmin=-ACTION_BOUND, vmax=ACTION_BOUND, aspect="auto")
    if problem.n_modes <= 8:                         # room to write the values
        for (i, j), a in np.ndenumerate(sched):
            ax.text(j, i, f"{a:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(a) > 0.3 else "black")
    ax.set_xticks(range(problem.n_modes), problem.mode_names, fontsize=8,
                  rotation=90 if problem.n_modes > 4 else 0)
    ax.set_yticks(range(problem.n_segments), [f"segment {i + 1}" for i in range(problem.n_segments)])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="amplitude", shrink=0.85)


def plot_fields(problem: KolmogorovControl, x: np.ndarray, path: Path) -> None:
    """Vorticity at the end of the episode with schedule x, against the flow it
    is compared with: the target flow (tracking) or the uncontrolled one (energy)."""
    tracking = problem.objective == "tracking"
    cost0, energy0, w0 = problem.rollout(np.zeros(problem.dim))
    cost, energy, w = problem.rollout(x)
    if tracking:
        w_ref = problem.rollout(problem.x_ref)[2]
        fields = ((w_ref, "target flow"),
                  (w, f"schedule found: cost = {cost:.2f}\n(no control: {cost0:.0f})"),
                  (w - w_ref, f"schedule found - target\n(no control - target: up to {np.abs(w0 - w_ref).max():.2f})"))
    else:
        fields = ((w0, f"no control: cost = {cost0:.2f}\nmean energy = {energy0.mean():.4f}"),
                  (w, f"schedule found: cost = {cost:.2f}\nmean energy = {energy.mean():.4f}"),
                  (w - w0, "schedule found - no control"))

    n_panels = 5 if tracking else 4
    fig, axs = plt.subplots(1, n_panels, figsize=(4.4 * n_panels, 4.2), constrained_layout=True)
    extent = (0, 2 * np.pi, 0, 2 * np.pi)
    for k, (ax, (field, title)) in enumerate(zip(axs, fields)):
        vmax = np.abs(field).max() if k == 2 else np.abs(fields[0][0]).max()     # the difference has its own scale
        im = ax.imshow(field.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set(xlabel="x", ylabel="y", title=title)
        fig.colorbar(im, ax=ax, label="vorticity", shrink=0.85)

    _plot_schedule(fig, axs[3], problem, x, "schedule found\n(amplitude of each forcing mode)")
    if tracking:
        _plot_schedule(fig, axs[4], problem, problem.x_ref, "reference schedule\n(hidden from the optimizers)")

    fig.suptitle(f"Kolmogorov flow, vorticity at the end of the episode (t = {problem.n_segments * problem.segment_time:g})")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="results sub-folder to read the best schedule from; default: the most recent one")
    args = ap.parse_args()

    x, settings = best_from_results(args.results_dir)
    problem = KolmogorovControl(objective=settings["objective"], n_segments=settings["n_segments"],
                                alpha=settings["alpha"], wavenumbers=tuple(settings["wavenumbers"]),
                                both_phases=settings["both_phases"])
    print(f"Best schedule:\n{np.round(x.reshape(problem.n_segments, problem.n_modes), 3)}")

    ASSETS.mkdir(exist_ok=True)
    plot_fields(problem, x, ASSETS / "fields.png")
    print(f"Saved to {ASSETS}")


if __name__ == "__main__":
    main()
