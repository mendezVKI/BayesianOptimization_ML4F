"""
Solve the Burgers control case with sbo and with scikit-optimize (same GP, same
hyperparameter optimization, same normalization), time both and plot the comparison.

    python run_comparison.py                       # 5 runs, each with its own random seed
    python run_comparison.py --n-runs 10 --master-seed 0   # more runs, reproducible seeds
    python run_comparison.py --n-iter 50 --n-runs 2        # quick run
    python run_comparison.py --plot-only           # re-plot (same --n-init/--n-iter/--n-runs as the run)

Outputs (in results/n_init<..>_n_iter<..>_n_runs<..>/): comparison.npz (every evaluation of every run),
summary.csv (one row per method), comparison.png (mean +/- std), comparison_per_run.png.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from burgers_case import BurgersControl, HERE
from solvers import METHODS, RunResult, run_all
from plot_comparison import plot_comparison, plot_per_run

RESULTS = HERE / "results"

# Inputs of the runs
BOUNDS = [(-0.1, 0.1)] * 3        # search space: one range per feedback weight


def save_results(results: Dict[str, List[RunResult]], baseline_cost: float, path: Path) -> None:
    arrays = {"baseline_cost": baseline_cost}
    for m, runs in results.items():
        key = m.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        arrays[f"{key}__X"] = np.array([r.X for r in runs])
        arrays[f"{key}__y"] = np.array([r.y for r in runs])
        arrays[f"{key}__times"] = np.array([[r.total_time, r.objective_time] for r in runs])
        arrays[f"{key}__seeds"] = np.array([r.seed for r in runs])
        arrays[f"{key}__n_init"] = runs[0].n_init
    np.savez(path, **arrays)


def load_results(path: Path):
    data = np.load(path)
    results = {}
    for m in METHODS:
        key = m.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        results[m] = [
            RunResult(m, int(seed), X, y, int(data[f"{key}__n_init"]), t[0], t[1])
            for seed, X, y, t in zip(data[f"{key}__seeds"], data[f"{key}__X"],
                                     data[f"{key}__y"], data[f"{key}__times"])
        ]
    return results, float(data["baseline_cost"])


def write_summary(results: Dict[str, List[RunResult]], path: Path) -> None:
    header = ["method", "n_runs", "best_cost_mean", "best_cost_std", "best_cost_min", "best_cost_max",
              "time_total_mean_s", "time_total_std_s", "time_objective_mean_s", "time_optimizer_mean_s"]
    sd = lambda a: np.std(a, ddof=1) if len(a) > 1 else 0.0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for m, runs in results.items():
            best = np.array([r.best_y for r in runs])
            total = np.array([r.total_time for r in runs])
            w.writerow([m, len(runs), f"{best.mean():.2f}", f"{sd(best):.2f}",
                        f"{best.min():.2f}", f"{best.max():.2f}",
                        f"{total.mean():.2f}", f"{sd(total):.2f}",
                        f"{np.mean([r.objective_time for r in runs]):.2f}",
                        f"{np.mean([r.overhead_time for r in runs]):.2f}"])
    print(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n-init", type=int, default=10, help="shared initial design size")
    ap.add_argument("--n-iter", type=int, default=50, help="evaluations after the initial design")
    ap.add_argument("--n-restarts", type=int, default=10, help="restarts of the acquisition optimization")
    ap.add_argument("--n-runs", type=int, default=3,
                    help="number of repetitions; each draws one random seed shared by sbo and scikit-optimize")
    ap.add_argument("--master-seed", type=int, default=None,
                    help="optional: makes the drawn seeds (hence the whole study) reproducible")
    ap.add_argument("--plot-only", action="store_true", help="re-plot the saved results")
    ap.add_argument("--show", action="store_true", help="also display the figure")
    args = ap.parse_args()

    out = RESULTS / f"n_init{args.n_init}_n_iter{args.n_iter}_n_runs{args.n_runs}"
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "comparison.npz"

    if args.plot_only:
        results, baseline = load_results(npz)
    else:
        env = BurgersControl()
        baseline = env.cost(np.zeros(3))            # cost without control
        results = run_all(env, BOUNDS, args.n_init, args.n_iter, args.n_restarts,
                           args.n_runs, args.master_seed)
        save_results(results, baseline, npz)

    write_summary(results, out / "summary.csv")
    plot_comparison(results, out / "comparison.png", baseline_cost=baseline, show=args.show)
    plot_per_run(results, out / "comparison_per_run.png", baseline_cost=baseline, show=args.show)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
