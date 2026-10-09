"""
Solve the Kolmogorov control case with sbo (without and with the rank-one
update) and with scikit-optimize (same GP, same hyperparameter optimization
rate, same normalization), time them and plot the comparison.

    python run_comparison.py                       # 1 run: 50 initial points + 1450 evaluations (~3 hours)
    python run_comparison.py --n-runs 3 --master-seed 0    # more runs, reproducible seeds
    python run_comparison.py --methods sbo         # one method only (the others can be added later)
    python run_comparison.py --n-iter 100 --rank-one-threshold 60   # quick test
    python run_comparison.py --plot-only           # re-plot (same options as the run)

Every finished optimization is saved at once (runs/), and a re-launch only
computes what is missing: a long study can be stopped and resumed.

Outputs (in results/<objective>_d<..>_n_init<..>_n_iter<..>_n_runs<..>/): runs/ (every evaluation
and the timings of every run), study.json (settings, seeds), summary.csv (one
row per method), comparison.png (mean +/- std), overhead.png (optimizer time
per iteration), comparison_per_run.png.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from kolmogorov_case import KolmogorovControl, HERE
from solvers import METHODS, HPO_EVERY, RANK_ONE_THRESHOLD, RunResult, run_all
from plot_comparison import plot_comparison, plot_overhead, plot_per_run

RESULTS = HERE / "results"


def run_file(runs_dir: Path, i: int, method: str) -> Path:
    key = method.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    return runs_dir / f"run{i + 1}_{key}.npz"


def save_run(r: RunResult, path: Path) -> None:
    np.savez(path, name=r.name, seed=r.seed, X=r.X, y=r.y, n_init=r.n_init,
             times=[r.total_time, r.objective_time], iter_time=r.iter_time, fit_time=r.fit_time)


def load_run(path: Path) -> RunResult:
    d = np.load(path)
    return RunResult(str(d["name"]), int(d["seed"]), d["X"], d["y"], int(d["n_init"]),
                     float(d["times"][0]), float(d["times"][1]), d["iter_time"], d["fit_time"])


def load_done(runs_dir: Path, n_runs: int) -> dict:
    """{(run index, method): RunResult} of the runs already saved."""
    return {(i, m): load_run(run_file(runs_dir, i, m))
            for i in range(n_runs) for m in METHODS if run_file(runs_dir, i, m).exists()}


def write_summary(results: Dict[str, List[RunResult]], path: Path) -> None:
    header = ["method", "n_runs", "best_cost_mean", "best_cost_std", "best_cost_min", "best_cost_max",
              "time_total_mean_s", "time_total_std_s", "time_objective_mean_s", "time_optimizer_mean_s",
              "time_gp_update_mean_s", "time_acquisition_mean_s"]
    sd = lambda a: np.std(a, ddof=1) if len(a) > 1 else 0.0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for m, runs in results.items():
            best = np.array([r.best_y for r in runs])
            total = np.array([r.total_time for r in runs])
            fit = np.mean([r.fit_time.sum() for r in runs])
            w.writerow([m, len(runs), f"{best.mean():.4f}", f"{sd(best):.4f}",
                        f"{best.min():.4f}", f"{best.max():.4f}",
                        f"{total.mean():.1f}", f"{sd(total):.1f}",
                        f"{np.mean([r.objective_time for r in runs]):.1f}",
                        f"{np.mean([r.overhead_time for r in runs]):.1f}",
                        f"{fit:.1f}", f"{np.mean([r.iter_time.sum() for r in runs]) - fit:.1f}"])
    print(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n-init", type=int, default=50, help="shared initial design size")
    ap.add_argument("--n-iter", type=int, default=1450, help="evaluations after the initial design")
    ap.add_argument("--n-restarts", type=int, default=3, help="restarts of the acquisition optimization")
    ap.add_argument("--n-runs", type=int, default=1,
                    help="number of repetitions; each draws one random seed shared by all the methods")
    ap.add_argument("--master-seed", type=int, default=None,
                    help="optional: makes the drawn seeds (hence the whole study) reproducible")
    ap.add_argument("--hpo-every", type=int, default=HPO_EVERY,
                    help="hyperparameters re-optimized every this many iterations (all methods)")
    ap.add_argument("--rank-one-threshold", type=int, default=RANK_ONE_THRESHOLD,
                    help="number of points above which sbo's rank-one update acts (sbo's default)")
    ap.add_argument("--objective", choices=("tracking", "energy"), default="tracking",
                    help="tracking: reach a target flow; energy: minus the return of the environment")
    ap.add_argument("--n-segments", type=int, default=3, help="segments of the forcing schedule")
    ap.add_argument("--wavenumbers", type=int, nargs="+", default=[1, 2, 3, 4], metavar="K",
                    help="forcing modes sin(k y), one amplitude each per segment (HydroGym's: 4 5 6 7)")
    ap.add_argument("--both-phases", action=argparse.BooleanOptionalAction, default=True,
                    help="also force the modes cos(k y): twice as many unknowns")
    ap.add_argument("--alpha", type=float, default=200.0,
                    help="energy objective: weight of the kinetic energy against the actuation penalty")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS), metavar="METHOD",
                    help=f"methods to run now, among {METHODS}")
    ap.add_argument("--plot-only", action="store_true", help="re-plot the saved results")
    ap.add_argument("--show", action="store_true", help="also display the figures")
    args = ap.parse_args()

    dim = args.n_segments * len(args.wavenumbers) * (2 if args.both_phases else 1)
    out = RESULTS / f"{args.objective}_d{dim}_n_init{args.n_init}_n_iter{args.n_iter}_n_runs{args.n_runs}"
    runs_dir, study_file = out / "runs", out / "study.json"
    runs_dir.mkdir(parents=True, exist_ok=True)
    settings = {k: getattr(args, k) for k in ("n_init", "n_iter", "n_restarts", "hpo_every",
                                              "rank_one_threshold", "objective", "n_segments",
                                              "wavenumbers", "both_phases", "alpha")}

    if args.plot_only:
        study = json.loads(study_file.read_text())
    else:
        if study_file.exists():                      # resume: same settings, same seeds
            study = json.loads(study_file.read_text())
            if study["settings"] != settings:
                raise SystemExit(f"{out} holds a study with other settings:\n  saved     {study['settings']}\n"
                                 f"  requested {settings}\nUse the same ones, or move/delete that folder.")
        else:
            seeds = np.random.default_rng(args.master_seed).integers(0, 2**31 - 1, size=args.n_runs)
            study = dict(settings=settings, seeds=seeds.tolist())
        problem = KolmogorovControl(objective=args.objective, n_segments=args.n_segments, alpha=args.alpha,
                                    wavenumbers=tuple(args.wavenumbers), both_phases=args.both_phases)
        study["baseline_cost"] = problem.cost(np.zeros(problem.dim))      # cost without control
        study_file.write_text(json.dumps(study, indent=2))
        print(f"Kolmogorov control ({args.objective}), {problem.dim} unknowns: "
              f"cost without control = {study['baseline_cost']:.4f}, "
              f"{problem.eval_time:.2f} s per evaluation")
        run_all(problem, problem.bounds, args.n_init, args.n_iter, args.n_restarts, study["seeds"],
                methods=args.methods, hpo_every=args.hpo_every, rank_one_threshold=args.rank_one_threshold,
                done=load_done(runs_dir, args.n_runs),
                on_run=lambda i, r: save_run(r, run_file(runs_dir, i, r.name)))

    # methods with all their runs available, in the order of METHODS
    done = load_done(runs_dir, args.n_runs)
    results = {m: [done[(i, m)] for i in range(args.n_runs)]
               for m in METHODS if all((i, m) in done for i in range(args.n_runs))}
    if not results:
        raise SystemExit(f"No complete method in {runs_dir}")
    cfg = study["settings"]
    # tracking: the cost goes to 0, log scale; energy: small changes of a large cost, shown relative to no control
    kw = dict(baseline_cost=study["baseline_cost"], relative=cfg["objective"] == "energy", show=args.show)

    write_summary(results, out / "summary.csv")
    plot_comparison(results, out / "comparison.png", **kw)
    plot_overhead(results, out / "overhead.png", hpo_every=cfg["hpo_every"],
                  rank_one_threshold=cfg["rank_one_threshold"], show=args.show)
    plot_per_run(results, out / "comparison_per_run.png", **kw)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
