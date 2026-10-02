"""
Presentation material for the Burgers control case: space-time maps and GIFs
of the uncontrolled system and of a controlled one.

    python presentation.py                      # maps only, weights = best found by run_comparison.py
    python presentation.py --gifs               # also write the GIFs (slower)
    python presentation.py --weights 0.05 -0.02 0.01 --gifs

Outputs go to assets/. (The original, unrefactored script is kept in legacy/.)

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from burgers_case import (EVAL_WINDOW, HERE, SENSORS, X_ACTUATION, X_PERTURBATION,
                          BurgersControl)

ASSETS = HERE / "assets"
RESULTS = HERE / "results"
FRAME_EVERY = 5      # one GIF frame every 5 time steps

plt.rc("font", family="serif")


def simulate(problem: BurgersControl, w: np.ndarray):
    """One episode: returns u(t, x), perturbation(t, x), action profile(t, x), a(t)."""
    env, n = problem.env, problem.n_steps
    obs = env.reset()
    u = np.zeros((n, env.Nx + 1))
    pert = np.zeros_like(u)
    act = np.zeros_like(u)
    a = np.zeros(n)
    for k in range(n):
        a[k] = float(w @ obs)
        obs, _, _, _ = env.step(a[k])
        u[k], pert[k], act[k] = env.u, env.pert, env.action_vec
    return u, pert, act, a


def plot_map(env, u: np.ndarray, title: str, path: Path) -> None:
    """Space-time contour map with sensors (black), evaluation window (white),
    perturbation (red, solid) and actuator (red, dotted) positions."""
    t = np.linspace(0, env.T, u.shape[0])
    fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
    cs = ax.contourf(env.x, t, u, levels=30, alpha=0.8)
    for i in SENSORS:
        ax.axvline(env.x[i], color="k", ls="--", lw=1)
    for i in EVAL_WINDOW:
        ax.axvline(env.x[i], color="w", ls="-.", lw=1)
    ax.axvline(X_PERTURBATION, color="r", ls="-", lw=1.2)
    ax.axvline(X_ACTUATION, color="r", ls=":", lw=1.2)
    ax.set(xlabel="x", ylabel="t", title=title)
    fig.colorbar(cs, ax=ax, label="u")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _frame(env, u, pert, act, a, k) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(env.x, u)
    ax.plot(env.x[list(SENSORS)], u[list(SENSORS)], "ro", label="sensors")
    ax.plot(env.x[list(EVAL_WINDOW)], u[list(EVAL_WINDOW)], "go", label="evaluation window")
    ref = abs(u[330]) or 1.0
    if np.abs(pert).max() > 0:
        ax.fill(env.x, pert / np.abs(pert).max() * ref, alpha=0.4, label="perturbation")
    if np.abs(act).max() > 0:
        ax.fill(env.x, act / np.abs(act).max() * abs(u[660] or 1.0), alpha=0.4, label="control")
    ax.set(xlabel="x", ylabel="u", title=f"a(t) = {a:+.3f}    t = {k * env.dt:.2f} s")
    ax.legend(loc="upper right", fontsize=8)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def make_gif(env, u_t, pert_t, act_t, a_t, path: Path) -> None:
    frames = [_frame(env, u_t[k], pert_t[k], act_t[k], a_t[k], k)
              for k in range(0, u_t.shape[0], FRAME_EVERY)]
    imageio.mimsave(path, frames, duration=0.15)


def best_weights_from_results(results_dir: Path | None = None) -> np.ndarray:
    """Best weights of `results_dir`/comparison.npz (default: the most recent run in results/)."""
    if results_dir is None:
        results_dir = max(RESULTS.glob("*/comparison.npz"), key=lambda p: p.stat().st_mtime).parent
    print(f"Reading {results_dir}")
    data = np.load(results_dir / "comparison.npz")
    best, w = np.inf, None
    for key in data.files:
        if key.endswith("__y"):
            y, X = data[key], data[key.replace("__y", "__X")]
            i = np.unravel_index(np.argmin(y), y.shape)
            if y[i] < best:
                best, w = y[i], X[i]
    return np.asarray(w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--weights", type=float, nargs=3, default=None,
                    help="feedback weights; default: best found by run_comparison.py")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="results sub-folder to read the best weights from; default: the most recent one")
    ap.add_argument("--gifs", action="store_true", help="also write the GIFs")
    args = ap.parse_args()

    w = np.array(args.weights) if args.weights else best_weights_from_results(args.results_dir)
    print(f"Controlled case: w = {np.round(w, 4)}")

    ASSETS.mkdir(exist_ok=True)
    problem = BurgersControl()
    cases = {"uncontrolled": np.zeros(3), "controlled": w}
    for name, weights in cases.items():
        u, pert, act, a = simulate(problem, weights)
        cost = problem.cost(weights)
        plot_map(problem.env, u, f"{name}, cost = {cost:.0f}", ASSETS / f"map_{name}.png")
        if args.gifs:
            print(f"GIF {name} ...")
            make_gif(problem.env, u, pert, act, a, ASSETS / f"gif_{name}.gif")
    print(f"Saved to {ASSETS}")


if __name__ == "__main__":
    main()
