# -*- coding: utf-8 -*-
"""
Post-processing: build a GIF / MP4 animation from the per-iteration .npz
snapshots exported by BO_ML4F when export_states=True.

Each frame shows:
  Top panel  — GP posterior mean ± 2σ, observations, and the newly proposed point.
  Bottom panel — acquisition function values and the argmax (next query).

Usage
-----
  python make_animation_1D.py                   # default paths
  python make_animation_1D.py ./out/states ./out/animation.gif 3

Arguments (positional, all optional)
  1. states_dir   : directory containing state_NNN.npz files  [./out/states]
  2. output_file  : output filename (.gif or .mp4)             [./out/animation.gif]
  3. fps          : frames per second                          [2]

Requirements
------------
  pip install matplotlib pillow          # for GIF output
  pip install matplotlib ffmpeg-python   # for MP4 output (needs ffmpeg installed)

@author: mendez, lecomte
"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ------------------------------------------------------------------ #
#  Parse arguments
# ------------------------------------------------------------------ #
states_dir   = sys.argv[1] if len(sys.argv) > 1 else "./out/states"
output_file  = sys.argv[2] if len(sys.argv) > 2 else "./out/animation.gif"
fps          = int(sys.argv[3]) if len(sys.argv) > 3 else 2

# ------------------------------------------------------------------ #
#  Load state files
# ------------------------------------------------------------------ #
files = sorted(glob.glob(os.path.join(states_dir, "state_*.npz")))
if not files:
    raise FileNotFoundError(
        f"No state_NNN.npz files found in '{states_dir}'.\n"
        "Run the BO with export_states=True first."
    )

print(f"Found {len(files)} snapshots in '{states_dir}'.")

# Check the first file to confirm 1D GP data is present
_check = np.load(files[0])
if 'Xplot' not in _check:
    raise ValueError(
        "State files do not contain 'Xplot' / 'mu' / 'std' arrays.\n"
        "These are only exported for 1D problems (d == 1)."
    )

# ------------------------------------------------------------------ #
#  Plot style
# ------------------------------------------------------------------ #
plt.rc('text', usetex=False)      # set True if LaTeX is available
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.rc('axes',  labelsize=12)

# ------------------------------------------------------------------ #
#  Build animation
# ------------------------------------------------------------------ #
fig, axs = plt.subplots(
    2, 1, figsize=(7, 6),
    constrained_layout=True,
    sharex=True,
    gridspec_kw=dict(height_ratios=[2, 1]),
)

def update(frame_idx):
    data = np.load(files[frame_idx])

    it     = int(data['it'][0])
    X      = data['X'].reshape(-1)
    y      = data['y'].reshape(-1)
    Xplot  = data['Xplot']
    mu     = data['mu']
    std    = data['std']
    Xcand  = data['Xcand'].reshape(-1)
    a      = data['a'].reshape(-1)
    x_next = float(data['x_next'][0])
    y_next = float(data['y_next'][0])
    a_best = float(data['a_best'][0])

    for ax in axs:
        ax.cla()

    # ---- Top: GP posterior
    axs[0].plot(Xplot, mu, 'C0', lw=2, label=r'$\mu_{\mathcal{GP}}$')
    axs[0].fill_between(
        Xplot, mu - 2 * std, mu + 2 * std,
        color='C0', alpha=0.25, label=r'$\mu \pm 2\sigma$',
    )
    axs[0].scatter(X, y, c='k', s=18, zorder=10, label='Observations')
    axs[0].axvline(x_next, color='C3', lw=1.2, ls='--', label=f'$x_{{next}}={x_next:.3f}$')
    axs[0].scatter([x_next], [y_next], c='C3', s=60, zorder=11, marker='*')
    axs[0].set_ylabel('f(x)')
    axs[0].set_title(f'Iteration {it}   |   n = {len(X)}   |   best y = {min(y):.4f}')
    axs[0].legend(fontsize=8, loc='upper right', ncol=2)

    # ---- Bottom: acquisition
    idx = np.argsort(Xcand)
    axs[1].plot(Xcand[idx], a[idx], 'C1', lw=1.5, label='Acquisition')
    axs[1].fill_between(Xcand[idx], a[idx], color='C1', alpha=0.2)
    axs[1].scatter([x_next], [a_best], c='C3', s=60, zorder=10, marker='*', label='Next query')
    axs[1].set_ylabel('acq(x)')
    axs[1].set_xlabel('x')
    axs[1].legend(fontsize=8, loc='lower right')

n_frames = len(files)
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)

# ------------------------------------------------------------------ #
#  Save
# ------------------------------------------------------------------ #
ext = os.path.splitext(output_file)[1].lower()
os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

if ext == ".gif":
    ani.save(output_file, writer='pillow', fps=fps)
elif ext in (".mp4", ".avi", ".mov"):
    ani.save(output_file, writer='ffmpeg', fps=fps,
             extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
else:
    raise ValueError(f"Unsupported output format '{ext}'. Use .gif or .mp4.")

plt.close(fig)
print(f"Animation saved → {output_file}  ({n_frames} frames @ {fps} fps)")
