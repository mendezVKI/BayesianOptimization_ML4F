"""
Experiment folder setup and run logging for sbo.

Split out of core.py so that the compute core (config/GP/acquisition/BO
loop) has no dependency on logging beyond what it needs internally. These
functions only touch their arguments through duck-typed attribute access, so
this module has no runtime dependency on core.py at all (the type hints
below are only evaluated by type checkers, thanks to
``from __future__ import annotations``).

Reproducibility checkpointing (Tier 1 trace / Tier 2 GP snapshots) lives in
persistence.py, not here.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, Optional, Tuple
from datetime import datetime, timedelta
import os
import logging
import re

import numpy as np

if TYPE_CHECKING:
    from .core import Array, SaveConfig

_RUN_DIR_RE = re.compile(r"^run_(\d+)$")

RUN_NAMINGS = ("timestamp", "run_id", "params", "custom")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _next_run_name(base_path: str) -> str:
    """'run_<n>', n = 1 + the highest existing run_<n> subfolder of
    base_path (run_1 if none exist yet). Used instead of a timestamp when
    save_cfg.run_naming == "run_id", so two quick runs with the same
    out_path still land in separate folders.

    Not race-safe across concurrent processes (two runs could list the same
    existing folders and pick the same next number); the subsequent
    os.makedirs(..., exist_ok=False) will raise in that case rather than
    silently overwrite.
    """
    existing = []
    if os.path.isdir(base_path):
        for name in os.listdir(base_path):
            m = _RUN_DIR_RE.match(name)
            if m and os.path.isdir(os.path.join(base_path, name)):
                existing.append(int(m.group(1)))
    return f"run_{(max(existing) + 1) if existing else 1}"


def _params_run_name(name_params: Optional[Dict[str, object]]) -> str:
    """'ninit_5_niter_10_xi_0.01' from {'ninit': 5, 'niter': 10, 'xi': 0.01}.
    Floats use the shortest general format (0.01, 1e-05), and anything not
    filesystem-safe is replaced by '-'."""
    if not name_params:
        raise ValueError(
            'run_naming="params" needs the parameters to put in the name '
            "(bayesian_optimization passes n_init, n_iter and xi)."
        )
    parts = []
    for key, value in name_params.items():
        text = f"{value:g}" if isinstance(value, float) else str(value)
        parts.append(f"{key}_{_SAFE_NAME_RE.sub('-', text)}")
    return "_".join(parts)


def _custom_run_name(run_name: Optional[str]) -> str:
    if not run_name or not run_name.strip():
        raise ValueError('run_naming="custom" needs a non-empty save_cfg.run_name.')
    if run_name in (".", "..") or any(sep in run_name for sep in ("/", "\\", os.sep)):
        raise ValueError(
            f"save_cfg.run_name={run_name!r} must be a plain folder name (no path separators): "
            "the run folder is always created directly under out_path."
        )
    return run_name


def setup_experiment_folder(
    save_cfg: "SaveConfig", name_params: Optional[Dict[str, object]] = None
) -> Tuple[Optional[str], "SaveConfig"]:
    """Resolve save_cfg.out_path into a run folder and create it on disk.

    Exactly one run-level folder is always created under out_path, never
    nested in a previous run's folder. Its name is chosen by
    save_cfg.run_naming:

        "timestamp"  2026-10-02_14-31-07                 (default)
        "run_id"     run_<n>, n = 1 + the highest existing run_<n>
        "params"     ninit_5_niter_10_xi_0.01, built from name_params
        "custom"     save_cfg.run_name

    A folder is never overwritten: if the chosen one already exists
    (two "timestamp" runs in the same second, the same "params" or "custom"
    name twice) a FileExistsError says so. "run_id" cannot collide.

    Returns (resolved_path, resolved_save_cfg): resolved_save_cfg is a COPY
    of save_cfg with out_path set to the run folder. The SaveConfig object
    the caller passed in is never mutated, so the same SaveConfig can be
    reused across multiple bayesian_optimization() calls.
    """
    if save_cfg.out_path is None:
        return None, save_cfg

    os.makedirs(save_cfg.out_path, exist_ok=True)

    naming = save_cfg.run_naming
    if naming == "timestamp":
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    elif naming == "run_id":
        run_name = _next_run_name(save_cfg.out_path)
    elif naming == "params":
        run_name = _params_run_name(name_params)
    elif naming == "custom":
        run_name = _custom_run_name(save_cfg.run_name)
    else:
        raise ValueError(f"save_cfg.run_naming must be one of {RUN_NAMINGS}, got {naming!r}.")

    exp_path = os.path.join(save_cfg.out_path, run_name)
    try:
        os.makedirs(exp_path, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"Run folder {exp_path} already exists (run_naming={naming!r}); it is never overwritten. "
            'Change the parameters / run_name, or use run_naming="run_id" or "timestamp".'
        ) from None

    resolved_cfg = replace(save_cfg, out_path=exp_path)  # new object, not save_cfg.out_path = exp_path
    return exp_path, resolved_cfg


def setup_logger(log_file_path: str) -> logging.Logger:
    """Set up the run logger writing directly to log_file_path (e.g.
    '<run folder>/log.log') -- a single file, not a directory."""
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger("BO")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file_path, mode="w")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def _format_duration(seconds: float) -> str:
    """'12.34 s', or '1234.56 s (0:20:34)' once it exceeds a minute."""
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    return f"{seconds:.2f} s ({str(timedelta(seconds=round(seconds)))})"


def log_summary(logger, best_x: "Array", best_y: float, n_evals: int, total_time: float) -> None:
    """One-line run summary, written once when the BO run finishes.

    Per-iteration values (x_next, y_next, best_y, l_c, sigma_f, sigma_y, ...)
    are no longer duplicated here -- they live in the Tier 1 trace
    (persistence.TraceLog: trace.npz for reloading, trace.csv to read by
    eye). The full configuration lives in meta.json (persistence.write_meta).
    So log.log stays lightweight run metadata: start/end markers only.
    """
    logger.info(
        f"Finished: {n_evals} evaluations | best_y = {best_y:.6e} | best_x = {np.asarray(best_x).tolist()} | "
        f"total time = {_format_duration(total_time)}"
    )


_FRAME_RE = re.compile(r"^it_(\d+)\.png$")


def make_gif(
    run_folder: str,
    frames_subdir: str = "GIF",
    out_name: str = "evolution.gif",
    fps: float = 2.0,
    last_frame_hold: float = 1.0,
    loop: int = 0,
) -> str:
    """Assemble the per-iteration state plots of a run into one animated GIF.

    Reads ``<run_folder>/plots/<frames_subdir>/it_XXX.png`` (written when
    ``SaveConfig.plt_state_enabled`` or ``plt_all`` is on) in iteration order
    and writes ``<run_folder>/plots/<out_name>``. ``run_folder`` is the resolved
    run folder, i.e. ``BOResult.out_path``.

    fps              frames per second.
    last_frame_hold  extra seconds the last frame stays on screen.
    loop             0 = loop forever, n = play n times.

    Frames saved with ``bbox_inches="tight"`` can differ by a few pixels; they
    are centred on a white canvas of the largest size so the GIF does not
    jitter. Uses Pillow, which ships with matplotlib (no extra dependency).
    Returns the path of the GIF.
    """
    from PIL import Image

    frames_dir = os.path.join(run_folder, "plots", frames_subdir)
    frames = []
    if os.path.isdir(frames_dir):
        for name in os.listdir(frames_dir):
            m = _FRAME_RE.match(name)
            if m:
                frames.append((int(m.group(1)), os.path.join(frames_dir, name)))
    frames.sort()

    if not frames:
        raise FileNotFoundError(
            f"No it_XXX.png frames in {frames_dir}. State plots are only saved when "
            f"SaveConfig.plt_state_enabled (or plt_all) is True and out_path is set."
        )

    images = [Image.open(path).convert("RGB") for _, path in frames]
    width = max(im.width for im in images)
    height = max(im.height for im in images)
    uniform = []
    for im in images:
        if im.size == (width, height):
            uniform.append(im)
        else:
            canvas = Image.new("RGB", (width, height), "white")
            canvas.paste(im, ((width - im.width) // 2, (height - im.height) // 2))
            uniform.append(canvas)

    frame_ms = int(round(1000.0 / fps))
    durations = [frame_ms] * len(uniform)
    durations[-1] += int(round(1000.0 * last_frame_hold))

    out_path = os.path.join(run_folder, "plots", out_name)
    uniform[0].save(
        out_path, save_all=True, append_images=uniform[1:],
        duration=durations, loop=loop, optimize=False,
    )
    return out_path
