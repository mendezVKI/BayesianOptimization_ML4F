"""
Experiment folder setup and run logging for mfbo.

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
from typing import TYPE_CHECKING, Optional, Tuple
from datetime import datetime
import os
import logging
import re

import numpy as np

if TYPE_CHECKING:
    from .core import Array, SaveConfig

_RUN_DIR_RE = re.compile(r"^run_(\d+)$")


def _next_run_name(base_path: str) -> str:
    """'run_<n>', n = 1 + the highest existing run_<n> subfolder of
    base_path (run_1 if none exist yet). Used instead of a timestamp when
    save_cfg.create_timestamp is False, so two quick runs with the same
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


def setup_experiment_folder(save_cfg: "SaveConfig") -> Tuple[Optional[str], "SaveConfig"]:
    """Resolve save_cfg.out_path into a run folder and create it on disk.

    Exactly one run-level folder is always created under out_path -- named
    by timestamp (save_cfg.create_timestamp=True, the default) or by
    auto-incrementing run_<n> (create_timestamp=False) -- so that two runs
    pointed at the same out_path never collide or nest into each other,
    whether or not timestamps are used.

    Returns (resolved_path, resolved_save_cfg): resolved_save_cfg is a COPY
    of save_cfg with out_path set to the run folder. The SaveConfig object
    the caller passed in is never mutated, so the same SaveConfig can be
    reused across multiple bayesian_optimization() calls.
    """
    if save_cfg.out_path is None:
        return None, save_cfg

    os.makedirs(save_cfg.out_path, exist_ok=True)

    if save_cfg.create_timestamp:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        run_name = _next_run_name(save_cfg.out_path)

    exp_path = os.path.join(save_cfg.out_path, run_name)
    os.makedirs(exp_path, exist_ok=False)

    resolved_cfg = replace(save_cfg, out_path=exp_path)  # new object, not save_cfg.out_path = exp_path
    return exp_path, resolved_cfg


def setup_logger(log_file_path: str) -> logging.Logger:
    """Set up the run logger writing directly to log_file_path (e.g.
    '<run folder>/log.log') -- a single file, not a directory."""
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger("MFBO")
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


def log_summary(logger, best_x: "Array", best_y: float, n_low: int, n_high: int, total_cost: float) -> None:
    """One-line run summary, written once when the MFBO run finishes.

    Per-iteration values (fidelity level, x_next, y_next, best_y, GP
    hyperparameters, cost, cumulative cost, ...) live in the Tier 1 trace
    (persistence.TraceLog: trace.npz for reloading, trace.csv to read by
    eye). The full configuration lives in meta.json (persistence.write_meta).
    So log.log stays lightweight run metadata: start/end markers only.
    """
    logger.info(
        f"Finished: {n_low} low-fidelity + {n_high} high-fidelity evaluations | "
        f"total cost = {total_cost:.6e} | best_y (high fidelity) = {best_y:.6e} | "
        f"best_x = {np.asarray(best_x).tolist()}"
    )
