"""
Experiment folder setup and run logging for mobo.

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
from typing import TYPE_CHECKING, Optional
from datetime import datetime
import os
import logging

import numpy as np

if TYPE_CHECKING:
    from .core import MOBOState, SaveConfig


def setup_experiment_folder(save_cfg: "SaveConfig") -> Optional[str]:
    if save_cfg.out_path is None:
        return None

    if save_cfg.create_timestamp:
        os.makedirs(save_cfg.out_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_path = os.path.join(save_cfg.out_path, timestamp)
        os.makedirs(exp_path, exist_ok=False)
        save_cfg.out_path = exp_path
    else:
        os.makedirs(save_cfg.out_path, exist_ok=True)
        exp_path = save_cfg.out_path

    return exp_path


def setup_logger(log_path: str, filename: str = "run.log") -> logging.Logger:
    os.makedirs(log_path, exist_ok=True)

    logger = logging.getLogger("MOBO")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(
        os.path.join(log_path, filename),
        mode="w"
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def log_update(logger, state: "MOBOState") -> None:
    """One line per iteration.

    A multi-objective run has no scalar incumbent to report, so the progress
    columns are the dominated hypervolume and the current front size; the
    whole objective vector is logged instead of a single y."""

    x_scalar = float(state.x_next[0]) if np.ndim(state.x_next) > 0 else float(state.x_next)
    y_str = " ".join(f"{v:>13.6e}" for v in np.atleast_1d(state.y_next))

    logger.info(
        f"{state.it:03d} | "
        f"{x_scalar:>14.6e} | "
        f"{y_str} | "
        f"{state.hypervolume:>13.6e} | "
        f"{state.pareto_Y.shape[0]:>5d} | "
        f"{float(np.atleast_1d(state.gp.length_scales)[0]):>10.3e} | "
        f"{state.acq_res.a_best:>10.3e}"
    )
