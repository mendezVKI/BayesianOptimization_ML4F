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
from typing import TYPE_CHECKING, Optional
from datetime import datetime
import os
import logging

import numpy as np

if TYPE_CHECKING:
    from .core import BOState


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

    logger = logging.getLogger("BO")
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


def log_update(logger, state: "BOState") -> None:

    # clean the format of n_next
    x_scalar = float(state.x_next[0]) if np.ndim(state.x_next) > 0 else float(state.x_next)

    n_proposed = 0 if state.x_proposed is None else len(state.x_proposed)
    n_added = state.n_added if state.n_added is not None else 1

    logger.info(
        f"{state.it:03d} | "
        f"{x_scalar:>14.6e} | "
        f"{state.y_next:>14.6e} | "
        f"{state.y_best:>14.6e} | "
        f"{state.gp.l_c:>12.3e} | "
        f"{state.gp.sigma_f:>12.3e} | "
        f"{state.gp.sigma_y:>12.3e} | "
        f"n_prop={n_proposed:>2d} | n_add={n_added:>2d}"
    )
