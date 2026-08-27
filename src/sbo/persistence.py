"""
Two-tier persistence for sbo Bayesian Optimization runs.

Tier 1 ("trace"): a small, fixed-shape-per-iteration tabular record (proposed
x, observed y, incumbent best x/y, wall-clock time, acquisition value at the
proposed point). Always recorded in memory (cheap: only scalars / (d,)
vectors, no model objects); written to disk as a single ``trace.npz`` file
whenever ``SaveConfig.out_path`` is set.

Tier 2 ("snapshots"): opt-in, heavier per-iteration records of everything
needed to *refit* the GP exactly -- the physical (X, y) design, fitted
hyperparameters, kernel identity, and normalization state. Dense covariance /
Cholesky factors are deliberately never stored: ``reconstruct_gp`` recomputes
them from (X, y, hyperparameters) via the same ``gp_fit`` routine used live,
so predictions reproduce bit-for-bit.

Storage format: numpy ``.npz`` (binary, lossless float64, no pickle -- every
array here is numeric or fixed-width unicode, so ``np.load`` needs no
``allow_pickle``) for the numeric payloads, plus one small ``meta.json`` for
the human-readable run configuration. ``.npz`` cannot be appended to in
place, so Tier 2 snapshots are written as one file per (flushed batch of)
iteration(s) rather than one growing archive -- this keeps periodic flushing
O(1) per flush instead of O(n) from rewriting everything each time.

This module imports concrete objects (GPModel, gp_fit, ...) from .core, so
core.py must import this module lazily (inside functions, not at module
scope) to avoid a circular import -- the same pattern already used for
plotting.py.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import glob
import json
import os
import re

import numpy as np

from .core import (
    Array,
    GPModel,
    NormalizationHelper,
    gp_fit,
    gp_predict,
)

_KERNEL_ID = "rbf_amp"  # the only kernel currently implemented in core.py

_SNAPSHOT_RE = re.compile(r"it_(\d+)\.npz$")


#%% ---------------------------------------------------------------------
# Tier 1: lightweight per-iteration trace
# -------------------------------------------------------------------------

@dataclass
class TraceLog:
    """Tier 1 record: one row per BO iteration, all scalars/(d,) vectors."""

    it: List[int] = field(default_factory=list)
    x_next: List[Array] = field(default_factory=list)
    y_next: List[float] = field(default_factory=list)
    x_best: List[Array] = field(default_factory=list)
    y_best: List[float] = field(default_factory=list)
    wall_time: List[float] = field(default_factory=list)
    acq_value: List[float] = field(default_factory=list)

    def append(
        self,
        it: int,
        x_next: Array,
        y_next: float,
        x_best: Array,
        y_best: float,
        wall_time: float,
        acq_value: float,
    ) -> None:
        self.it.append(int(it))
        self.x_next.append(np.asarray(x_next, dtype=float).copy())
        self.y_next.append(float(y_next))
        self.x_best.append(np.asarray(x_best, dtype=float).copy())
        self.y_best.append(float(y_best))
        self.wall_time.append(float(wall_time))
        self.acq_value.append(float(acq_value))

    def __len__(self) -> int:
        return len(self.it)

    def to_arrays(self) -> Dict[str, Array]:
        return dict(
            it=np.asarray(self.it, dtype=int),
            x_next=np.asarray(self.x_next, dtype=float),
            y_next=np.asarray(self.y_next, dtype=float),
            x_best=np.asarray(self.x_best, dtype=float),
            y_best=np.asarray(self.y_best, dtype=float),
            wall_time=np.asarray(self.wall_time, dtype=float),
            acq_value=np.asarray(self.acq_value, dtype=float),
        )

    def save(self, path: str) -> None:
        np.savez(path, **self.to_arrays())


def load_trace(path: str) -> TraceLog:
    data = np.load(path)
    trace = TraceLog()
    trace.it = data["it"].astype(int).tolist()
    trace.x_next = list(np.atleast_2d(data["x_next"]))
    trace.y_next = data["y_next"].astype(float).tolist()
    trace.x_best = list(np.atleast_2d(data["x_best"]))
    trace.y_best = data["y_best"].astype(float).tolist()
    trace.wall_time = data["wall_time"].astype(float).tolist()
    trace.acq_value = data["acq_value"].astype(float).tolist()
    return trace


#%% ---------------------------------------------------------------------
# Tier 2: full GP-reconstruction snapshots
# -------------------------------------------------------------------------

@dataclass
class GPSnapshot:
    """Everything needed to refit the GP exactly at one iteration.

    Deliberately excludes alpha/L (dense Cholesky factor): those are
    recomputed by ``reconstruct_gp`` via the same ``gp_fit`` used live.
    """

    it: int
    X: Array          # (n_k, d) physical design
    y: Array          # (n_k,) physical observations
    l_c: float
    sigma_f: float
    sigma_y: float
    jitter: float
    kernel: str
    x_lo: Array
    x_hi: Array
    y_mean: float
    y_std: float
    eps: float


def build_gp_snapshot(it: int, gp: GPModel) -> GPSnapshot:
    if gp.Xs is None or gp.ys is None or gp.normalizer is None:
        raise ValueError("GP has no fitted data / normalizer yet; cannot snapshot.")

    normalizer = gp.normalizer
    return GPSnapshot(
        it=int(it),
        X=np.asarray(gp.Xs, dtype=float).copy(),
        y=np.asarray(gp.ys, dtype=float).copy(),
        l_c=float(gp.l_c),
        sigma_f=float(gp.sigma_f),
        sigma_y=float(gp.sigma_y),
        jitter=float(gp.jitter),
        kernel=_KERNEL_ID,
        x_lo=np.asarray(normalizer.x_lo, dtype=float).copy(),
        x_hi=np.asarray(normalizer.x_hi, dtype=float).copy(),
        y_mean=float(normalizer.y_mean),
        y_std=float(normalizer.y_std),
        eps=float(normalizer.eps),
    )


def _snapshot_arrays(snap: GPSnapshot) -> Dict[str, Array]:
    return dict(
        it=np.asarray(snap.it),
        X=np.asarray(snap.X, dtype=float),
        y=np.asarray(snap.y, dtype=float),
        l_c=np.asarray(snap.l_c),
        sigma_f=np.asarray(snap.sigma_f),
        sigma_y=np.asarray(snap.sigma_y),
        jitter=np.asarray(snap.jitter),
        kernel=np.asarray(snap.kernel),
        x_lo=np.asarray(snap.x_lo, dtype=float),
        x_hi=np.asarray(snap.x_hi, dtype=float),
        y_mean=np.asarray(snap.y_mean),
        y_std=np.asarray(snap.y_std),
        eps=np.asarray(snap.eps),
    )


def _snapshot_path(snapshot_dir: str, it: int) -> str:
    return os.path.join(snapshot_dir, f"it_{it:06d}.npz")


def save_snapshot(snapshot_dir: str, snap: GPSnapshot) -> None:
    os.makedirs(snapshot_dir, exist_ok=True)
    np.savez(_snapshot_path(snapshot_dir, snap.it), **_snapshot_arrays(snap))


def flush_snapshots(snapshot_dir: str, snapshots: List[GPSnapshot]) -> None:
    for snap in snapshots:
        save_snapshot(snapshot_dir, snap)


def load_snapshot(path: str) -> GPSnapshot:
    # allow_pickle defaults to False: every array here is numeric or a
    # fixed-width unicode string, so no pickling is ever required.
    data = np.load(path)
    return GPSnapshot(
        it=int(data["it"]),
        X=data["X"],
        y=data["y"],
        l_c=float(data["l_c"]),
        sigma_f=float(data["sigma_f"]),
        sigma_y=float(data["sigma_y"]),
        jitter=float(data["jitter"]),
        kernel=str(data["kernel"]),
        x_lo=data["x_lo"],
        x_hi=data["x_hi"],
        y_mean=float(data["y_mean"]),
        y_std=float(data["y_std"]),
        eps=float(data["eps"]),
    )


def reconstruct_gp(snap: GPSnapshot) -> GPModel:
    """Refit a GPModel from a snapshot: deterministic, no randomness anywhere
    in gp_fit, so this reproduces the live alpha/L (and therefore
    predictions) bit-for-bit whenever the live run used a full refit at that
    iteration (i.e. below GPConfig.rank_one_threshold, which is the only
    other code path that touches alpha/L)."""

    if snap.kernel != _KERNEL_ID:
        raise ValueError(f"Unknown kernel identifier '{snap.kernel}' (expected '{_KERNEL_ID}').")

    normalizer = NormalizationHelper(
        x_lo=np.asarray(snap.x_lo, dtype=float),
        x_hi=np.asarray(snap.x_hi, dtype=float),
        y_mean=snap.y_mean,
        y_std=snap.y_std,
        eps=snap.eps,
    )

    gp = GPModel(
        l_c=snap.l_c,
        sigma_f=snap.sigma_f,
        sigma_y=snap.sigma_y,
        jitter=snap.jitter,
        normalizer=normalizer,
    )
    gp.Xs = np.asarray(snap.X, dtype=float)
    gp.ys = np.asarray(snap.y, dtype=float)
    gp.Xs_norm = normalizer.normalize_X(gp.Xs)
    gp.ys_norm = normalizer.normalize_y(gp.ys)
    gp.alpha, gp.L = gp_fit(
        gp.Xs_norm, gp.ys_norm, gp.l_c, gp.sigma_f, gp.sigma_y, gp.jitter
    )
    return gp


#%% ---------------------------------------------------------------------
# Run metadata (JSON)
# -------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def write_meta(out_path: str, bounds, random_state: Optional[int], **configs: Any) -> None:
    """Write run configuration (bounds, seed, and any dataclass configs
    passed as keyword args) to ``meta.json`` for human inspection."""

    payload = dict(bounds=list(bounds), random_state=random_state)
    for name, cfg in configs.items():
        payload[name] = cfg.__dict__ if hasattr(cfg, "__dict__") else cfg

    with open(os.path.join(out_path, "meta.json"), "w") as fh:
        json.dump(payload, fh, default=_json_default, indent=2)


def load_meta(out_path: str) -> Optional[Dict[str, Any]]:
    meta_path = os.path.join(out_path, "meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as fh:
        return json.load(fh)


#%% ---------------------------------------------------------------------
# Loader / replay
# -------------------------------------------------------------------------

@dataclass
class GPPosterior:
    """A GP posterior reconstructed at one iteration; call .predict() like a
    live GP (see core.gp_predict)."""

    it: int
    gp: GPModel

    def predict(self, X: Array, return_cov: bool = False):
        return gp_predict(
            X, self.gp.Xs, self.gp.alpha, self.gp.L,
            l_c=self.gp.l_c, sigma_f=self.gp.sigma_f,
            return_cov=return_cov, gp=self.gp,
        )


class ReplayRun:
    """Read-only view over a persisted run directory: Tier 1 trace plus
    whichever Tier 2 snapshots were recorded."""

    def __init__(
        self,
        out_path: str,
        meta: Optional[Dict[str, Any]],
        trace: Optional[TraceLog],
        snapshot_dir: str,
        available_iterations: List[int],
    ):
        self.out_path = out_path
        self.meta = meta
        self.trace = trace
        self._snapshot_dir = snapshot_dir
        self.available_iterations = available_iterations

    def posterior(self, it: int) -> GPPosterior:
        if it not in self.available_iterations:
            raise KeyError(
                f"No Tier-2 snapshot at iteration {it}. "
                f"Available iterations: {self.available_iterations}"
            )
        snap = load_snapshot(_snapshot_path(self._snapshot_dir, it))
        gp = reconstruct_gp(snap)
        return GPPosterior(it=it, gp=gp)


def load_run(
    out_path: str,
    trace_filename: str = "trace.npz",
    snapshot_dirname: str = "snapshots",
) -> ReplayRun:
    meta = load_meta(out_path)

    trace_path = os.path.join(out_path, trace_filename)
    trace = load_trace(trace_path) if os.path.exists(trace_path) else None

    snapshot_dir = os.path.join(out_path, snapshot_dirname)
    available: List[int] = []
    if os.path.isdir(snapshot_dir):
        for fp in glob.glob(os.path.join(snapshot_dir, "it_*.npz")):
            m = _SNAPSHOT_RE.search(os.path.basename(fp))
            if m:
                available.append(int(m.group(1)))
    available.sort()

    return ReplayRun(
        out_path=out_path,
        meta=meta,
        trace=trace,
        snapshot_dir=snapshot_dir,
        available_iterations=available,
    )
