"""
Two-tier persistence for mfbo Multi-Fidelity Bayesian Optimization runs.

Mirrors the two-tier design of sbo.persistence (Tier 1 "trace" always on and
cheap, Tier 2 "snapshots" opt-in and heavier), with the multi-fidelity
specifics called out explicitly rather than folded silently into the
sbo shape:

Tier 1 ("trace"): one row per BO iteration -- proposed x, the FIDELITY LEVEL
that was queried, observed y, incumbent best x/y (high-fidelity only), the
COST of that iteration's evaluation and the RUNNING CUMULATIVE COST, wall
time, and acquisition value. Always recorded in memory; written to disk as a
single ``trace.npz`` whenever ``SaveConfig.out_path`` is set.

Tier 2 ("snapshots"): opt-in, heavier per-iteration records of everything
needed to refit the joint AR1 GP exactly -- the physical (X_L, y_L, X_H, y_H)
design, fitted hyperparameters, kernel identity, and normalization state.
The dense joint Cholesky factor is deliberately never stored: reconstruct_gp
recomputes it from scratch via the same mf_gp_fit routine used live.

This module imports concrete objects from .core, so core.py must import this
module lazily (inside functions, not at module scope) -- the same pattern
sbo uses to avoid a circular import.

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
    MFGPModel,
    NormalizationHelper,
    mf_gp_fit,
    mf_gp_predict,
)

_KERNEL_ID = "mf_ar1"  # the only kernel currently implemented in core.py

_SNAPSHOT_RE = re.compile(r"it_(\d+)\.npz$")


#%% ---------------------------------------------------------------------
# Tier 1: lightweight per-iteration trace
# -------------------------------------------------------------------------

@dataclass
class TraceLog:
    """Tier 1 record: one row per MFBO iteration."""

    it: List[int] = field(default_factory=list)
    x_next: List[Array] = field(default_factory=list)
    level_next: List[str] = field(default_factory=list)
    y_next: List[float] = field(default_factory=list)
    x_best: List[Array] = field(default_factory=list)
    y_best: List[float] = field(default_factory=list)
    wall_time: List[float] = field(default_factory=list)
    acq_value: List[float] = field(default_factory=list)
    cost_next: List[float] = field(default_factory=list)
    cumulative_cost: List[float] = field(default_factory=list)

    def append(
        self,
        it: int,
        x_next: Array,
        level_next: str,
        y_next: float,
        x_best: Array,
        y_best: float,
        wall_time: float,
        acq_value: float,
        cost_next: float,
        cumulative_cost: float,
    ) -> None:
        self.it.append(int(it))
        self.x_next.append(np.asarray(x_next, dtype=float).copy())
        self.level_next.append(str(level_next))
        self.y_next.append(float(y_next))
        self.x_best.append(np.asarray(x_best, dtype=float).copy())
        self.y_best.append(float(y_best))
        self.wall_time.append(float(wall_time))
        self.acq_value.append(float(acq_value))
        self.cost_next.append(float(cost_next))
        self.cumulative_cost.append(float(cumulative_cost))

    def __len__(self) -> int:
        return len(self.it)

    def to_arrays(self) -> Dict[str, Array]:
        return dict(
            it=np.asarray(self.it, dtype=int),
            x_next=np.asarray(self.x_next, dtype=float),
            level_next=np.asarray(self.level_next, dtype="<U1"),
            y_next=np.asarray(self.y_next, dtype=float),
            x_best=np.asarray(self.x_best, dtype=float),
            y_best=np.asarray(self.y_best, dtype=float),
            wall_time=np.asarray(self.wall_time, dtype=float),
            acq_value=np.asarray(self.acq_value, dtype=float),
            cost_next=np.asarray(self.cost_next, dtype=float),
            cumulative_cost=np.asarray(self.cumulative_cost, dtype=float),
        )

    def save(self, path: str) -> None:
        np.savez(path, **self.to_arrays())


def load_trace(path: str) -> TraceLog:
    data = np.load(path)
    trace = TraceLog()
    trace.it = data["it"].astype(int).tolist()
    trace.x_next = list(np.atleast_2d(data["x_next"]))
    trace.level_next = data["level_next"].astype(str).tolist()
    trace.y_next = data["y_next"].astype(float).tolist()
    trace.x_best = list(np.atleast_2d(data["x_best"]))
    trace.y_best = data["y_best"].astype(float).tolist()
    trace.wall_time = data["wall_time"].astype(float).tolist()
    trace.acq_value = data["acq_value"].astype(float).tolist()
    trace.cost_next = data["cost_next"].astype(float).tolist()
    trace.cumulative_cost = data["cumulative_cost"].astype(float).tolist()
    return trace


#%% ---------------------------------------------------------------------
# Tier 2: full GP-reconstruction snapshots
# -------------------------------------------------------------------------

@dataclass
class GPSnapshot:
    """Everything needed to refit the joint AR1 GP exactly at one iteration.

    Deliberately excludes alpha/L (dense joint Cholesky factor): those are
    recomputed by ``reconstruct_gp`` via the same ``mf_gp_fit`` used live.
    """

    it: int
    X_L: Array
    y_L: Array
    X_H: Array
    y_H: Array
    l_lf: float
    sigma_lf: float
    l_delta: float
    sigma_delta: float
    rho: float
    sigma_L: float
    sigma_H: float
    jitter: float
    kernel: str
    x_lo: Array
    x_hi: Array


def build_gp_snapshot(it: int, gp: MFGPModel) -> GPSnapshot:
    if gp.X_L is None or gp.y_L is None or gp.X_H is None or gp.y_H is None or gp.normalizer is None:
        raise ValueError("GP has no fitted data / normalizer yet; cannot snapshot.")

    normalizer = gp.normalizer
    return GPSnapshot(
        it=int(it),
        X_L=np.asarray(gp.X_L, dtype=float).copy(),
        y_L=np.asarray(gp.y_L, dtype=float).copy(),
        X_H=np.asarray(gp.X_H, dtype=float).copy(),
        y_H=np.asarray(gp.y_H, dtype=float).copy(),
        l_lf=float(gp.l_lf), sigma_lf=float(gp.sigma_lf),
        l_delta=float(gp.l_delta), sigma_delta=float(gp.sigma_delta),
        rho=float(gp.rho), sigma_L=float(gp.sigma_L), sigma_H=float(gp.sigma_H),
        jitter=float(gp.jitter),
        kernel=_KERNEL_ID,
        x_lo=np.asarray(normalizer.x_lo, dtype=float).copy(),
        x_hi=np.asarray(normalizer.x_hi, dtype=float).copy(),
    )


def _snapshot_arrays(snap: GPSnapshot) -> Dict[str, Array]:
    return dict(
        it=np.asarray(snap.it),
        X_L=np.asarray(snap.X_L, dtype=float),
        y_L=np.asarray(snap.y_L, dtype=float),
        X_H=np.asarray(snap.X_H, dtype=float),
        y_H=np.asarray(snap.y_H, dtype=float),
        l_lf=np.asarray(snap.l_lf), sigma_lf=np.asarray(snap.sigma_lf),
        l_delta=np.asarray(snap.l_delta), sigma_delta=np.asarray(snap.sigma_delta),
        rho=np.asarray(snap.rho), sigma_L=np.asarray(snap.sigma_L), sigma_H=np.asarray(snap.sigma_H),
        jitter=np.asarray(snap.jitter),
        kernel=np.asarray(snap.kernel),
        x_lo=np.asarray(snap.x_lo, dtype=float),
        x_hi=np.asarray(snap.x_hi, dtype=float),
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
    data = np.load(path)
    return GPSnapshot(
        it=int(data["it"]),
        X_L=data["X_L"], y_L=data["y_L"], X_H=data["X_H"], y_H=data["y_H"],
        l_lf=float(data["l_lf"]), sigma_lf=float(data["sigma_lf"]),
        l_delta=float(data["l_delta"]), sigma_delta=float(data["sigma_delta"]),
        rho=float(data["rho"]), sigma_L=float(data["sigma_L"]), sigma_H=float(data["sigma_H"]),
        jitter=float(data["jitter"]),
        kernel=str(data["kernel"]),
        x_lo=data["x_lo"], x_hi=data["x_hi"],
    )


def reconstruct_gp(snap: GPSnapshot) -> MFGPModel:
    """Refit an MFGPModel from a snapshot: deterministic, so this reproduces
    the live alpha/L (and therefore predictions) bit-for-bit."""

    if snap.kernel != _KERNEL_ID:
        raise ValueError(f"Unknown kernel identifier '{snap.kernel}' (expected '{_KERNEL_ID}').")

    normalizer = NormalizationHelper(
        x_lo=np.asarray(snap.x_lo, dtype=float),
        x_hi=np.asarray(snap.x_hi, dtype=float),
    )

    gp = MFGPModel(
        l_lf=snap.l_lf, sigma_lf=snap.sigma_lf,
        l_delta=snap.l_delta, sigma_delta=snap.sigma_delta,
        rho=snap.rho, sigma_L=snap.sigma_L, sigma_H=snap.sigma_H,
        jitter=snap.jitter, normalizer=normalizer,
    )
    gp.X_L = np.asarray(snap.X_L, dtype=float)
    gp.y_L = np.asarray(snap.y_L, dtype=float)
    gp.X_H = np.asarray(snap.X_H, dtype=float)
    gp.y_H = np.asarray(snap.y_H, dtype=float)
    gp.X_L_norm = normalizer.normalize_X(gp.X_L)
    gp.X_H_norm = normalizer.normalize_X(gp.X_H)
    gp.alpha, gp.L = mf_gp_fit(gp, gp.X_L_norm, gp.y_L, gp.X_H_norm, gp.y_H)
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
    live GP (see core.mf_gp_predict). Always predicts the high-fidelity
    (target) function."""

    it: int
    gp: MFGPModel

    def predict(self, X: Array, return_cov: bool = False):
        return mf_gp_predict(
            X, self.gp, self.gp.X_L_norm, self.gp.X_H_norm, self.gp.alpha, self.gp.L,
            return_cov=return_cov,
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
