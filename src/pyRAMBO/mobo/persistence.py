"""
Two-tier persistence for mobo Multi-Objective Bayesian Optimization runs.

Mirrors the two-tier design of sbo.persistence and mfbo.persistence (Tier 1
"trace" always on and cheap, Tier 2 "snapshots" opt-in and heavier), with
the multi-objective specifics called out explicitly rather than folded
silently into the single-objective shape:

Tier 1 ("trace"): one row per BO iteration -- proposed x, the whole observed
objective VECTOR y, the dominated HYPERVOLUME and the SIZE OF THE PARETO
FRONT after that iteration's evaluation (i.e. including the point just
evaluated), wall time, acquisition value, and the GP hyperparameters of the
iteration (ARD length-scales, coregionalization matrix B, noise stds).
There is no x_best/y_best column: a multi-objective run has no scalar
incumbent, and the hypervolume is the progress measure that replaces it.
Always recorded in memory; written to disk as ``trace.npz`` plus a
human-readable sibling ``trace.csv`` whenever ``SaveConfig.out_path`` is set.
Both land in ``<run folder>/res/``, alongside ``meta.json`` and (if enabled)
``snapshots/`` -- ``<run folder>/plots/`` is the separate figures directory
written by plotting.py.

The Pareto front itself is deliberately NOT stored in the trace: it changes
length from iteration to iteration, which a fixed-shape .npz cannot hold.
It is recoverable exactly from the Tier 2 snapshots (which carry the full X
and Y), and from ``MOBOResult.history`` in memory.

Tier 2 ("snapshots"): opt-in, heavier per-iteration records of everything
needed to refit the ICM GP exactly -- the physical (X, Y) design, fitted
hyperparameters (ARD length-scales, the coregionalization matrix B, the
per-output noise), kernel identity, normalization state, and the run's
reference point. The dense Cholesky factor is deliberately never stored:
reconstruct_gp recomputes it from scratch via the same icm_gp_fit routine
used live.

This module imports concrete objects from .core, so core.py must import
this module lazily (inside functions, not at module scope) -- the same
pattern sbo and mfbo use to avoid a circular import.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import csv
import glob
import json
import os
import re

import numpy as np

from .core import (
    Array,
    ICMGPModel,
    NormalizationHelper,
    icm_gp_fit,
    icm_gp_predict,
)

_KERNEL_ID = "icm_rbf_ard"  # the only kernel currently implemented in core.py

_SNAPSHOT_RE = re.compile(r"it_(\d+)\.npz$")


#%% ---------------------------------------------------------------------
# Tier 1: lightweight per-iteration trace
# -------------------------------------------------------------------------

@dataclass
class TraceLog:
    """Tier 1 record: one row per MOBO iteration."""

    it: List[int] = field(default_factory=list)
    x_next: List[Array] = field(default_factory=list)
    y_next: List[Array] = field(default_factory=list)
    hypervolume: List[float] = field(default_factory=list)
    n_pareto: List[int] = field(default_factory=list)
    wall_time: List[float] = field(default_factory=list)
    acq_value: List[float] = field(default_factory=list)
    length_scales: List[Array] = field(default_factory=list)   # (d,) ARD length-scales
    B: List[Array] = field(default_factory=list)               # (M, M) coregionalization matrix
    sigma_n: List[Array] = field(default_factory=list)         # (M,) observation noise stds

    def append(
        self,
        it: int,
        x_next: Array,
        y_next: Array,
        hypervolume: float,
        n_pareto: int,
        wall_time: float,
        acq_value: float,
        length_scales: Array,
        B: Array,
        sigma_n: Array,
    ) -> None:
        self.it.append(int(it))
        self.x_next.append(np.asarray(x_next, dtype=float).copy())
        self.y_next.append(np.asarray(y_next, dtype=float).copy())
        self.hypervolume.append(float(hypervolume))
        self.n_pareto.append(int(n_pareto))
        self.wall_time.append(float(wall_time))
        self.acq_value.append(float(acq_value))
        self.length_scales.append(np.asarray(length_scales, dtype=float).reshape(-1).copy())
        self.B.append(np.asarray(B, dtype=float).copy())
        self.sigma_n.append(np.asarray(sigma_n, dtype=float).reshape(-1).copy())

    def __len__(self) -> int:
        return len(self.it)

    def to_arrays(self) -> Dict[str, Array]:
        return dict(
            it=np.asarray(self.it, dtype=int),
            x_next=np.asarray(self.x_next, dtype=float),
            y_next=np.asarray(self.y_next, dtype=float),
            hypervolume=np.asarray(self.hypervolume, dtype=float),
            n_pareto=np.asarray(self.n_pareto, dtype=int),
            wall_time=np.asarray(self.wall_time, dtype=float),
            acq_value=np.asarray(self.acq_value, dtype=float),
            length_scales=np.asarray(self.length_scales, dtype=float),
            B=np.asarray(self.B, dtype=float),
            sigma_n=np.asarray(self.sigma_n, dtype=float),
        )

    def save(self, path: str) -> None:
        """Write trace.npz (arrays) and a sibling trace.csv (flattened,
        human-readable: one row per iteration; x_next, y_next, the length
        scales, the lower triangle of B and sigma_n split into one column
        each)."""
        np.savez(path, **self.to_arrays())
        self._write_csv(os.path.splitext(path)[0] + ".csv")

    def _write_csv(self, path: str) -> None:
        n = len(self.it)
        d = np.asarray(self.x_next[0]).reshape(-1).shape[0] if n > 0 else 0
        M = np.asarray(self.y_next[0]).reshape(-1).shape[0] if n > 0 else 0
        tril = [(i, j) for i in range(M) for j in range(i + 1)]

        header = (
            ["it"]
            + [f"x_next_{j}" for j in range(d)]
            + [f"y_next_{m}" for m in range(M)]
            + ["hypervolume", "n_pareto", "wall_time", "acq_value"]
            + [f"length_scale_{j}" for j in range(d)]
            + [f"B_{i}{j}" for i, j in tril]
            + [f"sigma_n_{m}" for m in range(M)]
        )
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for k in range(n):
                writer.writerow(
                    [self.it[k]]
                    + list(np.asarray(self.x_next[k]).reshape(-1))
                    + list(np.asarray(self.y_next[k]).reshape(-1))
                    + [self.hypervolume[k], self.n_pareto[k], self.wall_time[k], self.acq_value[k]]
                    + list(self.length_scales[k])
                    + [self.B[k][i, j] for i, j in tril]
                    + list(self.sigma_n[k])
                )


def load_trace(path: str) -> TraceLog:
    data = np.load(path)
    trace = TraceLog()
    trace.it = data["it"].astype(int).tolist()
    trace.x_next = list(np.atleast_2d(data["x_next"]))
    trace.y_next = list(np.atleast_2d(data["y_next"]))
    trace.hypervolume = data["hypervolume"].astype(float).tolist()
    trace.n_pareto = data["n_pareto"].astype(int).tolist()
    trace.wall_time = data["wall_time"].astype(float).tolist()
    trace.acq_value = data["acq_value"].astype(float).tolist()
    # Backward compatible with trace.npz files written before the GP
    # hyperparameters were recorded: fall back to NaN arrays rather than raising.
    n = len(trace.it)
    d = np.atleast_2d(data["x_next"]).shape[1]
    M = np.atleast_2d(data["y_next"]).shape[1]
    trace.length_scales = (list(np.atleast_2d(data["length_scales"])) if "length_scales" in data.files
                           else [np.full(d, np.nan)] * n)
    trace.B = list(data["B"]) if "B" in data.files else [np.full((M, M), np.nan)] * n
    trace.sigma_n = (list(np.atleast_2d(data["sigma_n"])) if "sigma_n" in data.files
                     else [np.full(M, np.nan)] * n)
    return trace


#%% ---------------------------------------------------------------------
# Tier 2: full GP-reconstruction snapshots
# -------------------------------------------------------------------------

@dataclass
class GPSnapshot:
    """Everything needed to refit the ICM GP exactly at one iteration.

    Deliberately excludes alpha/L (the dense Cholesky factor): those are
    recomputed by ``reconstruct_gp`` via the same ``icm_gp_fit`` used live.
    The per-output normalization statistics ARE stored rather than
    re-derived from Y, so a reconstructed posterior stays valid even if the
    normalization convention in core.py were to change."""

    it: int
    X: Array
    Y: Array
    length_scales: Array
    B: Array
    sigma_n: Array
    jitter: float
    kernel: str
    x_lo: Array
    x_hi: Array
    y_mean: Array
    y_std: Array
    ref_point: Array


def build_gp_snapshot(it: int, gp: ICMGPModel, ref_point: Array) -> GPSnapshot:
    if gp.X is None or gp.Y is None or gp.normalizer is None:
        raise ValueError("GP has no fitted data / normalizer yet; cannot snapshot.")
    if gp.normalizer.y_mean is None or gp.normalizer.y_std is None:
        raise ValueError("GP normalizer has not seen any objective values yet; cannot snapshot.")

    normalizer = gp.normalizer
    return GPSnapshot(
        it=int(it),
        X=np.asarray(gp.X, dtype=float).copy(),
        Y=np.asarray(gp.Y, dtype=float).copy(),
        length_scales=np.asarray(gp.length_scales, dtype=float).copy(),
        B=np.asarray(gp.B, dtype=float).copy(),
        sigma_n=np.asarray(gp.sigma_n, dtype=float).copy(),
        jitter=float(gp.jitter),
        kernel=_KERNEL_ID,
        x_lo=np.asarray(normalizer.x_lo, dtype=float).copy(),
        x_hi=np.asarray(normalizer.x_hi, dtype=float).copy(),
        y_mean=np.asarray(normalizer.y_mean, dtype=float).copy(),
        y_std=np.asarray(normalizer.y_std, dtype=float).copy(),
        ref_point=np.asarray(ref_point, dtype=float).copy(),
    )


def _snapshot_arrays(snap: GPSnapshot) -> Dict[str, Array]:
    return dict(
        it=np.asarray(snap.it),
        X=np.asarray(snap.X, dtype=float),
        Y=np.asarray(snap.Y, dtype=float),
        length_scales=np.asarray(snap.length_scales, dtype=float),
        B=np.asarray(snap.B, dtype=float),
        sigma_n=np.asarray(snap.sigma_n, dtype=float),
        jitter=np.asarray(snap.jitter),
        kernel=np.asarray(snap.kernel),
        x_lo=np.asarray(snap.x_lo, dtype=float),
        x_hi=np.asarray(snap.x_hi, dtype=float),
        y_mean=np.asarray(snap.y_mean, dtype=float),
        y_std=np.asarray(snap.y_std, dtype=float),
        ref_point=np.asarray(snap.ref_point, dtype=float),
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
        X=data["X"], Y=data["Y"],
        length_scales=data["length_scales"], B=data["B"], sigma_n=data["sigma_n"],
        jitter=float(data["jitter"]),
        kernel=str(data["kernel"]),
        x_lo=data["x_lo"], x_hi=data["x_hi"],
        y_mean=data["y_mean"], y_std=data["y_std"],
        ref_point=data["ref_point"],
    )


def reconstruct_gp(snap: GPSnapshot) -> ICMGPModel:
    """Refit an ICMGPModel from a snapshot: deterministic, so this reproduces
    the live alpha/L (and therefore predictions) bit-for-bit."""

    if snap.kernel != _KERNEL_ID:
        raise ValueError(f"Unknown kernel identifier '{snap.kernel}' (expected '{_KERNEL_ID}').")

    normalizer = NormalizationHelper(
        x_lo=np.asarray(snap.x_lo, dtype=float),
        x_hi=np.asarray(snap.x_hi, dtype=float),
        y_mean=np.asarray(snap.y_mean, dtype=float),
        y_std=np.asarray(snap.y_std, dtype=float),
    )

    B = np.asarray(snap.B, dtype=float)
    gp = ICMGPModel(
        n_obj=B.shape[0],
        length_scales=np.asarray(snap.length_scales, dtype=float),
        B=B,
        sigma_n=np.asarray(snap.sigma_n, dtype=float),
        jitter=float(snap.jitter),
        normalizer=normalizer,
    )
    gp.X = np.asarray(snap.X, dtype=float)
    gp.Y = np.asarray(snap.Y, dtype=float)
    gp.X_norm = normalizer.normalize_X(gp.X)
    gp.Y_norm = normalizer.normalize_Y(gp.Y)
    gp.alpha, gp.L = icm_gp_fit(gp, gp.X_norm, gp.Y_norm)
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


def write_meta(out_path: str, bounds, random_state: Optional[int], ref_point=None, **configs: Any) -> None:
    """Write the run configuration to ``meta.json``. out_path is created if
    missing: this is the first thing written for a run, called before any
    trace/snapshot flush."""
    os.makedirs(out_path, exist_ok=True)

    payload: Dict[str, Any] = dict(bounds=list(bounds), random_state=random_state)
    if ref_point is not None:
        payload["ref_point"] = np.asarray(ref_point, dtype=float).tolist()
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
    live GP (see core.icm_gp_predict). Predicts all objectives jointly."""

    it: int
    gp: ICMGPModel
    ref_point: Array

    def predict(self, X: Array, return_cov: bool = False):
        return icm_gp_predict(X, self.gp, return_cov=return_cov)


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
        return GPPosterior(it=it, gp=gp, ref_point=np.asarray(snap.ref_point, dtype=float))


def load_run(
    out_path: str,
    trace_filename: str = "trace.npz",
    snapshot_dirname: str = "snapshots",
) -> ReplayRun:
    """out_path is the run folder (MOBOResult.out_path); persisted data lives
    under out_path/res/, mirroring the layout core writes."""
    res_path = os.path.join(out_path, "res")

    meta = load_meta(res_path)

    trace_path = os.path.join(res_path, trace_filename)
    trace = load_trace(trace_path) if os.path.exists(trace_path) else None

    snapshot_dir = os.path.join(res_path, snapshot_dirname)
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
