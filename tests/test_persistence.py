"""
Tests for the two-tier (trace / snapshot) persistence layer in
pyRAMBO.sbo.persistence.

Run with: pytest
"""

import os
from unittest.mock import patch

import numpy as np
import pytest

from pyRAMBO import sbo as bo
from pyRAMBO.sbo import persistence


def _quadratic(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return float((x[0] - 0.7) ** 2)


def _run(tmp_path, **save_kwargs):
    return bo.bayesian_optimization(
        f=_quadratic,
        bounds=[(-2.0, 2.0)],
        bo_cfg=bo.BOConfig(n_init=4, n_iter=10, random_state=0),
        gp_cfg=bo.GPConfig(),
        acq_cfg=bo.AcqConfig(),
        optim_cfg=bo.OptimConfig(n_raw_samples=200),
        save_cfg=bo.SaveConfig(
            out_path=str(tmp_path),
            create_timestamp=False,
            log_enabled=False,
            **save_kwargs,
        ),
    )


# ---------------------------------------------------------------------
# Tier 1: trace is always on and cheap
# ---------------------------------------------------------------------

def test_trace_enabled_by_default_no_out_path():
    """Default SaveConfig() = Tier 1 only, purely in-memory (no disk I/O)."""
    res = bo.bayesian_optimization(
        f=_quadratic,
        bounds=[(-2.0, 2.0)],
        bo_cfg=bo.BOConfig(n_init=4, n_iter=6, random_state=0),
        gp_cfg=bo.GPConfig(),
        acq_cfg=bo.AcqConfig(),
        optim_cfg=bo.OptimConfig(n_raw_samples=200),
        save_cfg=bo.SaveConfig(),
    )
    assert res.trace is not None
    assert len(res.trace) == 6
    assert res.snapshots is None  # Tier 2 never allocated


def test_trace_written_and_reloadable(tmp_path):
    res = _run(tmp_path)
    trace_path = os.path.join(res.out_path, "res", "trace.npz")
    assert os.path.exists(trace_path)
    assert os.path.exists(os.path.join(res.out_path, "res", "trace.csv"))

    loaded = persistence.load_trace(trace_path)
    assert loaded.it == res.trace.it
    np.testing.assert_allclose(np.asarray(loaded.y_best), np.asarray(res.trace.y_best))
    np.testing.assert_allclose(
        np.asarray(loaded.x_next), np.asarray(res.trace.x_next)
    )


def test_trace_fields_match_spec(tmp_path):
    res = _run(tmp_path)
    trace = res.trace
    n = 10
    assert len(trace.it) == n
    for field_name in ("x_next", "y_next", "x_best", "y_best", "wall_time", "acq_value"):
        assert len(getattr(trace, field_name)) == n
    # y_best must be monotonically non-increasing (minimization, incumbent)
    y_best = np.asarray(trace.y_best)
    assert np.all(np.diff(y_best) <= 1e-12)


def test_trace_best_includes_current_iteration(tmp_path):
    """Row `it` must be the running minimum INCLUDING the point evaluated at
    iteration `it` (no one-iteration lag), and match the final best."""
    res = _run(tmp_path)
    trace = res.trace
    n_init = 4
    y_best = np.asarray(trace.y_best)
    y_next = np.asarray(trace.y_next)
    running = np.minimum.accumulate(np.concatenate([[res.y[:n_init].min()], y_next]))[1:]
    np.testing.assert_allclose(y_best, running)
    assert y_best[-1] == pytest.approx(res.best_y)
    # State keeps the pre-update incumbent that the acquisition used.
    for st, yb in zip(res.states, y_best):
        assert st.y_best == pytest.approx(yb)
        assert st.y_best_acq >= st.y_best


# ---------------------------------------------------------------------
# Tier 2: off by default, never allocated when disabled
# ---------------------------------------------------------------------

def test_snapshot_disabled_by_default_never_built(tmp_path):
    with patch.object(persistence, "build_gp_snapshot", wraps=persistence.build_gp_snapshot) as spy:
        res = _run(tmp_path)  # snapshot_enabled defaults to False
    spy.assert_not_called()
    assert res.snapshots is None
    assert not os.path.isdir(os.path.join(res.out_path, "res", "snapshots"))


def test_snapshot_enabled_builds_one_per_iteration(tmp_path):
    with patch.object(persistence, "build_gp_snapshot", wraps=persistence.build_gp_snapshot) as spy:
        res = _run(tmp_path, snapshot_enabled=True, snapshot_every=1)
    assert spy.call_count == 10
    assert res.snapshots is not None
    assert len(res.snapshots) == 10

    snap_dir = os.path.join(res.out_path, "res", "snapshots")
    files = sorted(os.listdir(snap_dir))
    assert len(files) == 10


def test_snapshot_every_k_skips_iterations(tmp_path):
    res = _run(tmp_path, snapshot_enabled=True, snapshot_every=4)
    # iterations 0, 4, 8 -> 3 snapshots out of 10
    assert [s.it for s in res.snapshots] == [0, 4, 8]


def test_snapshot_flush_every_clears_memory_buffer(tmp_path):
    res = _run(tmp_path, snapshot_enabled=True, snapshot_every=1, snapshot_flush_every=3)
    # 10 snapshots, flushed+cleared every 3 -> buffers of [3,3,3], remainder 1 left in memory
    assert len(res.snapshots) == 1
    snap_dir = os.path.join(res.out_path, "res", "snapshots")
    assert len(os.listdir(snap_dir)) == 10  # all 10 still land on disk


def test_meta_json_written(tmp_path):
    res = _run(tmp_path)
    meta = persistence.load_meta(os.path.join(res.out_path, "res"))
    assert meta is not None
    assert meta["bo_cfg"]["n_iter"] == 10
    assert meta["bounds"] == [[-2.0, 2.0]]


def test_run_folder_created_under_out_path(tmp_path):
    """create_timestamp=False still gets its own run_<n> subfolder -- never
    written directly into out_path, never nesting into a previous run."""
    res1 = _run(tmp_path)
    res2 = _run(tmp_path)
    assert res1.out_path != res2.out_path
    assert os.path.dirname(res1.out_path) == str(tmp_path)
    assert os.path.dirname(res2.out_path) == str(tmp_path)
    assert os.path.basename(res1.out_path) == "run_1"
    assert os.path.basename(res2.out_path) == "run_2"


# ---------------------------------------------------------------------
# Round trip: reconstructed posterior must match the live GP
# ---------------------------------------------------------------------

def test_round_trip_posterior_matches_live_model(tmp_path):
    res = _run(tmp_path, snapshot_enabled=True, snapshot_every=1)

    run = persistence.load_run(res.out_path)
    assert run.available_iterations == list(range(10))

    Xgrid = np.linspace(-2.0, 2.0, 50).reshape(-1, 1)

    for it in (0, 4, 9):
        live_gp = res.states[it].gp
        mu_live, var_live = bo.gp_predict(
            Xgrid, live_gp.Xs, live_gp.alpha, live_gp.L,
            l_c=live_gp.l_c, sigma_f=live_gp.sigma_f,
            return_cov=False, gp=live_gp,
        )

        posterior = run.posterior(it)
        mu_loaded, var_loaded = posterior.predict(Xgrid)

        np.testing.assert_allclose(mu_loaded, mu_live, rtol=0, atol=1e-10)
        np.testing.assert_allclose(var_loaded, var_live, rtol=0, atol=1e-10)


def test_posterior_missing_iteration_raises(tmp_path):
    res = _run(tmp_path, snapshot_enabled=True, snapshot_every=5)  # its 0 and 5 only
    run = persistence.load_run(res.out_path)
    with pytest.raises(KeyError):
        run.posterior(1)


def test_reconstruct_gp_excludes_dense_factors():
    """GPSnapshot must not carry alpha/L/Xs_norm/ys_norm -- only what's
    needed to refit from scratch."""
    gp_cfg = bo.GPConfig()
    bounds = [(-2.0, 2.0)]
    gp = bo.build_gp_model(gp_cfg, bounds)
    X = np.array([[-1.0], [0.0], [1.0]])
    y = np.array([1.0, 0.0, 1.0])
    gp = bo.fit_gp(gp, X, y, gp_cfg, it=0)

    snap = persistence.build_gp_snapshot(0, gp)
    snap_fields = set(vars(snap).keys())
    assert "alpha" not in snap_fields
    assert "L" not in snap_fields
    assert "Xs_norm" not in snap_fields
    assert "ys_norm" not in snap_fields

    rebuilt = persistence.reconstruct_gp(snap)
    np.testing.assert_allclose(rebuilt.alpha, gp.alpha, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rebuilt.L, gp.L, rtol=0, atol=1e-12)



def test_log_records_total_run_time(tmp_path):
    """log.log ends with a summary line carrying the total wall time of the run."""
    res = bo.bayesian_optimization(
        f=_quadratic,
        bounds=[(-2.0, 2.0)],
        bo_cfg=bo.BOConfig(n_init=4, n_iter=5, random_state=0),
        gp_cfg=bo.GPConfig(),
        acq_cfg=bo.AcqConfig(),
        optim_cfg=bo.OptimConfig(n_raw_samples=100),
        save_cfg=bo.SaveConfig(out_path=str(tmp_path), log_enabled=True),
    )
    with open(os.path.join(res.out_path, "log.log")) as fh:
        lines = fh.read().strip().splitlines()
    assert "BO run started" in lines[0]
    last = lines[-1]
    assert "Finished" in last and "total time = " in last
    seconds = float(last.split("total time = ")[1].split(" s")[0])
    assert 0.0 < seconds < 60.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
