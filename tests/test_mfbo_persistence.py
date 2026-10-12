"""
Tests for the two-tier (trace / snapshot) persistence layer in
pyRAMBO.mfbo.persistence.

Run with: pytest
"""

import os
from unittest.mock import patch

import numpy as np
import pytest

from pyRAMBO import mfbo
from pyRAMBO.mfbo import persistence


def _f_high(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return float((x[0] - 0.7) ** 2)


def _f_low(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(0.9 * (x[0] - 0.7) ** 2 + 0.05 * np.sin(15.0 * x[0]))


def _run(tmp_path=None, **save_kwargs):
    save_cfg = mfbo.SaveConfig(
        out_path=str(tmp_path) if tmp_path is not None else None,
        create_timestamp=False,
        log_enabled=False,
        **save_kwargs,
    )
    return mfbo.multi_fidelity_bayesian_optimization(
        f_low=_f_low,
        f_high=_f_high,
        bounds=[(-2.0, 2.0)],
        mfbo_cfg=mfbo.MFBOConfig(n_init_L=4, n_init_H=2, n_iter=10, random_state=0),
        gp_cfg=mfbo.GPConfig(),
        acq_cfg=mfbo.AcqConfig(),
        optim_cfg=mfbo.OptimConfig(n_raw_samples=200),
        fidelity_cfg=mfbo.FidelityConfig(cost_low=1.0, cost_high=5.0),
        save_cfg=save_cfg,
    )


# ---------------------------------------------------------------------
# Tier 1: trace is always on and cheap
# ---------------------------------------------------------------------

def test_trace_enabled_by_default_no_out_path():
    res = mfbo.multi_fidelity_bayesian_optimization(
        f_low=_f_low, f_high=_f_high,
        bounds=[(-2.0, 2.0)],
        mfbo_cfg=mfbo.MFBOConfig(n_init_L=4, n_init_H=2, n_iter=6, random_state=0),
        gp_cfg=mfbo.GPConfig(),
        acq_cfg=mfbo.AcqConfig(),
        optim_cfg=mfbo.OptimConfig(n_raw_samples=200),
        fidelity_cfg=mfbo.FidelityConfig(),
        save_cfg=mfbo.SaveConfig(),
    )
    assert res.trace is not None
    assert len(res.trace) == 6
    assert res.snapshots is None


def test_trace_written_and_reloadable(tmp_path):
    res = _run(tmp_path)
    trace_path = os.path.join(res.out_path, "res", "trace.npz")
    assert os.path.exists(trace_path)
    assert os.path.exists(os.path.join(res.out_path, "res", "trace.csv"))

    loaded = persistence.load_trace(trace_path)
    assert loaded.it == res.trace.it
    assert loaded.level_next == res.trace.level_next
    np.testing.assert_allclose(np.asarray(loaded.y_best), np.asarray(res.trace.y_best))
    np.testing.assert_allclose(np.asarray(loaded.x_next), np.asarray(res.trace.x_next))
    np.testing.assert_allclose(np.asarray(loaded.cumulative_cost), np.asarray(res.trace.cumulative_cost))


def test_trace_fields_match_spec(tmp_path):
    res = _run(tmp_path)
    trace = res.trace
    n = 10
    assert len(trace.it) == n
    for field_name in (
        "x_next", "level_next", "y_next", "x_best", "y_best",
        "wall_time", "acq_value", "cost_next", "cumulative_cost",
    ):
        assert len(getattr(trace, field_name)) == n

    assert all(lvl in ("L", "H") for lvl in trace.level_next)

    y_best = np.asarray(trace.y_best)
    assert np.all(np.diff(y_best) <= 1e-12)  # incumbent (high-fidelity) never worsens

    # GP hyperparameters recorded at every iteration
    for field_name in ("l_lf", "sigma_lf", "l_delta", "sigma_delta", "rho", "sigma_L", "sigma_H"):
        assert len(getattr(trace, field_name)) == n

    cum_cost = np.asarray(trace.cumulative_cost)
    assert np.all(np.diff(cum_cost) > 0.0)  # cost strictly accumulates every iteration
    expected_total = sum(
        mfbo.FidelityConfig(cost_low=1.0, cost_high=5.0).cost(lvl) for lvl in trace.level_next
    )
    assert cum_cost[-1] == pytest.approx(expected_total)


def test_trace_best_includes_current_iteration(tmp_path):
    """Row `it` must be the running high-fidelity minimum INCLUDING the point
    evaluated at iteration `it` (no one-iteration lag), and match the final best."""
    res = _run(tmp_path)
    trace = res.trace
    n_init_H = 2
    y_H0 = res.y_H[:n_init_H]
    y_best = np.asarray(trace.y_best)
    expected, cur = [], float(y_H0.min())
    for lvl, y in zip(trace.level_next, trace.y_next):
        if lvl == "H":
            cur = min(cur, y)
        expected.append(cur)
    np.testing.assert_allclose(y_best, expected)
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
        res = _run(tmp_path)
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
    assert [s.it for s in res.snapshots] == [0, 4, 8]


def test_snapshot_flush_every_clears_memory_buffer(tmp_path):
    res = _run(tmp_path, snapshot_enabled=True, snapshot_every=1, snapshot_flush_every=3)
    assert len(res.snapshots) == 1
    snap_dir = os.path.join(res.out_path, "res", "snapshots")
    assert len(os.listdir(snap_dir)) == 10


def test_meta_json_written(tmp_path):
    res = _run(tmp_path)
    meta = persistence.load_meta(os.path.join(res.out_path, "res"))
    assert meta is not None
    assert meta["mfbo_cfg"]["n_iter"] == 10
    assert meta["bounds"] == [[-2.0, 2.0]]


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
        mu_live, var_live = mfbo.mf_gp_predict(
            Xgrid, live_gp, live_gp.X_L_norm, live_gp.X_H_norm, live_gp.alpha, live_gp.L,
        )

        posterior = run.posterior(it)
        mu_loaded, var_loaded = posterior.predict(Xgrid)

        np.testing.assert_allclose(mu_loaded, mu_live, rtol=0, atol=1e-8)
        np.testing.assert_allclose(var_loaded, var_live, rtol=0, atol=1e-8)


def test_posterior_missing_iteration_raises(tmp_path):
    res = _run(tmp_path, snapshot_enabled=True, snapshot_every=5)  # its 0 and 5 only
    run = persistence.load_run(res.out_path)
    with pytest.raises(KeyError):
        run.posterior(1)


def test_reconstruct_gp_excludes_dense_factors():
    gp_cfg = mfbo.GPConfig()
    bounds = [(-2.0, 2.0)]
    gp = mfbo.build_gp_model(gp_cfg, bounds)
    X_L = np.array([[-1.0], [0.0], [1.0]])
    y_L = np.array([_f_low(x) for x in X_L])
    X_H = np.array([[0.5]])
    y_H = np.array([_f_high(x) for x in X_H])
    gp = mfbo.fit_gp(gp, X_L, y_L, X_H, y_H, gp_cfg, it=0)

    snap = persistence.build_gp_snapshot(0, gp)
    snap_fields = set(vars(snap).keys())
    assert "alpha" not in snap_fields
    assert "L" not in snap_fields
    assert "X_L_norm" not in snap_fields
    assert "X_H_norm" not in snap_fields

    rebuilt = persistence.reconstruct_gp(snap)
    np.testing.assert_allclose(rebuilt.alpha, gp.alpha, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rebuilt.L, gp.L, rtol=0, atol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
