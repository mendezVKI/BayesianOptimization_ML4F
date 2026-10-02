"""
Tests for the two-tier (trace / snapshot) persistence layer in
pyRAMBO.mobo.persistence.

Run with: pytest
"""

import os
from unittest.mock import patch

import numpy as np
import pytest

from pyRAMBO import mobo
from pyRAMBO.mobo import persistence


def _f(x):
    """Schaffer N.1: minimize x^2 and (x-2)^2."""
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.array([x[0] ** 2, (x[0] - 2.0) ** 2])


def _run(tmp_path=None, **save_kwargs):
    save_cfg = mobo.SaveConfig(
        out_path=str(tmp_path) if tmp_path is not None else None,
        create_timestamp=False,
        log_enabled=False,
        **save_kwargs,
    )
    return mobo.multi_objective_bayesian_optimization(
        f=_f,
        bounds=[(-4.0, 4.0)],
        mobo_cfg=mobo.MOBOConfig(n_init=4, n_iter=10, random_state=0),
        gp_cfg=mobo.GPConfig(),
        acq_cfg=mobo.AcqConfig(n_mc=64),
        optim_cfg=mobo.OptimConfig(n_raw_samples=200),
        save_cfg=save_cfg,
    )


# ---------------------------------------------------------------------
# Tier 1: trace is always on and cheap
# ---------------------------------------------------------------------

def test_trace_enabled_by_default_no_out_path():
    res = mobo.multi_objective_bayesian_optimization(
        f=_f,
        bounds=[(-4.0, 4.0)],
        mobo_cfg=mobo.MOBOConfig(n_init=4, n_iter=6, random_state=0),
        gp_cfg=mobo.GPConfig(),
        acq_cfg=mobo.AcqConfig(n_mc=64),
        optim_cfg=mobo.OptimConfig(n_raw_samples=200),
        save_cfg=mobo.SaveConfig(),
    )
    assert res.trace is not None
    assert len(res.trace) == 6


def test_trace_written_and_reloadable(tmp_path):
    res = _run(tmp_path)
    trace_path = os.path.join(res.out_path, "res", "trace.npz")
    assert os.path.exists(os.path.join(res.out_path, "res", "trace.csv"))
    assert os.path.exists(trace_path)

    loaded = persistence.load_trace(trace_path)
    assert len(loaded) == len(res.trace) == 10
    np.testing.assert_allclose(loaded.hypervolume, res.trace.hypervolume)
    np.testing.assert_allclose(np.asarray(loaded.x_next), np.asarray(res.trace.x_next))
    np.testing.assert_allclose(np.asarray(loaded.y_next), np.asarray(res.trace.y_next))
    assert loaded.n_pareto == res.trace.n_pareto


def test_trace_fields_match_spec(tmp_path):
    """A multi-objective trace records the whole objective vector plus the
    hypervolume and front size, and has no scalar-incumbent columns."""
    res = _run(tmp_path)
    trace = res.trace

    arrays = trace.to_arrays()
    assert set(arrays) == {
        "it", "x_next", "y_next", "hypervolume", "n_pareto", "wall_time", "acq_value",
        "length_scales", "B", "sigma_n",
    }
    assert "x_best" not in arrays
    assert "y_best" not in arrays

    assert arrays["y_next"].shape == (10, 2)
    assert arrays["x_next"].shape == (10, 1)
    assert list(trace.it) == list(range(10))

    # the recorded rows must be the ones the driver actually evaluated
    np.testing.assert_allclose(np.asarray(trace.x_next), res.X[4:])
    np.testing.assert_allclose(np.asarray(trace.y_next), res.Y[4:])

    # and the hypervolume column must match the per-iteration history
    np.testing.assert_allclose(trace.hypervolume, [h["hypervolume"] for h in res.history])
    assert list(trace.n_pareto) == [h["n_pareto"] for h in res.history]
    assert all(w >= 0.0 for w in trace.wall_time)


def test_trace_hypervolume_includes_current_iteration(tmp_path):
    """Row `it` must be the hypervolume / front size INCLUDING the point
    evaluated at iteration `it` (no one-iteration lag), and match the final
    result. The state keeps the pre-update value the acquisition used."""
    res = _run(tmp_path)
    n_init = len(res.Y) - len(res.trace)
    ref_z = -np.asarray(res.ref_point)          # default: minimize both objectives -> Z = -Y
    for k, (hv, n_par) in enumerate(zip(res.trace.hypervolume, res.trace.n_pareto)):
        Z = -res.Y[: n_init + k + 1]
        mask = mobo.nondominated_mask(Z)
        assert hv == pytest.approx(mobo.hypervolume(Z[mask], ref_z))
        assert n_par == int(mask.sum())
    assert res.trace.hypervolume[-1] == pytest.approx(res.hypervolume)
    for st in res.states:
        assert st.hypervolume_acq <= st.hypervolume + 1e-9


def test_trace_flush_every_requires_out_path():
    with pytest.raises(ValueError, match="out_path"):
        mobo.multi_objective_bayesian_optimization(
            f=_f,
            bounds=[(-4.0, 4.0)],
            mobo_cfg=mobo.MOBOConfig(n_init=4, n_iter=3, random_state=0),
            gp_cfg=mobo.GPConfig(),
            acq_cfg=mobo.AcqConfig(n_mc=64),
            optim_cfg=mobo.OptimConfig(n_raw_samples=100),
            save_cfg=mobo.SaveConfig(trace_flush_every=1),
        )


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
    assert len(os.listdir(snap_dir)) == 10


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
    assert meta["mobo_cfg"]["n_iter"] == 10
    assert meta["bounds"] == [[-4.0, 4.0]]
    # the resolved reference point is recorded, not just the (None) request
    np.testing.assert_allclose(meta["ref_point"], res.ref_point)


# ---------------------------------------------------------------------
# Round trip: reconstructed posterior must match the live GP
# ---------------------------------------------------------------------

def test_round_trip_posterior_matches_live_model(tmp_path):
    res = _run(tmp_path, snapshot_enabled=True, snapshot_every=1)

    run = persistence.load_run(res.out_path)
    assert run.available_iterations == list(range(10))

    Xgrid = np.linspace(-4.0, 4.0, 50).reshape(-1, 1)

    for it in (0, 4, 9):
        live_gp = res.states[it].gp
        mu_live, var_live = mobo.icm_gp_predict(Xgrid, live_gp)

        posterior = run.posterior(it)
        mu_loaded, var_loaded = posterior.predict(Xgrid)

        np.testing.assert_allclose(mu_loaded, mu_live, rtol=0, atol=1e-8)
        np.testing.assert_allclose(var_loaded, var_live, rtol=0, atol=1e-8)
        np.testing.assert_allclose(posterior.ref_point, res.ref_point)


def test_round_trip_preserves_the_output_covariance(tmp_path):
    """The coregionalization matrix is what makes this GP multi-output, so
    the joint per-point covariance must survive the round trip too."""
    res = _run(tmp_path, snapshot_enabled=True, snapshot_every=1)
    run = persistence.load_run(res.out_path)

    Xgrid = np.linspace(-4.0, 4.0, 20).reshape(-1, 1)
    _, cov_live = mobo.icm_gp_predict(Xgrid, res.states[7].gp, return_cov=True)
    _, cov_loaded = run.posterior(7).predict(Xgrid, return_cov=True)

    np.testing.assert_allclose(cov_loaded, cov_live, rtol=0, atol=1e-10)


def test_posterior_missing_iteration_raises(tmp_path):
    res = _run(tmp_path, snapshot_enabled=True, snapshot_every=5)  # its 0 and 5 only
    run = persistence.load_run(res.out_path)
    with pytest.raises(KeyError):
        run.posterior(1)


def test_reconstruct_gp_excludes_dense_factors():
    gp_cfg = mobo.GPConfig()
    bounds = [(-4.0, 4.0)]
    gp = mobo.build_gp_model(gp_cfg, bounds, 2)
    X = np.array([[-1.0], [0.0], [1.0], [2.5]])
    Y = mobo.evaluate_objective(_f, X)
    gp = mobo.fit_gp(gp, X, Y, gp_cfg, it=0)

    snap = persistence.build_gp_snapshot(0, gp, np.array([20.0, 30.0]))
    snap_fields = set(vars(snap).keys())
    assert "alpha" not in snap_fields
    assert "L" not in snap_fields
    assert "X_norm" not in snap_fields
    assert "Y_norm" not in snap_fields

    rebuilt = persistence.reconstruct_gp(snap)
    np.testing.assert_allclose(rebuilt.alpha, gp.alpha, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rebuilt.L, gp.L, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rebuilt.B, gp.B, rtol=0, atol=1e-12)


def test_reconstruct_gp_rejects_an_unknown_kernel():
    gp_cfg = mobo.GPConfig()
    gp = mobo.build_gp_model(gp_cfg, [(-4.0, 4.0)], 2)
    X = np.array([[-1.0], [0.0], [1.0]])
    gp = mobo.fit_gp(gp, X, mobo.evaluate_objective(_f, X), gp_cfg, it=0)

    snap = persistence.build_gp_snapshot(0, gp, np.zeros(2))
    snap.kernel = "something_else"
    with pytest.raises(ValueError, match="kernel identifier"):
        persistence.reconstruct_gp(snap)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
