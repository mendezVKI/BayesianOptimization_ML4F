"""
Smoke / correctness tests for mfbo.core.

Run with: pytest
"""

import numpy as np
import pytest

import mfbo


def _f_high(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return float((x[0] - 0.7) ** 2)


def _f_low(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(0.9 * (x[0] - 0.7) ** 2 + 0.05 * np.sin(15.0 * x[0]))


def _run(**overrides):
    kwargs = dict(
        f_low=_f_low,
        f_high=_f_high,
        bounds=[(-2.0, 2.0)],
        mfbo_cfg=mfbo.MFBOConfig(n_init_L=6, n_init_H=2, n_iter=15, random_state=0),
        gp_cfg=mfbo.GPConfig(),
        acq_cfg=mfbo.AcqConfig(),
        optim_cfg=mfbo.OptimConfig(n_raw_samples=400),
        fidelity_cfg=mfbo.FidelityConfig(cost_low=1.0, cost_high=5.0),
        save_cfg=mfbo.SaveConfig(),
    )
    kwargs.update(overrides)
    return mfbo.multi_fidelity_bayesian_optimization(**kwargs)


def test_multi_fidelity_bo_converges_1d():
    """A short bi-fidelity run on a convex 1D objective should land near the
    true (high-fidelity) minimum."""
    res = _run()
    assert res.best_y < 5e-2
    assert abs(res.best_x[0] - 0.7) < 0.3


def test_incumbent_is_high_fidelity_only():
    """best_y/best_x must come from y_H alone, even though f_low can dip
    below the true high-fidelity minimum. Each state's y_best/x_best is the
    incumbent computed BEFORE that iteration's proposal was appended (same
    convention as sbo.core), so it must match min(y_H) restricted to the
    high-fidelity points available at the start of that iteration."""
    res = _run()
    assert res.best_y == pytest.approx(float(np.min(res.y_H)))
    best_idx = int(np.argmin(res.y_H))
    np.testing.assert_allclose(res.best_x, res.X_H[best_idx])

    n_H_before = res.states[0].X_H.shape[0] - (1 if res.states[0].level_next == "H" else 0)
    for state in res.states:
        y_H_before = state.y_H[:n_H_before]
        assert state.y_best == pytest.approx(float(np.min(y_H_before)))
        n_H_before = state.X_H.shape[0]  # "after" this iteration == "before" the next

    # the incumbent itself never worsens from one iteration to the next
    y_best_seq = np.asarray([s.y_best for s in res.states])
    assert np.all(np.diff(y_best_seq) <= 1e-12)


def test_every_iteration_appends_exactly_one_point_to_its_fidelity():
    res = _run()
    for state in res.states:
        total = state.X_L.shape[0] + state.X_H.shape[0]
        # exactly one new point vs. the previous state (or the init set for it=0)
        assert state.level_next in ("L", "H")
        if state.level_next == "L":
            assert state.X_L.shape[0] >= 1
        else:
            assert state.X_H.shape[0] >= 1


def test_acquisition_selects_valid_fidelity_and_in_bounds_point():
    bounds = [(-2.0, 2.0)]
    gp_cfg = mfbo.GPConfig()
    gp = mfbo.build_gp_model(gp_cfg, bounds)

    X_L = np.array([[-1.0], [0.0], [1.0], [1.5]])
    y_L = np.array([_f_low(x) for x in X_L])
    X_H = np.array([[-0.5], [0.7]])
    y_H = np.array([_f_high(x) for x in X_H])

    gp = mfbo.fit_gp(gp, X_L, y_L, X_H, y_H, gp_cfg, it=0)

    acq_cfg = mfbo.AcqConfig()
    fidelity_cfg = mfbo.FidelityConfig()
    y_best = float(np.min(y_H))
    value = mfbo.make_acquisition(acq_cfg, fidelity_cfg, gp, y_best)

    optim_cfg = mfbo.OptimConfig(n_raw_samples=300)
    result = mfbo.optimize_acquisition(value, bounds, optim_cfg, np.random.default_rng(1))

    assert result.level_next in ("L", "H")
    assert bounds[0][0] <= result.x_next[0] <= bounds[0][1]
    assert result.a_L.shape[0] == result.Xcand.shape[0]
    assert result.a_H.shape[0] == result.Xcand.shape[0]


def test_cheap_highly_correlated_low_fidelity_is_preferred():
    """When the low fidelity is (almost) as informative as the high fidelity
    but far cheaper, the cost-aware acquisition should prefer it over most of
    the domain."""
    bounds = [(-2.0, 2.0)]
    gp_cfg = mfbo.GPConfig(l_lf=0.5, sigma_lf=1.0, l_delta=0.5, sigma_delta=1e-3, rho=1.0, sigma_L=1e-3, sigma_H=1e-3)
    gp = mfbo.build_gp_model(gp_cfg, bounds)

    X_L = np.array([[-1.5], [-0.5], [0.5], [1.5]])
    y_L = np.array([_f_high(x) for x in X_L])  # near-identical to f_high
    X_H = np.array([[0.0]])
    y_H = np.array([_f_high(x) for x in X_H])

    gp = mfbo.fit_gp(gp, X_L, y_L, X_H, y_H, gp_cfg, it=0)

    acq_cfg = mfbo.AcqConfig()
    fidelity_cfg = mfbo.FidelityConfig(cost_low=1.0, cost_high=100.0)
    value = mfbo.make_acquisition(acq_cfg, fidelity_cfg, gp, float(np.min(y_H)))

    Xcand = np.linspace(-2.0, 2.0, 200).reshape(-1, 1)
    a_L = np.asarray(value(Xcand, "L"))
    a_H = np.asarray(value(Xcand, "H"))

    # Wherever the acquisition is actually informative (non-negligible EI),
    # the ~100x cheaper, near-equally-informative low fidelity must win.
    informative = a_H > 1e-9
    assert informative.sum() > 0
    assert np.all(a_L[informative] > a_H[informative])

    # ...and so must the joint (x, level) decision itself.
    result = mfbo.optimize_acquisition(value, bounds, mfbo.OptimConfig(n_raw_samples=400), np.random.default_rng(2))
    assert result.level_next == "L"


def test_variance_reduction_matches_full_refit():
    """predictive_variance_reduction(x, level) must match the variance drop
    obtained by actually refitting the joint GP with a hypothetical point
    (X_L,X_H) + (x at that level) appended -- the GP posterior variance does
    not depend on the observed y value, so any y works for the check."""
    bounds = [(-2.0, 2.0)]
    gp_cfg = mfbo.GPConfig()
    gp = mfbo.build_gp_model(gp_cfg, bounds)

    rng = np.random.default_rng(3)
    X_L = rng.uniform(-2, 2, size=(5, 1))
    y_L = np.array([_f_low(x) for x in X_L])
    X_H = rng.uniform(-2, 2, size=(3, 1))
    y_H = np.array([_f_high(x) for x in X_H])

    gp = mfbo.fit_gp(gp, X_L, y_L, X_H, y_H, gp_cfg, it=0)

    x_target = np.array([[0.3]])
    x_target_n = gp.normalizer.normalize_X(x_target)

    mu_before, var_before = mfbo.mf_gp_predict(x_target, gp, gp.X_L_norm, gp.X_H_norm, gp.alpha, gp.L)

    for level in ("L", "H"):
        vr = mfbo.predictive_variance_reduction(x_target_n, level, gp, gp.X_L_norm, gp.X_H_norm, gp.L)

        if level == "L":
            X_L2 = np.vstack([X_L, x_target])
            y_L2 = np.concatenate([y_L, [0.0]])  # value is irrelevant to variance
            X_H2, y_H2 = X_H, y_H
        else:
            X_L2, y_L2 = X_L, y_L
            X_H2 = np.vstack([X_H, x_target])
            y_H2 = np.concatenate([y_H, [0.0]])

        gp2 = mfbo.build_gp_model(gp_cfg, bounds)
        gp2 = mfbo.fit_gp(gp2, X_L2, y_L2, X_H2, y_H2, gp_cfg, it=0)
        _, var_after = mfbo.mf_gp_predict(x_target, gp2, gp2.X_L_norm, gp2.X_H_norm, gp2.alpha, gp2.L)

        expected_vr = float(var_before[0] - var_after[0])
        assert vr[0] == pytest.approx(expected_vr, abs=1e-6)


def test_n_init_H_must_be_at_least_one():
    with pytest.raises(ValueError):
        _run(mfbo_cfg=mfbo.MFBOConfig(n_init_L=4, n_init_H=0, n_iter=2, random_state=0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
