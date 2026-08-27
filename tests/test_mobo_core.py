"""
Smoke / correctness tests for mobo.core.

Run with: pytest
"""

import numpy as np
import pytest

import mobo


def _f(x):
    """Schaffer N.1: minimize x^2 and (x-2)^2. Pareto set is x in [0, 2]."""
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.array([x[0] ** 2, (x[0] - 2.0) ** 2])


def _run(**overrides):
    kwargs = dict(
        f=_f,
        bounds=[(-4.0, 4.0)],
        mobo_cfg=mobo.MOBOConfig(n_init=6, n_iter=12, random_state=0),
        gp_cfg=mobo.GPConfig(),
        acq_cfg=mobo.AcqConfig(n_mc=128),
        optim_cfg=mobo.OptimConfig(n_raw_samples=400),
        save_cfg=mobo.SaveConfig(),
    )
    kwargs.update(overrides)
    return mobo.multi_objective_bayesian_optimization(**kwargs)


#%% ---------------------------------------------------------------------
# Pareto dominance and hypervolume
# -------------------------------------------------------------------------

def test_nondominated_mask_maximization():
    Z = np.array([[1.0, 5.0], [2.0, 3.0], [0.0, 0.0], [2.0, 5.0]])
    mask = mobo.nondominated_mask(Z)
    # [2,5] dominates everything else here
    np.testing.assert_array_equal(mask, [False, False, False, True])


def test_nondominated_mask_keeps_duplicates():
    """Identical points do not dominate each other under the strict rule."""
    Z = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert mobo.nondominated_mask(Z).all()


def test_pareto_front_keeps_the_whole_staircase():
    """Regression guard on the legacy bug: sorting by f1 ascending and
    keeping *increasing* f2 collapsed every front to one point."""
    Z = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 1.0], [0.5, 0.5]])
    P = mobo.pareto_front(Z)
    assert P.shape[0] == 3
    # sorted by f1 ascending => f2 strictly decreasing
    assert np.all(np.diff(P[:, 0]) > 0)
    assert np.all(np.diff(P[:, 1]) < 0)


def test_hypervolume_known_values():
    ref = np.array([0.0, 0.0])
    assert mobo.hypervolume(np.array([[2.0, 3.0]]), ref) == pytest.approx(6.0)
    # two-point staircase: 6 + the 2x1 strip added by (4,1)
    assert mobo.hypervolume(np.array([[2.0, 3.0], [4.0, 1.0]]), ref) == pytest.approx(8.0)
    # same front, given in the other order
    assert mobo.hypervolume(np.array([[1.0, 5.0], [2.0, 3.0]]), ref) == pytest.approx(8.0)


def test_hypervolume_ignores_points_below_the_reference():
    ref = np.array([1.0, 1.0])
    assert mobo.hypervolume(np.array([[0.5, 0.5]]), ref) == pytest.approx(0.0)
    assert mobo.hypervolume(np.zeros((0, 2)), ref) == pytest.approx(0.0)


def test_hypervolume_improvement_matches_recomputing_the_hypervolume():
    """HVI(z) must be exactly hv(front + z) - hv(front), for any z: the
    acquisition and the reported progress measure have to agree."""
    rng = np.random.default_rng(0)
    ref = np.array([-3.0, -3.0])

    for _ in range(20):
        m = int(rng.integers(0, 6))
        P = rng.normal(size=(m, 2)) * 2.0 if m else np.zeros((0, 2))
        Zq = rng.normal(size=(10, 2)) * 2.0

        hvi = mobo.hypervolume_improvement(Zq, P, ref)
        hv_before = mobo.hypervolume(P, ref)
        for k in range(Zq.shape[0]):
            stacked = np.vstack([P, Zq[k][None, :]])
            expected = mobo.hypervolume(stacked, ref) - hv_before
            assert hvi[k] == pytest.approx(expected, abs=1e-9)


def test_hypervolume_improvement_is_non_negative():
    rng = np.random.default_rng(1)
    P = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    Zq = rng.normal(size=(50, 2)) * 3.0
    assert np.all(mobo.hypervolume_improvement(Zq, P, np.array([-1.0, -1.0])) >= 0.0)


#%% ---------------------------------------------------------------------
# ICM GP
# -------------------------------------------------------------------------

def test_icm_gp_interpolates_noiseless_training_data():
    """With tiny noise the posterior mean must reproduce the observations."""
    rng = np.random.default_rng(2)
    bounds = [(0.0, 1.0)]
    X = rng.uniform(0.0, 1.0, size=(8, 1))
    Y = np.column_stack([np.sin(6 * X[:, 0]), np.cos(6 * X[:, 0])])

    gp_cfg = mobo.GPConfig(length_scales=0.25, sigma_n=1e-4)
    gp = mobo.fit_gp(mobo.build_gp_model(gp_cfg, bounds, 2), X, Y, gp_cfg, it=0)

    mu, var = mobo.icm_gp_predict(X, gp, return_cov=False)
    np.testing.assert_allclose(mu, Y, atol=1e-3)
    assert np.all(var >= 0.0)
    assert np.all(var < 1e-2)


def test_icm_gp_covariance_diagonal_matches_variance():
    rng = np.random.default_rng(3)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    X = rng.uniform(size=(7, 2))
    Y = np.column_stack([X[:, 0] + X[:, 1], X[:, 0] - 2 * X[:, 1]])

    gp_cfg = mobo.GPConfig(length_scales=0.4, L_B_offdiag=-0.5)
    gp = mobo.fit_gp(mobo.build_gp_model(gp_cfg, bounds, 2), X, Y, gp_cfg, it=0)

    Xt = rng.uniform(size=(5, 2))
    _, var = mobo.icm_gp_predict(Xt, gp, return_cov=False)
    _, cov = mobo.icm_gp_predict(Xt, gp, return_cov=True)

    np.testing.assert_allclose(var, np.diagonal(cov, axis1=1, axis2=2), atol=1e-12)
    # every per-point block must be symmetric positive semi-definite
    np.testing.assert_allclose(cov, np.swapaxes(cov, 1, 2), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-10)


def test_build_B_is_psd_and_theta_round_trips():
    for params in ([0.0, 0.0, 0.0], [1.0, -3.0, 0.5], [-2.0, 7.0, -1.0]):
        B = mobo.build_B(np.asarray(params), 2)
        assert np.all(np.linalg.eigvalsh(B) >= -1e-12)
        np.testing.assert_allclose(B, B.T)

    length_scales = np.array([0.3, 1.7])
    B = mobo.build_B(np.array([0.2, -0.6, 0.1]), 2)
    sigma_n = np.array([0.05, 0.2])

    theta = mobo.pack_theta(length_scales, B, sigma_n)
    ls2, B2, sn2 = mobo.unpack_theta(theta, 2, 2)

    np.testing.assert_allclose(ls2, length_scales)
    np.testing.assert_allclose(B2, B, atol=1e-9)
    np.testing.assert_allclose(sn2, sigma_n)


def test_pack_unpack_Y_round_trip():
    Y = np.arange(12, dtype=float).reshape(6, 2)
    np.testing.assert_allclose(mobo.unpack_Y(mobo.pack_Y(Y), 6), Y)


def test_hyperparameter_optimization_improves_the_likelihood():
    rng = np.random.default_rng(4)
    bounds = [(0.0, 1.0)]
    X = rng.uniform(size=(12, 1))
    Y = np.column_stack([np.sin(4 * X[:, 0]), -np.sin(4 * X[:, 0]) + 0.2 * X[:, 0]])

    gp_cfg = mobo.GPConfig(length_scales=2.0, sigma_n=0.5)
    gp = mobo.fit_gp(mobo.build_gp_model(gp_cfg, bounds, 2), X, Y, gp_cfg, it=0)

    theta0 = mobo.pack_theta(gp.length_scales, gp.B, gp.sigma_n)
    nll0 = mobo.negative_log_marginal_likelihood(theta0, gp.X_norm, gp.Y_norm, 2, gp.jitter)

    gp_cfg_hpo = mobo.GPConfig(length_scales=2.0, sigma_n=0.5, optimize_hyperparams=True)
    gp2 = mobo.fit_gp(mobo.build_gp_model(gp_cfg_hpo, bounds, 2), X, Y, gp_cfg_hpo, it=0)
    nll1 = mobo.negative_log_marginal_likelihood(gp2.theta, gp2.X_norm, gp2.Y_norm, 2, gp2.jitter)

    assert nll1 < nll0


def test_hyperparameter_optimization_stays_finite_without_user_bounds():
    """Regression guard: an unbounded L-BFGS-B run on the ICM likelihood
    used to walk a log-scale into exp() overflow and abort the run."""
    rng = np.random.default_rng(5)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    X = rng.uniform(size=(10, 2))
    Y = np.column_stack([X[:, 0] * 1e6, X[:, 1] * 1e-6])  # wildly different scales

    gp_cfg = mobo.GPConfig(optimize_hyperparams=True)
    gp = mobo.fit_gp(mobo.build_gp_model(gp_cfg, bounds, 2), X, Y, gp_cfg, it=0)

    assert np.all(np.isfinite(gp.length_scales))
    assert np.all(np.isfinite(gp.B))
    assert np.all(np.isfinite(gp.sigma_n))
    assert np.all(np.isfinite(gp.alpha))


#%% ---------------------------------------------------------------------
# Acquisition
# -------------------------------------------------------------------------

def _fitted_gp_and_front(seed=6):
    rng = np.random.default_rng(seed)
    bounds = [(-4.0, 4.0)]
    X = rng.uniform(-4.0, 4.0, size=(8, 1))
    Y = mobo.evaluate_objective(_f, X)

    gp_cfg = mobo.GPConfig(length_scales=0.2, sigma_n=0.01)
    gp = mobo.fit_gp(mobo.build_gp_model(gp_cfg, bounds, 2), X, Y, gp_cfg, it=0)

    signs = -np.ones(2)  # both minimized
    Z = signs * Y
    mask = mobo.nondominated_mask(Z)
    ref_z = np.min(Z, axis=0) - 0.1 * np.ptp(Z, axis=0)
    return bounds, gp, Z[mask], ref_z, signs


def test_acquisition_is_non_negative_and_deterministic():
    """Common random numbers: two calls on the same points must return
    exactly the same values, otherwise the L-BFGS-B refinement would be
    differentiating Monte-Carlo noise."""
    bounds, gp, front, ref_z, signs = _fitted_gp_and_front()
    acq = mobo.make_acquisition(mobo.AcqConfig(n_mc=128), gp, front, ref_z, signs,
                                np.random.default_rng(0))

    Xq = np.linspace(-4.0, 4.0, 50).reshape(-1, 1)
    a1 = acq(Xq)
    a2 = acq(Xq)

    assert np.all(a1 >= 0.0)
    np.testing.assert_array_equal(a1, a2)


def test_acquisition_matches_the_naive_per_sample_definition():
    """The vectorized closed-form EHVI must equal the textbook definition --
    draw from the posterior, recompute the hypervolume with each draw added,
    average the differences -- evaluated on exactly the same draws."""
    bounds, gp, front, ref_z, signs = _fitted_gp_and_front()

    n_mc = 128
    acq = mobo.make_acquisition(mobo.AcqConfig(n_mc=n_mc), gp, front, ref_z, signs,
                                np.random.default_rng(3))

    # replicate the antithetic common random numbers the acquisition drew
    half = np.random.default_rng(3).standard_normal((n_mc // 2, 2))
    Z_mc = np.vstack([half, -half])

    Xq = np.linspace(-3.5, 3.5, 7).reshape(-1, 1)
    mu, cov = mobo.icm_gp_predict(Xq, gp, return_cov=True)
    F = mobo.core._psd_factor(cov)

    hv_before = mobo.hypervolume(front, ref_z)
    expected = []
    for k in range(Xq.shape[0]):
        samples = mu[k] + Z_mc @ F[k].T
        gains = [
            mobo.hypervolume(np.vstack([front, (signs * s)[None, :]]), ref_z) - hv_before
            for s in samples
        ]
        expected.append(np.mean(gains))

    np.testing.assert_allclose(acq(Xq), expected, rtol=1e-9, atol=1e-12)


def test_acquisition_is_symmetric_under_a_sign_flip():
    """Minimizing f and maximizing -f must give the same acquisition
    surface, not merely the same expectation: the antithetic draws make the
    Monte-Carlo estimator itself sign-symmetric."""
    rng = np.random.default_rng(7)
    bounds = [(-4.0, 4.0)]
    X = rng.uniform(-4.0, 4.0, size=(8, 1))
    Y = mobo.evaluate_objective(_f, X)
    Xq = np.linspace(-4.0, 4.0, 40).reshape(-1, 1)

    values = []
    for sign, maximize in ((1.0, False), (-1.0, True)):
        Ys = sign * Y
        gp_cfg = mobo.GPConfig(length_scales=0.2, sigma_n=0.01)
        gp = mobo.fit_gp(mobo.build_gp_model(gp_cfg, bounds, 2), X, Ys, gp_cfg, it=0)

        signs = mobo.core._objective_signs(maximize, 2)
        Z = signs * Ys
        mask = mobo.nondominated_mask(Z)
        ref_z = np.min(Z, axis=0) - 0.1 * np.ptp(Z, axis=0)

        acq = mobo.make_acquisition(mobo.AcqConfig(n_mc=256, maximize=maximize), gp,
                                    Z[mask], ref_z, signs, np.random.default_rng(0))
        values.append(acq(Xq))

    np.testing.assert_allclose(values[0], values[1], atol=1e-9)


def test_acquisition_rejects_more_than_two_objectives():
    rng = np.random.default_rng(8)
    bounds = [(0.0, 1.0)]
    X = rng.uniform(size=(6, 1))
    Y = rng.normal(size=(6, 3))

    gp_cfg = mobo.GPConfig()
    gp = mobo.fit_gp(mobo.build_gp_model(gp_cfg, bounds, 3), X, Y, gp_cfg, it=0)

    with pytest.raises(ValueError, match="2 objectives"):
        mobo.make_acquisition(mobo.AcqConfig(), gp, np.zeros((0, 2)), np.zeros(2),
                              -np.ones(3), np.random.default_rng(0))


#%% ---------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

def test_mobo_finds_the_pareto_set():
    """Every proposed Pareto-optimal design must land in the true Pareto set
    x in [0, 2], and the front must actually spread along it rather than
    collapsing to one point."""
    res = _run()

    assert res.pareto_X.shape[0] >= 4
    assert np.all(res.pareto_X[:, 0] > -0.1)
    assert np.all(res.pareto_X[:, 0] < 2.1)
    assert np.ptp(res.pareto_X[:, 0]) > 1.0


def test_hypervolume_is_monotonically_non_decreasing():
    """With a FIXED reference point the dominated hypervolume can only grow:
    adding a point never removes dominated volume."""
    res = _run()
    hv = [h["hypervolume"] for h in res.history]
    assert all(hv[i] <= hv[i + 1] + 1e-9 for i in range(len(hv) - 1))
    assert res.hypervolume >= hv[-1] - 1e-9


def test_every_iteration_appends_exactly_one_point():
    res = _run()
    n_init = 6
    assert res.X.shape[0] == n_init + 12
    assert res.Y.shape == (n_init + 12, 2)
    for k, state in enumerate(res.states):
        assert state.X.shape[0] == n_init + k + 1


def test_pareto_set_and_front_are_consistent():
    """result.pareto_X / pareto_Y must be matching rows of X / Y, and be
    exactly the nondominated subset."""
    res = _run()

    for x, y in zip(res.pareto_X, res.pareto_Y):
        idx = np.flatnonzero(np.all(np.isclose(res.X, x), axis=1))
        assert idx.size >= 1
        assert np.any(np.all(np.isclose(res.Y[idx], y), axis=1))

    mask = mobo.nondominated_mask(-res.Y)
    np.testing.assert_allclose(res.pareto_Y, res.Y[mask])


def test_proposals_stay_inside_the_bounds():
    res = _run()
    for state in res.states:
        assert -4.0 <= state.x_next[0] <= 4.0


def test_run_is_reproducible_from_the_seed():
    a = _run()
    b = _run()
    np.testing.assert_allclose(a.X, b.X)
    np.testing.assert_allclose(a.Y, b.Y)
    assert a.hypervolume == pytest.approx(b.hypervolume)


def test_reference_point_is_reported_in_physical_units():
    """Objectives are minimized here, so the reference point must sit ABOVE
    the worst observed value of each one."""
    res = _run()
    assert np.all(res.ref_point > np.max(res.Y[:6], axis=0) - 1e-9)


def test_user_reference_point_is_honoured():
    ref = np.array([25.0, 40.0])
    res = _run(acq_cfg=mobo.AcqConfig(n_mc=128, ref_point=ref))
    np.testing.assert_allclose(res.ref_point, ref)


def test_maximization_flag_mirrors_the_run():
    """Minimizing f and maximizing -f must explore the same designs."""
    a = _run()
    b = _run(f=lambda x: -_f(x), acq_cfg=mobo.AcqConfig(n_mc=128, maximize=True))
    np.testing.assert_allclose(a.X, b.X)
    np.testing.assert_allclose(a.Y, -b.Y)


def test_mixed_objective_directions_are_accepted():
    """One objective minimized, the other maximized."""
    res = _run(
        f=lambda x: np.array([_f(x)[0], -_f(x)[1]]),
        acq_cfg=mobo.AcqConfig(n_mc=128, maximize=[False, True]),
    )
    assert np.all(res.pareto_X[:, 0] > -0.1)
    assert np.all(res.pareto_X[:, 0] < 2.1)


def test_more_than_two_objectives_is_rejected_by_the_driver():
    with pytest.raises(ValueError, match="2 objectives"):
        _run(f=lambda x: np.array([x[0] ** 2, (x[0] - 2.0) ** 2, x[0]]))


def test_refined_optimizer_does_not_worsen_the_acquisition():
    """The local refinement can only accept a strictly better point than the
    global scan found."""
    for state in _run(optim_cfg=mobo.OptimConfig(method="refined", n_raw_samples=300, n_restarts=4)).states:
        assert state.acq_res.a_best >= np.max(state.acq_res.a) - 1e-12


def test_too_few_initial_points_raises():
    with pytest.raises(ValueError, match="two initial points"):
        _run(mobo_cfg=mobo.MOBOConfig(n_init=1, n_iter=2, random_state=0))
