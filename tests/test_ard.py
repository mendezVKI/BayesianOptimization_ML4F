# -*- coding: utf-8 -*-
"""
Tests for Automatic Relevance Determination (ARD) in BO_ML4F.

The suite is organised around two promises made by the ARD patch:

  1. BACKWARDS COMPATIBILITY.  With a scalar length scale, every routine takes
     the pre-ARD code path and returns pre-ARD numbers.  The legacy kernel
     formulas are frozen verbatim at the top of this file and compared with
     `assert_array_equal` -- exact equality, not `allclose` -- so that any
     future refactor of the isotropic path is caught immediately.

  2. CORRECTNESS OF THE ARD PATH.  The anisotropic kernel is the isotropic
     kernel on rescaled inputs; HPO recovers known length scales; and ARD
     genuinely predicts better than an isotropic kernel on data that is
     genuinely anisotropic.

Run with:  pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

import BO_ML4F as bo


# ----------------------------------------------------------------------------
# Frozen copies of the pre-ARD kernels.  DO NOT "simplify" these -- they are the
# reference the isotropic path is held against.
# ----------------------------------------------------------------------------

def _legacy_rbf(X1, X2, l_c, sigma_f):
    from BO_func_YL import rbf_kernel_
    gamma = 0.5 / (l_c ** 2)
    return (sigma_f ** 2) * rbf_kernel_(X1, X2, gamma=gamma)


def _legacy_matern52(X1, X2, l_c, sigma_f):
    sqdist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
    r = np.sqrt(np.maximum(sqdist, 0.0))
    a = np.sqrt(5.0) * r / l_c
    return (sigma_f ** 2) * (1.0 + a + a ** 2 / 3.0) * np.exp(-a)


LEGACY = {"rbf": _legacy_rbf, "matern52": _legacy_matern52}


# ----------------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------------

D = 4
ANISOTROPIC_BOUNDS = [
    (np.log(0.02), np.log(50.0)),   # length scale(s)
    (np.log(0.10), np.log(10.0)),   # sigma_f
    (np.log(1e-3), np.log(2.00)),   # sigma_y
]


def anisotropic_dataset(n=140, d=D, noise=0.02, seed=0):
    """Coordinate 0 varies fast, coordinate 2 slowly, 1 and 3 are inert."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    y = np.sin(6.0 * X[:, 0]) + 0.7 * X[:, 2]
    y = y + noise * rng.standard_normal(n)
    return X, y


def two_point_sets(n1=5, n2=7, d=D, seed=11):
    rng = np.random.default_rng(seed)
    return rng.random((n1, d)), rng.random((n2, d))


# ============================================================================
# 1. Backwards compatibility -- the isotropic path is untouched
# ============================================================================

@pytest.mark.parametrize("kind", ["rbf", "matern52"])
def test_scalar_length_scale_is_bit_identical_to_legacy(kind):
    X1, X2 = two_point_sets()
    for l_c in (0.13, 0.9, 3.7):
        got = bo.kernel_amp(X1, X2, l_c=l_c, sigma_f=1.7, kind=kind)
        want = LEGACY[kind](X1, X2, l_c, 1.7)
        np.testing.assert_array_equal(got, want)


def test_default_config_stays_scalar_and_isotropic():
    cfg = bo.GPConfig()
    assert cfg.ard is False
    assert isinstance(cfg.l_c, float)
    gp = bo.build_gp_model(cfg)
    assert isinstance(gp.l_c, float)
    assert bo.is_ard(gp.l_c) is False


def test_isotropic_fit_keeps_a_float_length_scale_and_3_hyperparameters():
    X, y = anisotropic_dataset()
    cfg = bo.GPConfig(kernel="matern52", normalize_y=True,
                      optimize_hyperparams=True, hpo_every=1,
                      theta_bounds_log=ANISOTROPIC_BOUNDS, n_hpo_restarts=2,
                      hpo_random_state=1)
    gp = bo.fit_gp(bo.build_gp_model(cfg), X, y, cfg, it=0)
    assert isinstance(gp.l_c, float)
    assert gp.theta_log.size == 3


def test_param_hist_layout_unchanged_for_isotropic():
    X, y = anisotropic_dataset(n=40)
    hist = []
    theta = np.log([0.5, 1.0, 0.1])
    bo.negative_log_marginal_likelihood(theta, X, y, param_hist=hist,
                                        kernel="matern52")
    assert len(hist[0]) == 3


def test_legacy_batch_ei_suite_still_passes():
    """The pre-existing hybrid/batch-EI regression file must be unaffected."""
    import test_batch_ei_adam as legacy
    legacy.test_default_is_backward_compatible()
    legacy.test_batch_ei_without_gradient()
    legacy.test_close_pair_adds_only_final_point()


# ============================================================================
# 2. Kernel algebra
# ============================================================================

@pytest.mark.parametrize("kind", ["rbf", "matern52"])
def test_ard_with_equal_length_scales_matches_isotropic(kind):
    X1, X2 = two_point_sets()
    l = 0.83
    iso = bo.kernel_amp(X1, X2, l_c=l, sigma_f=1.3, kind=kind)
    ard = bo.kernel_amp(X1, X2, l_c=np.full(D, l), sigma_f=1.3, kind=kind)
    np.testing.assert_allclose(ard, iso, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("kind", ["rbf", "matern52"])
def test_ard_equals_isotropic_on_rescaled_inputs(kind):
    """The identity the frozen-whitening route relies on: k_ARD(x, x'; l)
    == k_iso(x/l, x'/l; 1).  If this ever breaks, the RT whitening plan and
    the library's ARD stop being the same model."""
    X1, X2 = two_point_sets()
    ell = np.array([0.2, 1.0, 3.0, 0.7])
    ard = bo.kernel_amp(X1, X2, l_c=ell, sigma_f=2.1, kind=kind)
    iso = bo.kernel_amp(X1 / ell, X2 / ell, l_c=1.0, sigma_f=2.1, kind=kind)
    np.testing.assert_allclose(ard, iso, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("kind", ["rbf", "matern52"])
def test_ard_diagonal_is_sigma_f_squared_and_matrix_is_psd(kind):
    X, _ = two_point_sets(n1=9)
    ell = np.array([0.15, 2.0, 0.6, 5.0])
    K = bo.kernel_amp(X, X, l_c=ell, sigma_f=1.4, kind=kind)
    np.testing.assert_allclose(np.diag(K), 1.4 ** 2, rtol=1e-12)
    np.testing.assert_allclose(K, K.T, rtol=1e-12)
    eigenvalues = np.linalg.eigvalsh(K)
    assert eigenvalues.min() > -1e-8


@pytest.mark.parametrize("kind", ["rbf", "matern52"])
def test_a_short_length_scale_decorrelates_that_coordinate_only(kind):
    """Moving along a short-scale coordinate must drop the correlation far
    more than the same step along a long-scale one."""
    ell = np.array([0.1, 10.0, 10.0, 10.0])
    x = np.zeros((1, D))
    step_relevant = np.array([[0.3, 0.0, 0.0, 0.0]])
    step_inert = np.array([[0.0, 0.3, 0.0, 0.0]])
    k_rel = bo.kernel_amp(x, step_relevant, l_c=ell, sigma_f=1.0, kind=kind)[0, 0]
    k_inert = bo.kernel_amp(x, step_inert, l_c=ell, sigma_f=1.0, kind=kind)[0, 0]
    assert k_rel < 0.2
    assert k_inert > 0.99


def test_ard_with_one_dimension_degenerates_to_isotropic():
    X1, X2 = two_point_sets(d=1)
    iso = bo.kernel_amp(X1, X2, l_c=0.4, sigma_f=1.0, kind="matern52")
    ard = bo.kernel_amp(X1, X2, l_c=np.array([0.4]), sigma_f=1.0, kind="matern52")
    np.testing.assert_array_equal(ard, iso)


# ============================================================================
# 3. Validation and broadcasting helpers
# ============================================================================

def test_as_length_scales_accepts_scalar_and_vector():
    np.testing.assert_array_equal(bo.as_length_scales(0.5), np.array([0.5]))
    np.testing.assert_array_equal(bo.as_length_scales([1.0, 2.0], d=2),
                                  np.array([1.0, 2.0]))
    # a scalar is always legal whatever d is
    np.testing.assert_array_equal(bo.as_length_scales(0.5, d=7), np.array([0.5]))


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_as_length_scales_rejects_non_positive_or_non_finite(bad):
    with pytest.raises(ValueError):
        bo.as_length_scales(bad)


def test_as_length_scales_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="d = 4"):
        bo.as_length_scales(np.ones(3), d=4)


def test_kernel_rejects_length_scale_of_wrong_dimension():
    X1, X2 = two_point_sets()
    with pytest.raises(ValueError):
        bo.kernel_amp(X1, X2, l_c=np.ones(D - 1), sigma_f=1.0, kind="matern52")


def test_theta_bounds_broadcast_from_three_entries():
    expanded = bo._expand_theta_bounds(ANISOTROPIC_BOUNDS, n_ell=D)
    assert len(expanded) == D + 2
    assert expanded[:D] == [ANISOTROPIC_BOUNDS[0]] * D
    assert expanded[D:] == ANISOTROPIC_BOUNDS[1:]


def test_theta_bounds_passthrough_and_rejection():
    full = [(0.0, 1.0)] * (D + 2)
    assert bo._expand_theta_bounds(full, n_ell=D) == full
    assert bo._expand_theta_bounds(None, n_ell=D) is None
    with pytest.raises(ValueError):
        bo._expand_theta_bounds([(0.0, 1.0)] * 5, n_ell=D)


def test_whitening_weights_have_unit_geometric_mean():
    ell = np.array([0.1, 1.0, 10.0, 2.0])
    w = bo.ard_whitening_weights(ell)
    np.testing.assert_allclose(np.exp(np.mean(np.log(w))), 1.0, rtol=1e-12)
    # ordering: the shortest length scale gets the largest weight
    assert np.argmax(w) == int(np.argmin(ell))
    unnormalised = bo.ard_whitening_weights(ell, normalize=False)
    np.testing.assert_allclose(unnormalised, 1.0 / ell, rtol=1e-12)


def test_format_length_scales():
    assert bo.format_length_scales(0.5) == format(0.5, ".3e")
    assert bo.format_length_scales(np.array([0.5, 2.0])).startswith("[")


# ============================================================================
# 4. Marginal likelihood
# ============================================================================

def test_nll_infers_layout_from_theta_length():
    X, y = anisotropic_dataset(n=50)
    y = (y - y.mean()) / y.std()
    theta_iso = np.log([0.7, 1.0, 0.1])
    theta_ard = np.log(np.concatenate([np.full(D, 0.7), [1.0, 0.1]]))
    a = bo.negative_log_marginal_likelihood(theta_iso, X, y, kernel="matern52")
    b = bo.negative_log_marginal_likelihood(theta_ard, X, y, kernel="matern52")
    assert np.isclose(a, b, rtol=1e-10)


def test_nll_matches_a_direct_computation():
    from scipy.linalg import cholesky, cho_solve
    X, y = anisotropic_dataset(n=45)
    y = (y - y.mean()) / y.std()
    ell = np.array([0.3, 2.0, 0.8, 4.0])
    sigma_f, sigma_y, jitter = 1.2, 0.05, 1e-10
    theta = np.log(np.concatenate([ell, [sigma_f, sigma_y]]))

    K = bo.kernel_amp(X, X, l_c=ell, sigma_f=sigma_f, kind="matern52")
    Ky = K + (sigma_y ** 2 + jitter) * np.eye(len(X))
    L = cholesky(Ky, lower=True)
    alpha = cho_solve((L, True), y)
    ll = (-0.5 * (y @ alpha) - np.sum(np.log(np.diag(L)))
          - 0.5 * len(X) * np.log(2.0 * np.pi))

    got = bo.negative_log_marginal_likelihood(theta, X, y, jitter=jitter,
                                              kernel="matern52")
    assert np.isclose(got, -ll, rtol=1e-12)


def test_nll_param_hist_records_every_length_scale():
    X, y = anisotropic_dataset(n=40)
    hist = []
    theta = np.log(np.concatenate([np.full(D, 0.7), [1.0, 0.1]]))
    bo.negative_log_marginal_likelihood(theta, X, y, param_hist=hist,
                                        kernel="matern52")
    assert len(hist[0]) == D + 2


def test_ard_marginal_likelihood_beats_isotropic_on_anisotropic_data():
    X, y = anisotropic_dataset(n=160)
    y = (y - y.mean()) / y.std()

    def best_nll(ard):
        cfg = bo.GPConfig(kernel="matern52", ard=ard, normalize_y=False,
                          optimize_hyperparams=True, hpo_every=1,
                          n_hpo_restarts=6, theta_bounds_log=ANISOTROPIC_BOUNDS,
                          hpo_random_state=5)
        gp = bo.fit_gp(bo.build_gp_model(cfg), X, y, cfg, it=0)
        return bo.negative_log_marginal_likelihood(gp.theta_log, X, y,
                                                   kernel="matern52")

    assert best_nll(True) < best_nll(False) - 5.0


# ============================================================================
# 5. Hyperparameter optimisation
# ============================================================================

def test_hpo_recovers_the_relevant_coordinates():
    X, y = anisotropic_dataset(n=200, seed=4)
    result = bo.estimate_ard_length_scales(
        X, y, kernel="matern52", theta_bounds_log=ANISOTROPIC_BOUNDS,
        n_restarts=6, random_state=3,
    )
    ell = result["length_scales"]
    assert ell.shape == (D,)
    # coordinate 0 drives the objective and drives it fastest
    assert int(result["relevance_order"][0]) == 0
    # the two inert coordinates must be far longer than the active one
    assert ell[1] > 20.0 * ell[0]
    assert ell[3] > 20.0 * ell[0]
    # and the anisotropy must actually be reported
    assert result["spread"] > 10.0


def test_hpo_length_scales_stay_inside_the_declared_box():
    X, y = anisotropic_dataset(n=120)
    cfg = bo.GPConfig(kernel="matern52", ard=True, normalize_y=True,
                      optimize_hyperparams=True, hpo_every=1, n_hpo_restarts=4,
                      theta_bounds_log=ANISOTROPIC_BOUNDS, hpo_random_state=7)
    gp = bo.fit_gp(bo.build_gp_model(cfg), X, y, cfg, it=0)
    ell = bo.as_length_scales(gp.l_c, D)
    assert np.all(ell >= np.exp(ANISOTROPIC_BOUNDS[0][0]) - 1e-9)
    assert np.all(ell <= np.exp(ANISOTROPIC_BOUNDS[0][1]) + 1e-9)


def test_hpo_random_state_makes_the_fit_reproducible():
    X, y = anisotropic_dataset(n=90)

    def fit(seed):
        cfg = bo.GPConfig(kernel="matern52", ard=True, normalize_y=True,
                          optimize_hyperparams=True, hpo_every=1,
                          n_hpo_restarts=4, theta_bounds_log=ANISOTROPIC_BOUNDS,
                          hpo_random_state=seed)
        return bo.fit_gp(bo.build_gp_model(cfg), X, y, cfg, it=0).theta_log

    np.testing.assert_allclose(fit(42), fit(42), rtol=1e-12)


def test_unseeded_hpo_still_works():
    """hpo_random_state=None is the historical default and must not break."""
    X, y = anisotropic_dataset(n=60)
    cfg = bo.GPConfig(kernel="matern52", ard=True, normalize_y=True,
                      optimize_hyperparams=True, hpo_every=1, n_hpo_restarts=2,
                      theta_bounds_log=ANISOTROPIC_BOUNDS)
    gp = bo.fit_gp(bo.build_gp_model(cfg), X, y, cfg, it=0)
    assert bo.as_length_scales(gp.l_c, D).size == D


def test_stale_isotropic_warm_start_is_discarded_not_reshaped():
    X, y = anisotropic_dataset(n=70)
    cfg = bo.GPConfig(kernel="matern52", ard=True, normalize_y=True,
                      optimize_hyperparams=True, hpo_every=1, n_hpo_restarts=2,
                      theta_bounds_log=ANISOTROPIC_BOUNDS, hpo_random_state=2)
    gp = bo.build_gp_model(cfg)
    gp.theta_log = np.log([0.9, 1.0, 0.1])       # 3 entries, wrong layout
    gp = bo.fit_gp(gp, X, y, cfg, it=0)
    assert gp.theta_log.size == D + 2


def test_theta0_log_is_broadcast_from_three_entries():
    X, y = anisotropic_dataset(n=60)
    cfg = bo.GPConfig(kernel="matern52", ard=True, normalize_y=True,
                      optimize_hyperparams=True, hpo_every=1, n_hpo_restarts=0,
                      theta_bounds_log=ANISOTROPIC_BOUNDS,
                      theta0_log=np.log([0.8, 1.0, 0.1]), hpo_random_state=1)
    gp = bo.fit_gp(bo.build_gp_model(cfg), X, y, cfg, it=0)
    assert gp.theta_log.size == D + 2


def test_scalar_seed_is_expanded_when_ard_is_enabled():
    X, y = anisotropic_dataset(n=50)
    cfg = bo.GPConfig(l_c=0.7, kernel="matern52", ard=True, normalize_y=True,
                      optimize_hyperparams=False)
    gp = bo.fit_gp(bo.build_gp_model(cfg), X, y, cfg, it=0)
    np.testing.assert_allclose(bo.as_length_scales(gp.l_c, D), np.full(D, 0.7))


# ============================================================================
# 6. Prediction quality -- the reason ARD exists
# ============================================================================

def test_ard_predicts_anisotropic_data_better_than_isotropic():
    X, y = anisotropic_dataset(n=260, seed=9)
    X_train, y_train, X_test, y_test = X[:180], y[:180], X[180:], y[180:]

    def rmse(ard):
        cfg = bo.GPConfig(kernel="matern52", ard=ard, normalize_y=True,
                          optimize_hyperparams=True, hpo_every=1,
                          n_hpo_restarts=6, theta_bounds_log=ANISOTROPIC_BOUNDS,
                          hpo_random_state=11)
        gp = bo.fit_gp(bo.build_gp_model(cfg), X_train, y_train, cfg, it=0)
        mu, _ = bo.gp_predict(X_test, gp.Xs, gp.alpha, gp.L, l_c=gp.l_c,
                              sigma_f=gp.sigma_f, kernel=gp.kernel)
        mu = mu * gp.y_std + gp.y_mean
        return float(np.sqrt(np.mean((mu - y_test) ** 2)))

    assert rmse(True) < 0.7 * rmse(False)


def test_frozen_ard_predicts_as_well_as_fitted_ard():
    """Freezing measured length scales -- the pre-registered route -- must
    reproduce the fitted ARD model, not merely approximate it."""
    X, y = anisotropic_dataset(n=200, seed=13)
    fitted = bo.estimate_ard_length_scales(
        X, y, kernel="matern52", theta_bounds_log=ANISOTROPIC_BOUNDS,
        n_restarts=6, random_state=3,
    )
    frozen_cfg = bo.GPConfig(l_c=fitted["length_scales"], kernel="matern52",
                             sigma_f=fitted["sigma_f"], sigma_y=fitted["sigma_y"],
                             normalize_y=True, optimize_hyperparams=False)
    gp = bo.fit_gp(bo.build_gp_model(frozen_cfg), X, y, frozen_cfg, it=0)
    np.testing.assert_allclose(bo.as_length_scales(gp.l_c, D),
                               fitted["length_scales"], rtol=1e-12)

    grid = np.random.default_rng(0).random((25, D))
    mu_frozen, var_frozen = bo.gp_predict(grid, gp.Xs, gp.alpha, gp.L,
                                          l_c=gp.l_c, sigma_f=gp.sigma_f,
                                          kernel="matern52")

    # the same model expressed as an isotropic kernel on rescaled inputs
    ell = fitted["length_scales"]
    iso_cfg = bo.GPConfig(l_c=1.0, kernel="matern52", sigma_f=fitted["sigma_f"],
                          sigma_y=fitted["sigma_y"], normalize_y=True,
                          optimize_hyperparams=False)
    gp_iso = bo.fit_gp(bo.build_gp_model(iso_cfg), X / ell, y, iso_cfg, it=0)
    mu_iso, var_iso = bo.gp_predict(grid / ell, gp_iso.Xs, gp_iso.alpha,
                                    gp_iso.L, l_c=1.0, sigma_f=gp_iso.sigma_f,
                                    kernel="matern52")
    np.testing.assert_allclose(mu_frozen, mu_iso, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(var_frozen, var_iso, rtol=1e-9, atol=1e-11)


# ============================================================================
# 7. Rank-1 update legality under ARD
# ============================================================================

def test_rank1_update_matches_full_refit_under_ard():
    """With ARD length scales the O(n^2) Cholesky extension must reproduce the
    O(n^3) refit exactly.  normalize_y is off on purpose: the rank-1 path keeps
    the normalisation statistics it was built with (by design, see
    `update_gp_rank1`), so leaving it on would compare two different models."""
    X, y = anisotropic_dataset(n=40, seed=21)
    ell = np.array([0.25, 3.0, 0.9, 6.0])
    cfg = bo.GPConfig(l_c=ell, kernel="matern52", sigma_f=1.0, sigma_y=0.05,
                      normalize_y=False, optimize_hyperparams=False)

    incremental = bo.fit_gp(bo.build_gp_model(cfg), X[:-1], y[:-1], cfg, it=0)
    incremental = bo.update_gp_rank1(incremental, X[-1], float(y[-1]))
    full = bo.fit_gp(bo.build_gp_model(cfg), X, y, cfg, it=0)

    np.testing.assert_allclose(incremental.Xs, full.Xs, rtol=1e-12)
    np.testing.assert_allclose(incremental.L, full.L, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(incremental.alpha, full.alpha,
                               rtol=1e-8, atol=1e-11)

    grid = np.random.default_rng(1).random((15, D))
    mu_r1, var_r1 = bo.gp_predict(grid, incremental.Xs, incremental.alpha,
                                  incremental.L, l_c=incremental.l_c,
                                  sigma_f=incremental.sigma_f, kernel="matern52")
    mu_full, var_full = bo.gp_predict(grid, full.Xs, full.alpha, full.L,
                                      l_c=full.l_c, sigma_f=full.sigma_f,
                                      kernel="matern52")
    np.testing.assert_allclose(mu_r1, mu_full, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(var_r1, var_full, rtol=1e-8, atol=1e-10)


def test_rank1_update_rejects_nothing_new_under_ard():
    """The vector length scale must survive a rank-1 extension unchanged."""
    X, y = anisotropic_dataset(n=20, seed=22)
    ell = np.array([0.25, 3.0, 0.9, 6.0])
    cfg = bo.GPConfig(l_c=ell, kernel="matern52", normalize_y=False,
                      optimize_hyperparams=False)
    gp = bo.fit_gp(bo.build_gp_model(cfg), X[:-1], y[:-1], cfg, it=0)
    gp = bo.update_gp_rank1(gp, X[-1], float(y[-1]))
    np.testing.assert_allclose(bo.as_length_scales(gp.l_c, D), ell, rtol=1e-12)
    assert gp.Xs.shape == (20, D)


# ============================================================================
# 8. End-to-end BO driver
# ============================================================================

def _quadratic(x):
    x = np.asarray(x, dtype=float)
    # only coordinates 0 and 1 matter; 2 is inert
    return float((x[0] - 0.3) ** 2 + 0.05 * (x[1] + 0.4) ** 2)


def _run_bo(ard, tmp_path=None, export=False):
    return bo.bayesian_optimization(
        f=_quadratic,
        bounds=[(-2.0, 2.0)] * 3,
        bo_cfg=bo.BOConfig(n_init=12, n_iter=3, random_state=5),
        gp_cfg=bo.GPConfig(kernel="matern52", normalize_y=True, ard=ard,
                           optimize_hyperparams=True, hpo_every=1,
                           n_hpo_restarts=2,
                           theta_bounds_log=ANISOTROPIC_BOUNDS,
                           hpo_random_state=17),
        acq_cfg=bo.AcqConfig(kind="EI", maximize=False),
        optim_cfg=bo.OptimConfig(method="random", n_raw_samples=200),
        save_cfg=bo.SaveConfig(save_path=str(tmp_path) if export else None,
                               log_enabled=export, export_states=export),
    )


def test_bo_driver_runs_with_ard_and_records_every_length_scale():
    result = _run_bo(ard=True)
    assert result.X.shape == (15, 3)
    assert bo.as_length_scales(result.gp.l_c, 3).size == 3
    for item in result.history:
        assert np.asarray(item["l_c"]).size == 3


def test_bo_driver_isotropic_history_stays_scalar():
    result = _run_bo(ard=False)
    for item in result.history:
        assert np.ndim(item["l_c"]) == 0


def test_bo_driver_logging_and_export_survive_a_vector_length_scale(tmp_path):
    result = _run_bo(ard=True, tmp_path=tmp_path, export=True)
    states = sorted((tmp_path / "states").glob("state_*.npz"))
    assert states
    data = np.load(states[-1])
    assert data["l_c"].shape == (3,)
    log = (tmp_path / "run.log").read_text()
    assert "l_c=[" in log            # the vector was rendered, not crashed on
    assert result.best_y >= 0.0


# ============================================================================
# 9. Offline estimator contract
# ============================================================================

def test_estimate_ard_length_scales_contract():
    X, y = anisotropic_dataset(n=150, seed=31)
    result = bo.estimate_ard_length_scales(
        X, y, kernel="matern52", theta_bounds_log=ANISOTROPIC_BOUNDS,
        n_restarts=4, random_state=1,
    )
    for key in ("length_scales", "sigma_f", "sigma_y", "nll", "theta_log",
                "spread", "relevance_order", "whitening_weights",
                "y_mean", "y_std"):
        assert key in result
    assert result["length_scales"].shape == (D,)
    assert result["whitening_weights"].shape == (D,)
    assert result["theta_log"].size == D + 2
    assert result["spread"] >= 1.0
    assert np.isfinite(result["nll"])
    assert set(result["relevance_order"].tolist()) == set(range(D))


def test_estimate_ard_rejects_malformed_input():
    with pytest.raises(ValueError):
        bo.estimate_ard_length_scales(np.ones((3, 2)), np.ones(4))
    with pytest.raises(ValueError):
        bo.estimate_ard_length_scales(np.array([[np.nan, 1.0]]), np.ones(1))


def test_vector_length_scale_is_never_collapsed_by_hpo():
    """A frozen vector l_c with HPO left on must keep d length scales.
    Refitting a single scalar over it would silently discard the anisotropy
    the caller declared -- the one footgun this API could plausibly have."""
    X, y = anisotropic_dataset(n=140, seed=17)
    cfg = bo.GPConfig(l_c=np.array([0.3, 5.0, 1.0, 5.0]), kernel="matern52",
                      normalize_y=True, ard=False,          # note: ard NOT set
                      optimize_hyperparams=True, hpo_every=1, n_hpo_restarts=3,
                      theta_bounds_log=ANISOTROPIC_BOUNDS, hpo_random_state=4)
    gp = bo.fit_gp(bo.build_gp_model(cfg), X, y, cfg, it=0)
    assert bo.is_ard(gp.l_c)
    assert gp.theta_log.size == D + 2
    ell = bo.as_length_scales(gp.l_c, D)
    assert ell[0] < ell[1]          # the active coordinate stays the short one
