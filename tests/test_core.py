"""
Smoke / correctness tests for pyRAMBO.sbo.core.

Run with: pytest
"""

import numpy as np
import pytest

from pyRAMBO import sbo as bo


def _quadratic(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return float((x[0] - 0.7) ** 2)


def _quadratic_grad(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.array([2.0 * (x[0] - 0.7)])


def test_bayesian_optimization_converges_1d():
    """A short run on a convex 1D objective should land near the minimum."""
    res = bo.bayesian_optimization(
        f=_quadratic,
        bounds=[(-2.0, 2.0)],
        bo_cfg=bo.BOConfig(n_init=5, n_iter=8, random_state=0),
        gp_cfg=bo.GPConfig(),
        acq_cfg=bo.AcqConfig(),
        optim_cfg=bo.OptimConfig(n_raw_samples=300),
        save_cfg=bo.SaveConfig(),  # no plotting/export/logging
    )
    assert res.best_y < 1e-2
    assert abs(res.best_x[0] - 0.7) < 0.2


def test_batch_and_gradient_refinement():
    """Batch acquisition (n_candidates>1) + ADAM gradient refinement should
    also converge, and every iteration should add at least n_candidates
    points."""
    res = bo.bayesian_optimization(
        f=_quadratic,
        bounds=[(-2.0, 2.0)],
        bo_cfg=bo.BOConfig(n_init=4, n_iter=4, random_state=0),
        gp_cfg=bo.GPConfig(),
        acq_cfg=bo.AcqConfig(),
        optim_cfg=bo.OptimConfig(n_raw_samples=200, n_candidates=2, batch_distance_scale=0.1),
        save_cfg=bo.SaveConfig(),
        gradient=_quadratic_grad,
        refinement_cfg=bo.GradientRefinementConfig(enabled=True, n_steps=20, learning_rate=0.05),
    )
    assert res.best_y < 1e-2
    for state in res.states:
        assert state.n_added >= 2


def test_rank1_update_matches_full_refit():
    """The rank-1 Cholesky extension (possibly several new points at once)
    must reproduce a full refit to numerical precision."""
    rng = np.random.default_rng(0)
    bounds = [(-2.0, 2.0), (-1.0, 1.0)]
    # fixed hyperparameters: with HPO the factors are refit, so rank-one would never be used
    gp_cfg = bo.GPConfig(rank_one=True, rank_one_threshold=0, optimize_hyperparams=False)

    def sample(n):
        X = np.column_stack([
            rng.uniform(-2, 2, size=n),
            rng.uniform(-1, 1, size=n),
        ])
        y = np.sin(X[:, 0]) + 0.1 * X[:, 1]
        return X, y

    X0, y0 = sample(3)
    gp = bo.build_gp_model(gp_cfg, bounds)
    gp = bo.fit_gp(gp, X0, y0, gp_cfg, it=0)

    X_new, y_new = sample(3)  # simulate a 3-point batch round
    X2 = np.vstack([X0, X_new])
    y2 = np.concatenate([y0, y_new])

    gp_rank1 = bo.fit_gp(gp, X2, y2, gp_cfg, it=1)

    gp_ref = bo.build_gp_model(gp_cfg, bounds)
    gp_ref = bo.fit_gp(gp_ref, X2, y2, gp_cfg, it=0)  # forces a full refit

    assert np.max(np.abs(gp_rank1.L - gp_ref.L)) < 1e-8
    assert np.max(np.abs(gp_rank1.alpha - gp_ref.alpha)) < 1e-8


def test_normalization_flags():
    """normalize_X / normalize_y = False make the GP see the raw data; True
    (default) scales X to [0,1] and standardizes y. Predictions are always
    in physical units."""
    bounds = [(-2.0, 2.0), (-1.0, 1.0)]
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-2, 2, 6), rng.uniform(-1, 1, 6)])
    y = 100.0 + 5.0 * np.sin(X[:, 0]) + X[:, 1]

    for nX, ny in [(True, True), (False, False), (True, False), (False, True)]:
        cfg = bo.GPConfig(normalize_X=nX, normalize_y=ny, optimize_hyperparams=False, sigma_y=1e-3)
        gp = bo.fit_gp(bo.build_gp_model(cfg, bounds), X, y, cfg, it=0)
        np.testing.assert_allclose(gp.Xs_norm, (X - [-2, -1]) / [4, 2] if nX else X)
        if ny:
            assert abs(gp.ys_norm.mean()) < 1e-9 and abs(gp.ys_norm.std() - 1) < 1e-9
        else:
            np.testing.assert_allclose(gp.ys_norm, y)
        mu, _ = bo.gp_predict(X, gp.Xs, gp.alpha, gp.L, l_c=gp.l_c, sigma_f=gp.sigma_f, gp=gp)
        np.testing.assert_allclose(mu, y, atol=0.5)   # interpolates, in physical units


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
