"""
Smoke / correctness tests for bo_ml4f.core.

Run with: pytest
"""

import numpy as np
import pytest

import bo_ml4f as bo


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
    gp_cfg = bo.GPConfig(rank_one=True, rank_one_threshold=0)

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
