"""
Tests for pyRAMBO.sbo.refinement (gradient-based refinement of proposals).
"""

import numpy as np
import pytest

from pyRAMBO import sbo as bo
from pyRAMBO.sbo import refinement as ref

BOUNDS = [(-2.0, 2.0)]


def _f(x):
    return float((np.asarray(x).reshape(-1)[0] - 0.7) ** 2)


def _grad(x):
    return np.array([2.0 * (np.asarray(x).reshape(-1)[0] - 0.7)])


def test_public_api_reexported():
    assert bo.GradientRefinementConfig is ref.GradientRefinementConfig
    assert bo.adam_refine_candidate is ref.adam_refine_candidate
    assert bo.refine_candidates is ref.refine_candidates


def test_validate_config():
    ref.validate_refinement_config(ref.GradientRefinementConfig(), None)  # disabled: ok
    with pytest.raises(ValueError):
        ref.validate_refinement_config(ref.GradientRefinementConfig(enabled=True), None)
    with pytest.raises(ValueError):
        ref.validate_refinement_config(
            ref.GradientRefinementConfig(close_pair_policy="nope"), None
        )


def test_should_merge_pair_and_policies():
    cfg = ref.GradientRefinementConfig(distance_threshold=1e-3)
    assert ref.should_merge_pair(1e-4, cfg)
    assert ref.should_merge_pair(1e-3, cfg)
    assert not ref.should_merge_pair(1e-2, cfg)

    x0, xL = np.array([0.0]), np.array([0.2])
    x, y = ref.handle_close_pair(x0, _f(x0), xL, _f(xL), _f, cfg)  # "final"
    np.testing.assert_allclose(x, xL)
    assert y == pytest.approx(_f(xL))

    cfg_mid = ref.GradientRefinementConfig(close_pair_policy="midpoint")
    x, y = ref.handle_close_pair(x0, _f(x0), xL, _f(xL), _f, cfg_mid)
    np.testing.assert_allclose(x, [0.1])
    assert y == pytest.approx(_f([0.1]))


def test_refine_disabled_keeps_proposals():
    xp = np.array([[0.0], [1.0]])
    yp = np.array([_f(x) for x in xp])
    out = ref.refine_candidates(xp, yp, _f, None, BOUNDS, ref.GradientRefinementConfig())
    np.testing.assert_allclose(out.round_X, xp)
    np.testing.assert_allclose(out.round_y, yp)
    assert np.all(np.isnan(out.x_refined)) and np.all(np.isnan(out.y_refined))
    assert np.all(out.displacements == 0.0)


def test_refine_enabled_pair_improves_and_keeps_both():
    cfg = ref.GradientRefinementConfig(enabled=True, n_steps=50, learning_rate=0.05)
    xp = np.array([[-1.5]])
    yp = np.array([_f(xp[0])])
    out = ref.refine_candidates(xp, yp, _f, _grad, BOUNDS, cfg)
    assert out.y_refined[0] < yp[0]
    assert out.displacements[0] > cfg.distance_threshold
    assert len(out.round_y) == 2  # proposal + refined endpoint, far apart
    assert out.round_y[1] == pytest.approx(out.y_refined[0])


def test_refine_close_pair_merged_to_one_point():
    # zero learning steps -> refined == proposal -> displacement 0 -> merged
    cfg = ref.GradientRefinementConfig(enabled=True, n_steps=0)
    xp = np.array([[0.3]])
    yp = np.array([_f(xp[0])])
    out = ref.refine_candidates(xp, yp, _f, _grad, BOUNDS, cfg)
    assert len(out.round_y) == 1


def test_maximize_negates_gradient():
    # maximizing -(x-0.7)^2 : user supplies gradient of the maximized fn
    cfg = ref.GradientRefinementConfig(enabled=True, n_steps=50, learning_rate=0.05)
    g = lambda x: -_grad(x)
    xp = np.array([[-1.5]])
    out = ref.refine_candidates(xp, np.array([0.0]), lambda x: -_f(x), g, BOUNDS, cfg, maximize=True)
    assert abs(out.x_refined[0, 0] - 0.7) < abs(-1.5 - 0.7)
