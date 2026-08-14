"""Fast regression diagnostics for ML4F batch EI and optional ADAM."""

from __future__ import annotations

import numpy as np

import BO_ML4F as bo


BOUNDS = [(-2.0, 2.0), (-2.0, 2.0)]


def objective(x):
    x = np.asarray(x, dtype=float)
    return float((x[0] - 0.3) ** 2 + 2.0 * (x[1] + 0.4) ** 2)


def gradient(x):
    x = np.asarray(x, dtype=float)
    return np.array([2.0 * (x[0] - 0.3), 4.0 * (x[1] + 0.4)])


def run(*, n_candidates=1, refinement=None, gradient_function=None):
    return bo.bayesian_optimization(
        f=objective,
        bounds=BOUNDS,
        bo_cfg=bo.BOConfig(n_init=4, n_iter=2, random_state=12),
        gp_cfg=bo.GPConfig(normalize_y=True),
        acq_cfg=bo.AcqConfig(kind="EI", maximize=False),
        optim_cfg=bo.OptimConfig(
            method="random",
            n_raw_samples=300,
            n_candidates=n_candidates,
            batch_distance_scale=0.08,
        ),
        save_cfg=bo.SaveConfig(log_enabled=False),
        gradient=gradient_function,
        refinement_cfg=refinement,
    )


def test_default_is_backward_compatible():
    result = run()
    assert result.X.shape == (6, 2)
    assert all(item["n_added"] == 1 for item in result.history)


def test_batch_ei_without_gradient():
    result = run(n_candidates=3)
    assert result.X.shape == (10, 2)
    assert all(item["n_added"] == 3 for item in result.history)
    for item in result.history:
        assert item["x_proposed"].shape == (3, 2)


def test_moving_adam_adds_both_endpoints():
    config = bo.GradientRefinementConfig(
        enabled=True,
        n_steps=8,
        learning_rate=0.05,
        distance_threshold=1e-8,
    )
    result = run(
        n_candidates=3, refinement=config, gradient_function=gradient
    )
    assert result.X.shape == (16, 2)
    assert all(item["n_added"] == 6 for item in result.history)
    assert all(
        np.all(item["refinement_displacement"] > config.distance_threshold)
        for item in result.history
    )
    assert np.isclose(result.best_y, np.min(result.y))


def test_close_pair_adds_only_final_point():
    config = bo.GradientRefinementConfig(
        enabled=True,
        n_steps=5,
        learning_rate=0.05,
        distance_threshold=1e-3,
        close_pair_policy="final",
    )
    result = run(
        n_candidates=3,
        refinement=config,
        gradient_function=lambda x: np.zeros(2),
    )
    assert result.X.shape == (10, 2)
    assert all(item["n_added"] == 3 for item in result.history)
    assert all(
        np.allclose(item["refinement_displacement"], 0.0)
        for item in result.history
    )


def test_gradient_is_required_only_when_enabled():
    config = bo.GradientRefinementConfig(enabled=True)
    try:
        run(refinement=config)
    except ValueError as error:
        assert "gradient must be provided" in str(error)
    else:
        raise AssertionError("Enabled refinement accepted a missing gradient.")


if __name__ == "__main__":
    test_default_is_backward_compatible()
    test_batch_ei_without_gradient()
    test_moving_adam_adds_both_endpoints()
    test_close_pair_adds_only_final_point()
    test_gradient_is_required_only_when_enabled()
    print("ML4F batch-EI/ADAM diagnostics passed.")
