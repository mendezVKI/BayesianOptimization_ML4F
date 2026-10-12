"""
Gradient-based local refinement of acquisition proposals for sbo.

Split out of core.py so that the compute core (config/GP/acquisition/BO
loop) only calls a single entry point, ``refine_candidates``. Each proposal
x0 coming from the acquisition optimizer is optionally polished by projected
ADAM on the true objective gradient (e.g. adjoint / autodiff), giving a
refined endpoint xL. Then, per candidate, either both (x0, xL) are kept as
observations, or -- if they ended up closer than ``distance_threshold`` --
they are merged into a single point according to ``close_pair_policy``.

This module has no runtime dependency on core.py (type hints only, thanks to
``from __future__ import annotations``), so there is no circular import.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .core import Array, Bounds


# Optional local refinement of each acquisition proposal using the true
# objective gradient (e.g. adjoint/AD gradients), via projected ADAM.
@dataclass
class GradientRefinementConfig:
    enabled: bool = False
    n_steps: int = 50
    learning_rate: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    distance_threshold: float = 1e-3  # normalized parameter-space distance
    close_pair_policy: str = "final"  # "final" or "midpoint"


@dataclass
class RefinementResult:
    """Outcome of refining one batch of proposals.

    x_refined / y_refined / displacements have one row per proposal (NaN
    refined point/value and 0 displacement where refinement is disabled).
    round_X / round_y are the observations to append to the dataset this
    round (one or two rows per proposal, see handle_close_pair).
    """
    x_refined: "Array"         # (n_candidates, d), NaN if not refined
    y_refined: "Array"         # (n_candidates,),   NaN if not refined
    displacements: "Array"     # (n_candidates,) normalized |x_refined - x_proposed|
    round_X: "Array"           # (n_added, d)
    round_y: "Array"           # (n_added,)


def validate_refinement_config(config: GradientRefinementConfig, gradient: Optional[Callable]) -> None:
    """Raise ValueError on an inconsistent refinement configuration."""
    if config.enabled and gradient is None:
        raise ValueError("gradient must be provided when gradient refinement is enabled.")
    if config.close_pair_policy not in {"final", "midpoint"}:
        raise ValueError("refinement_cfg.close_pair_policy must be 'final' or 'midpoint'.")
    if config.distance_threshold < 0.0:
        raise ValueError("refinement_cfg.distance_threshold must be non-negative.")
    if config.n_steps < 0:
        raise ValueError("refinement_cfg.n_steps must be non-negative.")
    if config.learning_rate <= 0.0:
        raise ValueError("refinement_cfg.learning_rate must be positive.")
    if not (0.0 <= config.beta1 < 1.0):
        raise ValueError("refinement_cfg.beta1 must lie in [0, 1).")
    if not (0.0 <= config.beta2 < 1.0):
        raise ValueError("refinement_cfg.beta2 must lie in [0, 1).")


def adam_refine_candidate(
    x0: "Array",
    gradient: Callable[["Array"], "Array"],
    bounds: "Bounds",
    config: GradientRefinementConfig,
) -> "Array":
    """
    Locally refine a single candidate by projected ADAM gradient descent, in
    normalized [0,1]^d parameter space (so a single learning_rate is
    meaningful across dimensions with different physical scales).

    ``gradient`` must return the gradient of the (minimization) objective at
    a point in physical coordinates; it may also return a tuple whose last
    element is the gradient (e.g. (value, grad)), matching common
    autodiff/adjoint solver conventions.
    """
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    width = np.asarray([b[1] - b[0] for b in bounds], dtype=float)
    z = np.clip((np.asarray(x0, dtype=float) - lo) / width, 0.0, 1.0)
    first = np.zeros_like(z)
    second = np.zeros_like(z)

    for step in range(1, int(config.n_steps) + 1):
        raw = gradient(lo + width * z)
        if isinstance(raw, tuple):
            raw = raw[-1]
        grad_z = np.asarray(raw, dtype=float).reshape(z.shape) * width
        if not np.all(np.isfinite(grad_z)):
            break
        first = config.beta1 * first + (1.0 - config.beta1) * grad_z
        second = config.beta2 * second + (1.0 - config.beta2) * grad_z**2
        first_hat = first / (1.0 - config.beta1**step)
        second_hat = second / (1.0 - config.beta2**step)
        z = np.clip(
            z - config.learning_rate * first_hat / (np.sqrt(second_hat) + config.epsilon),
            0.0,
            1.0,
        )

    return lo + width * z


def should_merge_pair(displacement: float, config: GradientRefinementConfig) -> bool:
    """True if the refined point is too close to its proposal (normalized
    displacement <= distance_threshold) to be worth two evaluations."""
    return not (displacement > config.distance_threshold)


def handle_close_pair(
    x0: "Array",
    y0: float,
    xL: "Array",
    yL: float,
    f: Callable[["Array"], float],
    config: GradientRefinementConfig,
) -> Tuple["Array", float]:
    """Merge a close (proposal, refined) pair into ONE observation.

    "midpoint": evaluate f at 0.5*(x0 + xL) (one extra evaluation).
    "final":    keep the refined endpoint (xL, yL), no extra evaluation.
    """
    if config.close_pair_policy == "midpoint":
        midpoint = 0.5 * (x0 + xL)
        return midpoint, float(np.asarray(f(midpoint)).reshape(-1)[0])
    return xL, float(yL)


def refine_candidates(
    x_proposed: "Array",
    y_proposed: "Array",
    f: Callable[["Array"], float],
    gradient: Optional[Callable[["Array"], "Array"]],
    bounds: "Bounds",
    config: GradientRefinementConfig,
    maximize: bool = False,
) -> RefinementResult:
    """Refine a batch of acquisition proposals and decide what to observe.

    With ``config.enabled=False`` nothing is refined: every proposal is kept
    as is. Otherwise each proposal is polished by adam_refine_candidate and
    kept as a (proposal, refined) pair, or merged via handle_close_pair when
    they are closer than ``config.distance_threshold``.

    ``maximize=True`` negates the user gradient (the refinement always
    minimizes).
    """
    x_proposed = np.asarray(x_proposed, dtype=float)
    n = len(x_proposed)
    x_refined = np.full_like(x_proposed, np.nan)
    y_refined = np.full(n, np.nan, dtype=float)
    displacements = np.zeros(n, dtype=float)
    round_X: List["Array"] = []
    round_y: List[float] = []

    if not config.enabled:
        for x0, y0 in zip(x_proposed, y_proposed):
            round_X.append(x0)
            round_y.append(float(y0))
        return RefinementResult(
            x_refined, y_refined, displacements,
            np.asarray(round_X, dtype=float), np.asarray(round_y, dtype=float),
        )

    local_gradient = gradient
    if maximize:
        def local_gradient(x, grad=gradient):
            value = grad(x)
            raw_gradient = value[-1] if isinstance(value, tuple) else value
            return -np.asarray(raw_gradient)

    width = np.asarray([b[1] - b[0] for b in bounds], dtype=float)

    for q, (x0, y0) in enumerate(zip(x_proposed, y_proposed)):
        xL = adam_refine_candidate(x0, local_gradient, bounds, config)
        yL = float(np.asarray(f(xL)).reshape(-1)[0])
        displacement = float(np.linalg.norm((xL - x0) / width))

        x_refined[q] = xL
        y_refined[q] = yL
        displacements[q] = displacement

        if should_merge_pair(displacement, config):
            x_obs, y_obs = handle_close_pair(x0, float(y0), xL, yL, f, config)
            round_X.append(x_obs)
            round_y.append(y_obs)
        else:
            round_X.extend([x0, xL])
            round_y.extend([float(y0), yL])

    return RefinementResult(
        x_refined, y_refined, displacements,
        np.asarray(round_X, dtype=float), np.asarray(round_y, dtype=float),
    )
