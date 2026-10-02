"""
Burgers control case: the problem definition shared by every script.

A Burgers wave is perturbed at x = 6.6 and controlled at x = 13.2 through a
linear feedback law

    a(t) = w . [u(x_1, t), u(x_2, t), u(x_3, t)]         (3 sensors, 3 weights)

The objective is the cumulative cost over one episode (perturbation damping
in 770 <= i < 820 plus an actuation penalty, see Burgers_implicit_env.py):
lower is better. Finding w is a 3-D black-box minimization, each evaluation
being one PDE simulation (~0.2 s).

The optimizers (sbo, scikit-optimize) only ever see ``BurgersControl.cost``,
so they are compared on exactly the same function.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # Burgers/ (the environment)
sys.path.insert(0, str(HERE.parents[1] / "src"))    # pyRAMBO, if not pip-installed

from Burgers.Burgers_implicit_env import Burgers_training

Array = np.ndarray

# Sensor locations (grid indices), control/perturbation positions and the
# evaluation window: used by the plotting code too.
SENSORS = (400, 450, 500)
EVAL_WINDOW = (770, 820)
X_PERTURBATION, X_ACTUATION = 6.6, 13.2

@dataclass
class BurgersControl:
    """Cost function w -> cumulative episode cost, with an evaluation counter
    and a timer so the time spent *inside* the objective can be separated from
    the optimizer's own overhead."""

    gamma: float = 10.0           # actuation penalty weight
    ic: str = "fully_developed_deterministic"
    env: Burgers_training = field(init=False, repr=False)
    n_evals: int = field(init=False, default=0)
    eval_time: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.env = Burgers_training(name="burgers_control", ic=self.ic, GAMMA=self.gamma)

    @property
    def n_steps(self) -> int:
        return int(self.env.T / self.env.dt)

    def reset_counters(self) -> None:
        self.n_evals, self.eval_time = 0, 0.0

    def rollout(self, w: Array, record_field: bool = False):
        """Run one episode with weights w. Returns (cost, u_history) where
        u_history is the (n_steps, Nx+1) field if record_field else None."""
        w = np.asarray(w, dtype=float).reshape(-1)
        obs = self.env.reset()
        cost = 0.0
        u_hist = np.zeros((self.n_steps, self.env.Nx + 1)) if record_field else None
        for k in range(self.n_steps):
            obs, reward, done, _ = self.env.step(float(w @ obs))
            cost += reward
            if record_field:
                u_hist[k] = self.env.u
            if done and k < self.n_steps - 1:   # solver crash: penalty already in reward
                break
        return cost, u_hist

    def cost(self, w: Array) -> float:
        """Objective handed to the optimizers (timed and counted)."""
        t0 = time.perf_counter()
        value = float(self.rollout(w)[0])
        self.eval_time += time.perf_counter() - t0
        self.n_evals += 1
        return value
