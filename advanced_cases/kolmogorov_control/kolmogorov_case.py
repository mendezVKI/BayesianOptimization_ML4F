"""
Kolmogorov flow control case: the problem definition shared by every script.

The flow is HydroGym's 2-D Kolmogorov flow (hydrogym.jax, pseudo-spectral,
Re = 200, 64 x 64, driven by the body force sin(4 y)). It is actuated by a
body force made of shear modes

    f_c(y, t) = sum_k a_k(t) sin(k y) + b_k(t) cos(k y),   k = 1, 2, 3, 4,   |a|, |b| <= 0.5

(HydroGym's own actuation is sin(k y), k = 4, 5, 6, 7.) The control is an
open-loop schedule: the episode is cut in n_segments segments of segment_time
time units and the amplitudes are constant over a segment. The unknowns are
therefore x = [amplitudes(segment 1), ..., amplitudes(segment n)]: 3 * 8 = 24
values by default.

Two objectives, lower is better (see KolmogorovControl.rollout):

    "tracking"  reach the flow that a hidden reference schedule gives; the cost
                is the distance to it seen by the 8 x 8 velocity probes of the
                environment (100 without control, 0 at the reference schedule)
    "energy"    minus the return of the HydroGym environment: kinetic energy
                plus an actuation penalty

Each evaluation is one simulation (~0.5 s); the flow is deterministic, so is
the objective.

The optimizers (sbo, scikit-optimize) only ever see ``KolmogorovControl.cost``,
so they are compared on exactly the same function.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import logging
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))    # pyRAMBO, if not pip-installed

Array = np.ndarray

ACTION_BOUND = 0.5              # |a_i| <= 0.5: the action box of the HydroGym environment


def _make_environment(dt: float, segment_time: float, alpha: float, wavenumbers: tuple, both_phases: bool):
    """HydroGym's Kolmogorov environment with the settings of this case: one
    environment step lasts `segment_time`, the reward weighs the energy by
    `alpha` (HydroGym's default is 0: actuation penalty only), and the forcing
    modes are chosen (HydroGym's: sin(k y), k = 4, 5, 6, 7)."""
    import jax.numpy as jnp
    from jax import lax
    from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

    n_modes = len(wavenumbers) * (2 if both_phases else 1)

    class Environment(KolmogorovFlow):
        @property
        def default_params(self):
            return super().default_params.replace(action_time=segment_time, reward_alpha=alpha,
                                                  action_dim=n_modes)

        def _control_field(self, action, params):
            """HydroGym's forcing, sum_i a_i sin(k_i y), for any wavenumbers k_i,
            followed by the modes cos(k_i y) if both_phases."""
            modes = [jnp.sin(k * self.y) for k in wavenumbers]
            if both_phases:
                modes += [jnp.cos(k * self.y) for k in wavenumbers]
            return sum(action[i] * mode for i, mode in enumerate(modes)), jnp.zeros_like(self.y)

        def _rollout(self, omega_hat0, params, control_field=None):
            """Same integration as HydroGym's (RungeKuttaCrankNicolson.solve), which
            advances every block of steps twice when a segment holds several
            blocks (segment_time > 1): same result, half the time."""
            p = self.default_params
            n_inner = int(int(p.save_time) // float(p.dt))                      # steps between two saves
            n_outer = int(int(float(p.action_time) // float(p.dt)) // n_inner)  # saves per segment
            rk_step = self.integrator.RK4_CN(control_field=control_field)

            def block(omega_hat, _):
                omega_hat = lax.scan(lambda w, _: (rk_step(w), None), omega_hat, xs=None, length=n_inner)[0]
                return omega_hat, omega_hat

            return lax.scan(block, omega_hat0, xs=None, length=n_outer)

    return Environment(env_config={"dt": dt})


@dataclass
class KolmogorovControl:
    """Cost function x -> cost of the forcing schedule x, with an evaluation
    counter and a timer so the time spent *inside* the objective can be
    separated from the optimizer's own overhead."""

    n_segments: int = 3           # segments of the forcing schedule
    wavenumbers: tuple = (1, 2, 3, 4)   # forcing modes sin(k y) (HydroGym's: 4, 5, 6, 7)
    both_phases: bool = True      # also force the modes cos(k y): twice as many amplitudes
    objective: str = "tracking"   # "tracking" or "energy" (see rollout)
    alpha: float = 200.0          # "energy": weight of the kinetic energy against the actuation penalty
    reference_seed: int = 0       # "tracking": seed of the hidden reference schedule
    segment_time: float = 1.0     # duration of a segment, in time units (>= 1, HydroGym's solver)
    dt: float = 5e-3              # time step (HydroGym's default 1e-3 is 5x slower; 1e-2 blows up)
    spinup_time: float = 10.0     # uncontrolled time before the episode (HydroGym's reset: 10)
    n_evals: int = field(init=False, default=0)
    eval_time: float = field(init=False, default=0.0)
    eval_times: list = field(init=False, default_factory=list, repr=False)   # one entry per evaluation

    def __post_init__(self):
        # Imported here: HydroGym's JAX stack takes ~30 s to load, and it is
        # not needed to re-plot saved results.
        logging.getLogger("tensorflow").setLevel(logging.ERROR)   # its import noise
        import jax
        from jax import lax

        self.env = _make_environment(self.dt, self.segment_time, self.alpha,
                                     tuple(self.wavenumbers), self.both_phases)
        params = self.env.default_params
        key = jax.random.PRNGKey(0)                  # required by the API; the flow is deterministic

        def advance(state, actions):
            """Run len(actions) segments from `state`."""
            def segment(state, action):
                obs, state, reward, _, info = self.env.step_env(key, state, action, params)
                return state, (reward, info["mean_tke"], obs)
            state, (rewards, energies, probes) = lax.scan(segment, state, actions)
            return state, rewards, energies, probes

        self._advance = jax.jit(advance)

        # Initial state of every episode: the environment's reset, run without
        # control up to spinup_time. Computed once.
        _, state = jax.jit(self.env.reset_env)(key, params)
        n_spinup = int(round(self.spinup_time / self.segment_time)) - 1
        if n_spinup > 0:
            state = self._advance(state, np.zeros((n_spinup, self.n_modes), dtype=np.float32))[0]
        self.state0 = state
        self._probes = lambda x: np.asarray(self._advance(self.state0, self._actions(x))[3], dtype=float)
        probes0 = self._probes(np.zeros(self.dim))   # also compiles, outside any timed region

        if self.objective == "tracking":
            # Target: what the probes see with a reference schedule, hidden from the optimizers.
            self.x_ref = 0.6 * ACTION_BOUND * np.random.default_rng(self.reference_seed).uniform(-1, 1, self.dim)
            self._target = self._probes(self.x_ref)
            self._mismatch0 = np.sum((probes0 - self._target) ** 2)      # without control
        elif self.objective != "energy":
            raise ValueError(f"Unknown objective '{self.objective}'. Use 'energy' or 'tracking'.")

    @property
    def n_modes(self) -> int:
        """Forcing amplitudes per segment."""
        return len(self.wavenumbers) * (2 if self.both_phases else 1)

    @property
    def mode_names(self) -> list:
        k = self.wavenumbers
        return [f"sin {i}y" for i in k] + ([f"cos {i}y" for i in k] if self.both_phases else [])

    @property
    def dim(self) -> int:
        return self.n_segments * self.n_modes

    @property
    def bounds(self) -> list:
        return [(-ACTION_BOUND, ACTION_BOUND)] * self.dim

    def reset_counters(self) -> None:
        self.n_evals, self.eval_time, self.eval_times = 0, 0.0, []

    def _actions(self, x: Array) -> Array:
        return np.asarray(x, dtype=np.float32).reshape(self.n_segments, self.n_modes)

    def rollout(self, x: Array):
        """Run one episode with schedule x. Returns (cost, energy per segment,
        final vorticity field). The cost is, with objective =

        "energy"    minus the return of the HydroGym environment: summed over the
                    segments, alpha * (mean kinetic energy) + sum_i |a_i|
        "tracking"  the squared distance between the probe velocities (8 x 8
                    probes, every segment) and those of the target flow, in % of
                    its value without control: 100 at x = 0, 0 at the reference
                    schedule."""
        state, rewards, energies, probes = self._advance(self.state0, self._actions(x))
        if self.objective == "tracking":
            cost = 100.0 * float(np.sum((np.asarray(probes, dtype=float) - self._target) ** 2) / self._mismatch0)
        else:
            cost = -float(np.sum(np.asarray(rewards, dtype=float)))
        if not np.isfinite(cost):                    # solver blow-up
            cost = 1e6
        return cost, np.asarray(energies, dtype=float), np.fft.irfftn(np.asarray(state.omega_hat))

    def cost(self, x: Array) -> float:
        """Objective handed to the optimizers (timed and counted)."""
        t0 = time.perf_counter()
        value = self.rollout(x)[0]
        dt = time.perf_counter() - t0
        self.eval_time += dt
        self.eval_times.append(dt)
        self.n_evals += 1
        return value
