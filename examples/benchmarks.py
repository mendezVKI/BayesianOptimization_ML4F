"""
Benchmark objective functions for optimization.

All functions follow the interface:

    f(x: np.ndarray) -> float

where:
    - x is a 1D numpy array of shape (d,)
    - returns a scalar float

Designed for development, debugging, and performance evaluation
of optimization algorithms.

Import this as a module of the examples package:

    from examples.benchmarks import branin_2d

and never as a bare top-level ``import benchmarks``, which would require
putting ``examples/`` on sys.path -- the one thing that makes ``import sbo``
/ ``mfbo`` / ``mobo`` resolve to the example folders of the same name. See
examples/README.md.

Author: Yannick Lecomte
"""

import numpy as np


#%% Utilities

def _validate_input(x: np.ndarray, dim: int):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.shape[0] != dim:
        raise ValueError(f"Input must be 1D array of size {dim}")
    return x


#%% 1D Functions

def quadratic_1d(x: np.ndarray) -> float:
    """Simple convex quadratic (minimum at x=2)."""
    x = _validate_input(x, 1)
    return (x[0] - 2.0) ** 2


def sinusoidal_1d(x, noise_level=0.01, rng=None):
    """
    Non-convex 1D sinusoidal function.
    Accepts scalar or array.
    Returns scalar or array consistently.
    """
    
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float)

    # Core deterministic function (vectorized automatically)
    val = np.sin(5 * x) * (1 - np.tanh(x**2))

    # Noise with matching shape
    noise = np.random.randn(*x.shape) * noise_level

    return val + noise


def sinusoidal_1d_large_scale(x, noise_level=2.0, rng=None):
    """
    Large-scale non-convex 1D sinusoidal test function.

    Intended for testing BO implementations with input/output normalization.

    Recommended input range:
        x in [-20, 20]   or even [-50, 50]

    Typical output scale:
        O(10^2)

    Parameters
    ----------
    x : scalar or array-like
        Input location(s).
    noise_level : float, optional
        Standard deviation of additive Gaussian noise.
    rng : np.random.Generator, optional
        Random number generator for reproducible noise.

    Returns
    -------
    scalar or np.ndarray
        Function value(s), with shape matching x.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float)

    # Same spirit as your original function:
    # oscillatory part * smooth envelope, but with much larger scales
    val = 80.0 * np.sin(0.8 * x) * (1.2 - np.tanh((x / 8.0) ** 2)) \
        + 25.0 * np.cos(0.25 * x) \
        + 0.8 * x

    noise = rng.normal(loc=0.0, scale=noise_level, size=x.shape)

    out = val + noise
    return float(out) if out.ndim == 0 else out


#%% 2D Functions

def _prepare_input(x, d):
    x = np.asarray(x, dtype=float)
    
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    if x.shape[1] != d:
        raise ValueError(f"Expected input dimension {d}, got {x.shape[1]}")
    
    return x

def rosenbrock_2d(x, noise_level=0.0, rng=None):
    """
    Rosenbrock function (global minimum at [1,1]).
    Accepts (d,) or (n,d). Returns scalar or (n,) consistently.
    """

    if rng is None:
        rng = np.random.default_rng()
        
    # Ensure the input format is convinient 
    x = _prepare_input(x, 2)
    x0 = x[:, 0]
    x1 = x[:, 1]
    # Compute rosenbrock function
    val = 100 * (x1 - x0**2)**2 + (1 - x0)**2
    # Generate noise using the random generator
    noise = rng.normal(size=val.shape) * noise_level
    out = val/100 + noise
    return out if out.shape[0] > 1 else out[0]

def branin_2d(x, noise_level=0.0, rng=None):
    """
    Branin function.
    Accepts (d,) or (n,d). Returns scalar or (n,) consistently.
    """

    if rng is None:
        rng = np.random.default_rng()
    # Ensure the input format is convinient 
    x = _prepare_input(x, 2)
    x0 = x[:, 0]
    x1 = x[:, 1]
    # Coefs of the branin function
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    # Compute branin function
    val = a * (x1 - b * x0**2 + c * x0 - r)**2 + s * (1 - t) * np.cos(x0) + s
    # Generate noise using the random generator
    noise = rng.normal(size=val.shape) * noise_level
    out = val + noise

    return out if out.shape[0] > 1 else out[0]



def himmelblau_2d(x, noise_level=0.0, rng=None):
    """
    Himmelblau function (multiple minima).
    Accepts (d,) or (n,d). Returns scalar or (n,) consistently.
    """

    if rng is None:
        rng = np.random.default_rng()
    # Ensure the input format is convinient 
    x = _prepare_input(x, 2)
    x0 = x[:, 0]
    x1 = x[:, 1]
    # Compute himmelblau function
    val = (x0**2 + x1 - 11)**2 + (x0 + x1**2 - 7)**2
    # Generate noise using the random generator
    noise = rng.normal(size=val.shape) * noise_level
    out = val + noise

    return out if out.shape[0] > 1 else out[0]


#%% 3D Functions

def sphere_3d(x: np.ndarray) -> float:
    """Simple convex sphere function."""
    x = _validate_input(x, 3)
    return np.sum(x**2)


def rastrigin_3d(x: np.ndarray) -> float:
    """Rastrigin function (multimodal)."""
    x = _validate_input(x, 3)
    A = 10
    return A * 3 + np.sum(x**2 - A * np.cos(2 * np.pi * x))


#%% 4D Functions

def sphere_4d(x: np.ndarray) -> float:
    """4D sphere function."""
    x = _validate_input(x, 4)
    return np.sum(x**2)


def ackley_4d(x: np.ndarray) -> float:
    """Ackley function in 4D."""
    x = _validate_input(x, 4)
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = 4

    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))

    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)

    return term1 + term2 + a + np.exp(1)


#%% Multi-fidelity 1D functions (Forrester, Sobester & Keane, 2008)

def forrester_high(x, noise_level=0.0, rng=None):
    """
    High-fidelity Forrester function. Standard multi-fidelity BO benchmark.
    Global minimum at x* ~= 0.7572, f(x*) ~= -6.0207. Domain: x in [0, 1].
    Accepts (d,) or (n,d) with d=1. Returns scalar or (n,) consistently.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = _prepare_input(x, 1)[:, 0]
    val = (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)
    noise = rng.normal(size=val.shape) * noise_level
    out = val + noise
    return out if out.shape[0] > 1 else out[0]


def forrester_low(x, noise_level=0.0, rng=None, A=0.5, B=10.0, C=-5.0):
    """
    Low-fidelity Forrester function: a cheap, systematically-biased but
    strongly correlated surrogate for forrester_high (same domain/minimizer
    region), following Forrester, Sobester & Keane (2008):

        f_low(x) = A*f_high(x) + B*(x - 0.5) + C
    """
    if rng is None:
        rng = np.random.default_rng()
    x2d = _prepare_input(x, 1)
    x = x2d[:, 0]
    val = A * forrester_high(x2d, noise_level=0.0) + B * (x - 0.5) + C
    noise = rng.normal(size=val.shape) * noise_level
    out = val + noise
    return out if out.shape[0] > 1 else out[0]


#%% Scalable N-D

def sphere_nd(x: np.ndarray) -> float:
    """N-dimensional sphere function."""
    x = np.asarray(x, dtype=float)
    return np.sum(x**2)


def rastrigin_nd(x: np.ndarray) -> float:
    """N-dimensional Rastrigin."""
    x = np.asarray(x, dtype=float)
    A = 10
    d = len(x)
    return A * d + np.sum(x**2 - A * np.cos(2 * np.pi * x))


#%% Multi-objective functions
#
# Interface: like the single-objective helpers above, these accept (d,) or
# (n,d) and return either an (M,) objective vector for a single point or an
# (n,M) array for a batch. Every objective is stated for MINIMIZATION, which
# is what mobo assumes by default.

def schaffer_n1(x, noise_level=0.0, rng=None):
    """
    Schaffer problem N.1: the standard 1D bi-objective benchmark.

        f1(x) = x^2
        f2(x) = (x - 2)^2                     x in [-4, 4]

    Both minimized. The two objectives pull in opposite directions, so the
    Pareto set is the whole interval x in [0, 2] and the Pareto front is the
    convex curve it traces in objective space.
    """
    if rng is None:
        rng = np.random.default_rng()
    x0 = _prepare_input(x, 1)[:, 0]
    out = np.column_stack([x0 ** 2, (x0 - 2.0) ** 2])
    out = out + rng.normal(size=out.shape) * noise_level
    return out if out.shape[0] > 1 else out[0]


def binh_korn(x, noise_level=0.0, rng=None):
    """
    Binh & Korn problem (unconstrained form): the standard 2D bi-objective
    benchmark.

        f1(x) = 4*x1^2 + 4*x2^2
        f2(x) = (x1 - 5)^2 + (x2 - 5)^2       x1 in [0,5], x2 in [0,3]

    Both minimized. f1 pulls towards the origin and f2 towards (5,5), so the
    Pareto set is the segment between them and the objectives are strongly
    anti-correlated -- which is exactly the regime the off-diagonal entry of
    mobo's coregionalization matrix B is there to capture.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = _prepare_input(x, 2)
    x0, x1 = x[:, 0], x[:, 1]
    out = np.column_stack([
        4.0 * x0 ** 2 + 4.0 * x1 ** 2,
        (x0 - 5.0) ** 2 + (x1 - 5.0) ** 2,
    ])
    out = out + rng.normal(size=out.shape) * noise_level
    return out if out.shape[0] > 1 else out[0]
