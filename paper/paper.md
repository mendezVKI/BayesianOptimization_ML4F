---
title: 'pyRAMBO: A lightweight Bayesian Optimization framework with gradient-refined and batch acquisition'
tags:
  - Python
  - Bayesian optimization
  - Gaussian processes
  - optimization
  - machine learning
authors:
  - name: Yannick Lecomte
    orcid: 0000-0000-0000-0000  # TODO: replace with actual ORCID
    affiliation: 1
  - name: Miguel A. Mendez
    orcid: 0000-0000-0000-0000  # TODO: replace with actual ORCID
    affiliation: 1
affiliations:
  - name: von Karman Institute for Fluid Dynamics, Belgium  # TODO: confirm affiliation/institution name
    index: 1
date: 26 August 2026
bibliography: paper.bib
---

# Summary

`pyRAMBO` is a compact, dependency-light Bayesian Optimization (BO) library
built around a Gaussian Process (GP) surrogate with an exact rank-1 Cholesky
update scheme, three standard acquisition functions (Expected Improvement,
Probability of Improvement, Upper Confidence Bound), and optional
hyperparameter optimization by marginal-likelihood maximization. Beyond the
standard single-point BO loop, `pyRAMBO` supports two extensions relevant to
expensive, simulation-driven objectives common in fluid mechanics and
machine-learning-for-fluids workflows: (i) diverse batch acquisition, which
proposes several candidate evaluations per iteration to better exploit
parallel evaluation budgets, and (ii) gradient-based local refinement, which
uses a user-supplied true objective gradient (e.g. from an adjoint solver or
automatic differentiation) to locally refine each acquisition proposal via
projected ADAM before evaluation.

# Statement of need

TODO: describe the gap `pyRAMBO` fills relative to existing BO packages
(e.g. BoTorch, GPyOpt, scikit-optimize) for the intended audience --
researchers in the ML4F group and, more broadly, users who have access to
gradients of expensive black-box objectives (e.g. via adjoint CFD solvers)
and want to combine them with sample-efficient BO rather than choosing one
or the other.

# Software design

`pyRAMBO` is organized into three independent subpackages under
`src/pyRAMBO`, one per flavour of BO: `pyRAMBO.sbo` (single-fidelity,
single-objective), `pyRAMBO.mfbo` (multi-fidelity, AR1 model of Kennedy and
O'Hagan) and `pyRAMBO.mobo` (multi-objective, intrinsic coregionalization
GP with Monte-Carlo expected hypervolume improvement). Each subpackage has
the same four modules:

- `core.py`: configuration dataclasses, the GP model (kernel, fit, predict),
  acquisition functions and their optimizers, and the main BO driver. In
  `pyRAMBO.sbo` this also includes the rank-1 Cholesky extension, batch
  acquisition and ADAM-based gradient refinement.
- `saving.py`: experiment folder management and run logging.
- `persistence.py`: two-tier run persistence (a lightweight per-iteration
  trace, plus opt-in snapshots sufficient to rebuild the GP exactly), stored
  as NumPy/JSON files.
- `plotting.py`: all matplotlib-based visualization, imported lazily so
  that headless/HPC usage does not require a configured matplotlib backend.

Runnable tutorials against standard benchmark functions (1D sinusoidal,
Branin, Rosenbrock, Forrester, Schaffer, Binh-Korn) are provided in
`examples/`, and correctness is checked in `tests/`, including a numerical
check that the rank-1 Cholesky update (generalized to accept more than one
new point per BO iteration, as batch acquisition and gradient refinement can
produce) exactly reproduces a full Cholesky refit.

# Acknowledgements

TODO: acknowledge funding sources / colleagues, if any.

# References
