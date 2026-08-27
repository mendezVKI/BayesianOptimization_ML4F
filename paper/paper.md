---
title: 'bo_ml4f: A lightweight Bayesian Optimization framework with gradient-refined and batch acquisition'
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

`bo_ml4f` is a compact, dependency-light Bayesian Optimization (BO) library
built around a Gaussian Process (GP) surrogate with an exact rank-1 Cholesky
update scheme, three standard acquisition functions (Expected Improvement,
Probability of Improvement, Upper Confidence Bound), and optional
hyperparameter optimization by marginal-likelihood maximization. Beyond the
standard single-point BO loop, `bo_ml4f` supports two extensions relevant to
expensive, simulation-driven objectives common in fluid mechanics and
machine-learning-for-fluids workflows: (i) diverse batch acquisition, which
proposes several candidate evaluations per iteration to better exploit
parallel evaluation budgets, and (ii) gradient-based local refinement, which
uses a user-supplied true objective gradient (e.g. from an adjoint solver or
automatic differentiation) to locally refine each acquisition proposal via
projected ADAM before evaluation.

# Statement of need

TODO: describe the gap `bo_ml4f` fills relative to existing BO packages
(e.g. BoTorch, GPyOpt, scikit-optimize) for the intended audience --
researchers in the ML4F group and, more broadly, users who have access to
gradients of expensive black-box objectives (e.g. via adjoint CFD solvers)
and want to combine them with sample-efficient BO rather than choosing one
or the other.

# Software design

The package is organized into three modules under `src/bo_ml4f`:

- `core.py`: configuration dataclasses, the GP model (kernel, fit, predict,
  rank-1 Cholesky extension), acquisition functions and their optimizers
  (including batch acquisition and ADAM-based gradient refinement), and the
  main `bayesian_optimization` driver.
- `saving.py`: experiment folder management, HDF5 checkpointing of each BO
  iteration, and run logging.
- `plotting.py`: all matplotlib-based visualization, imported lazily so
  that headless/HPC usage does not require a configured matplotlib backend.

Runnable tutorials against standard benchmark functions (1D sinusoidal,
Branin, Rosenbrock) are provided in `examples/`, and correctness is checked
in `tests/`, including a numerical check that the rank-1 Cholesky update
(generalized to accept more than one new point per BO iteration, as batch
acquisition and gradient refinement can produce) exactly reproduces a full
Cholesky refit.

# Acknowledgements

TODO: acknowledge funding sources / colleagues, if any.

# References
