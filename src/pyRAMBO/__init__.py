"""
pyRAMBO: Rank-One Accelerated Multi-objective/fidelity/output for Bayesian
Optimization.

A Bayesian Optimization framework for the Machine Learning for Fluid Systems
group (ML4F). It bundles three sibling subpackages, one per flavour of BO:

    pyRAMBO.sbo     single-fidelity, single-objective BO
    pyRAMBO.mfbo    multi-fidelity BO (AR1 / Kennedy & O'Hagan)
    pyRAMBO.mobo    multi-objective BO (ICM GP + Monte-Carlo EHVI)

They share a deliberately identical module structure but no code: each is
independent, with no common base module and no imports between them. Use
them as, e.g.:

    from pyRAMBO import sbo
    res = sbo.bayesian_optimization(...)

link: https://www.mendezma.com/

@authors: Yannick Lecomte and Miguel A. Mendez
"""

__version__ = "0.3.0"

from . import sbo, mfbo, mobo

__all__ = ["sbo", "mfbo", "mobo", "__version__"]
