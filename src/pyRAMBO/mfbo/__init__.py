"""
pyRAMBO.mfbo: a Multi-Fidelity Bayesian Optimization package for the Machine
Learning for Fluid Systems group (ML4F).

Two fidelity levels ("L" low, "H" high) coupled by the autoregressive AR1
model of Kennedy & O'Hagan (2000): f_H(x) = rho*f_L(x) + delta(x). Each BO
iteration proposes both a location x AND a fidelity level, trading expected
improvement against the cost of each fidelity (see core.make_acquisition).
The incumbent is always defined at the high fidelity only.

mfbo intentionally mirrors the structure, API shape, naming conventions,
docstring style, and config-dataclass pattern of sbo (src/pyRAMBO/sbo), but is a
fully independent package: no shared base module, no imports from sbo.
Some code is duplicated between the two on purpose -- see the project-level
notes for why.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from .core import (
    Array,
    Bounds,
    Level,
    GPConfig,
    FidelityConfig,
    AcqConfig,
    OptimConfig,
    MFBOConfig,
    SaveConfig,
    NormalizationHelper,
    MFGPModel,
    MFBOState,
    MFBOResult,
    MFAcqOptimizationResult,
    sample_initial_points,
    evaluate_objective,
    init_dataset,
    rbf_kernel,
    rbf_kernel_amp,
    mf_gp_fit,
    mf_gp_predict,
    predictive_variance_reduction,
    optimize_gp_hyperparams,
    build_gp_model,
    fit_gp,
    make_acquisition,
    optimize_acquisition,
    multi_fidelity_bayesian_optimization,
)

from .persistence import (
    TraceLog,
    GPSnapshot,
    GPPosterior,
    ReplayRun,
    build_gp_snapshot,
    reconstruct_gp,
    load_trace,
    load_snapshot,
    load_run,
)

__version__ = "0.1.0"

__all__ = [
    "Array",
    "Bounds",
    "Level",
    "GPConfig",
    "FidelityConfig",
    "AcqConfig",
    "OptimConfig",
    "MFBOConfig",
    "SaveConfig",
    "NormalizationHelper",
    "MFGPModel",
    "MFBOState",
    "MFBOResult",
    "MFAcqOptimizationResult",
    "sample_initial_points",
    "evaluate_objective",
    "init_dataset",
    "rbf_kernel",
    "rbf_kernel_amp",
    "mf_gp_fit",
    "mf_gp_predict",
    "predictive_variance_reduction",
    "optimize_gp_hyperparams",
    "build_gp_model",
    "fit_gp",
    "make_acquisition",
    "optimize_acquisition",
    "multi_fidelity_bayesian_optimization",
    "TraceLog",
    "GPSnapshot",
    "GPPosterior",
    "ReplayRun",
    "build_gp_snapshot",
    "reconstruct_gp",
    "load_trace",
    "load_snapshot",
    "load_run",
]
