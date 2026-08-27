"""
bo_ml4f: a Bayesian Optimization framework for the Machine Learning for
Fluid Systems group (ML4F).

link: https://www.mendezma.com/

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from .core import (
    Array,
    Bounds,
    GPConfig,
    AcqConfig,
    OptimConfig,
    GradientRefinementConfig,
    BOConfig,
    SaveConfig,
    NormalizationHelper,
    GPModel,
    BOState,
    BOResult,
    AcqOptimizationResult,
    sample_initial_points,
    evaluate_objective,
    init_dataset,
    rbf_kernel,
    rbf_kernel_amp,
    gp_fit,
    gp_predict,
    optimize_gp_hyperparams,
    build_gp_model,
    fit_gp,
    make_acquisition,
    optimize_acquisition,
    optimize_acquisition_batch,
    adam_refine_candidate,
    bayesian_optimization,
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

__version__ = "0.3.0"

__all__ = [
    "Array",
    "Bounds",
    "GPConfig",
    "AcqConfig",
    "OptimConfig",
    "GradientRefinementConfig",
    "BOConfig",
    "SaveConfig",
    "NormalizationHelper",
    "GPModel",
    "BOState",
    "BOResult",
    "AcqOptimizationResult",
    "sample_initial_points",
    "evaluate_objective",
    "init_dataset",
    "rbf_kernel",
    "rbf_kernel_amp",
    "gp_fit",
    "gp_predict",
    "optimize_gp_hyperparams",
    "build_gp_model",
    "fit_gp",
    "make_acquisition",
    "optimize_acquisition",
    "optimize_acquisition_batch",
    "adam_refine_candidate",
    "bayesian_optimization",
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
