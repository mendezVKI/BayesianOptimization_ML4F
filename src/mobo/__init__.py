"""
mobo: a Multi-Objective Bayesian Optimization package for the Machine
Learning for Fluid Systems group (ML4F).

Several objectives are modeled JOINTLY by a single multi-output GP with an
intrinsic coregionalization model (ICM): a shared ARD RBF kernel over the
inputs, scaled by a learned output covariance matrix B. B captures how the
objectives co-vary in the LATENT processes -- that is, whatever correlation
remains once the fact that they are all smooth functions of the same x has
been accounted for; it is not the empirical correlation of the observed
objective values, and on a deterministic benchmark whose objectives are each
modelled well on their own it correctly comes out near zero. Each BO iteration proposes the location that maximizes
the Expected Hypervolume Improvement (EHVI) of the current Pareto front,
estimated by Monte Carlo from the joint posterior. There is no scalar
incumbent: progress is the dominated hypervolume with respect to a fixed
reference point.

Two objectives are supported. The ICM GP itself is written for an arbitrary
number of outputs; only the hypervolume machinery is 2D-specific, and it
raises explicitly beyond that (see core.N_OBJ_SUPPORTED).

mobo intentionally mirrors the structure, API shape, naming conventions,
docstring style, and config-dataclass pattern of sbo (src/sbo) and mfbo
(src/mfbo), but is a fully independent package: no shared base module, no
imports from either. Some code is duplicated between the three on purpose --
see the project-level notes for why.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from .core import (
    Array,
    Bounds,
    N_OBJ_SUPPORTED,
    GPConfig,
    AcqConfig,
    OptimConfig,
    MOBOConfig,
    SaveConfig,
    NormalizationHelper,
    ICMGPModel,
    MOBOState,
    MOBOResult,
    AcqOptimizationResult,
    sample_initial_points,
    evaluate_objective,
    init_dataset,
    rbf_kernel,
    rbf_kernel_ard,
    build_B,
    pack_theta,
    unpack_theta,
    pack_Y,
    unpack_Y,
    build_icm_covariance,
    icm_gp_fit,
    icm_gp_predict,
    nondominated_mask,
    pareto_front,
    hypervolume,
    hypervolume_improvement,
    default_theta_bounds,
    negative_log_marginal_likelihood,
    optimize_gp_hyperparams,
    build_gp_model,
    fit_gp,
    make_acquisition,
    optimize_acquisition,
    multi_objective_bayesian_optimization,
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
    "N_OBJ_SUPPORTED",
    "GPConfig",
    "AcqConfig",
    "OptimConfig",
    "MOBOConfig",
    "SaveConfig",
    "NormalizationHelper",
    "ICMGPModel",
    "MOBOState",
    "MOBOResult",
    "AcqOptimizationResult",
    "sample_initial_points",
    "evaluate_objective",
    "init_dataset",
    "rbf_kernel",
    "rbf_kernel_ard",
    "build_B",
    "pack_theta",
    "unpack_theta",
    "pack_Y",
    "unpack_Y",
    "build_icm_covariance",
    "icm_gp_fit",
    "icm_gp_predict",
    "nondominated_mask",
    "pareto_front",
    "hypervolume",
    "hypervolume_improvement",
    "default_theta_bounds",
    "negative_log_marginal_likelihood",
    "optimize_gp_hyperparams",
    "build_gp_model",
    "fit_gp",
    "make_acquisition",
    "optimize_acquisition",
    "multi_objective_bayesian_optimization",
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
