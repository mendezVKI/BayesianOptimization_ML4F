"""
Regenerates the library-output figures used by pyRAMBO_overview.tex.

Every picture is produced by the real pyRAMBO plotting code (not mocked up):
one short run per subpackage, then a few frames are copied into figures/.

    python Presentations/make_figures.py

@authors: Yannick Lecomte and Miguel A. Mendez
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from simple_cases.benchmarks import (  # noqa: E402
    sinusoidal_1d, branin_2d, forrester_high, forrester_low, schaffer_n1,
)
from pyRAMBO import sbo, mfbo, mobo  # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix="pyrambo_figs_"))


def run_dir(out):
    """The single run subfolder the library creates inside `out`."""
    return next(p for p in Path(out).iterdir() if p.is_dir())


def grab(run, subdir, frames, prefix):
    for it in frames:
        src = run / "plots" / subdir / f"it_{it:03d}.png"
        if not src.exists():
            src = run / subdir / f"it_{it:03d}.png"   # older layouts
        shutil.copy(src, FIG / f"{prefix}_it{it:02d}.png")


def grab_one(run, name, dst):
    src = run / "plots" / name
    if not src.exists():
        src = run / name
    shutil.copy(src, FIG / dst)


# ---------------------------------------------------------------- sbo, 1D
rng = np.random.default_rng(47)
out = TMP / "sbo1d"
res = sbo.bayesian_optimization(
    f=lambda x: sinusoidal_1d(x, noise_level=0.1, rng=rng),
    f_true=lambda x: sinusoidal_1d(x, noise_level=0.0),
    bounds=[(-2.0, 2.0)],
    bo_cfg=sbo.BOConfig(n_init=5, n_iter=10, random_state=1234),
    gp_cfg=sbo.GPConfig(),
    acq_cfg=sbo.AcqConfig(xi=0.01),
    optim_cfg=sbo.OptimConfig(global_method="refine"),
    save_cfg=sbo.SaveConfig(out_path=str(out), plt_all=True, plot_every=1,
                            snapshot_enabled=True, dpi=200),
)
run = run_dir(out)
grab(run, "GIF", [0, 2, 9], "sbo1d")
grab_one(run, "conv.png", "sbo1d_conv.png")
grab_one(run, "hist.png", "sbo1d_hist.png")
print("sbo 1D done:", res.best_x, res.best_y)

# ---------------------------------------------------------------- sbo, 2D
rng = np.random.default_rng(47)
out = TMP / "sbo2d"
sbo.bayesian_optimization(
    f=lambda x: branin_2d(x, noise_level=0.1, rng=rng),
    f_true=lambda x: branin_2d(x, noise_level=0.0),
    bounds=[(-2.0, 2.0), (-2.0, 2.0)],
    bo_cfg=sbo.BOConfig(n_init=20, n_iter=8, random_state=1234),
    gp_cfg=sbo.GPConfig(),
    acq_cfg=sbo.AcqConfig(xi=0.01),
    optim_cfg=sbo.OptimConfig(),
    save_cfg=sbo.SaveConfig(out_path=str(out), plt_all=True, plot_every=1, dpi=200),
)
grab(run_dir(out), "GIF", [7], "sbo2d")

# ----------------------------------------- sbo, batch + gradient refinement
def rosen(x):
    x = np.atleast_2d(x)
    return (100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / 100


def rosen_grad(x):
    x = np.asarray(x, dtype=float).ravel()
    return np.array([
        -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0] ** 2),
    ]) / 100


out = TMP / "refine"
sbo.bayesian_optimization(
    f=lambda x: float(rosen(x)[0]),
    f_true=lambda x: rosen(x),
    gradient=rosen_grad,
    bounds=[(-2.0, 2.0), (-1.0, 3.0)],
    bo_cfg=sbo.BOConfig(n_init=10, n_iter=6, random_state=3),
    gp_cfg=sbo.GPConfig(),
    acq_cfg=sbo.AcqConfig(),
    optim_cfg=sbo.OptimConfig(n_candidates=3),
    refinement_cfg=sbo.GradientRefinementConfig(enabled=True, n_steps=60,
                                                learning_rate=2e-2),
    save_cfg=sbo.SaveConfig(out_path=str(out), plt_all=True, plot_every=1, dpi=200),
)
grab(run_dir(out), "GIF", [5], "refine")

# ------------------------------------------------------------------- mfbo
rng = np.random.default_rng(47)
out = TMP / "mfbo"
mfbo.multi_fidelity_bayesian_optimization(
    f_low=lambda x: forrester_low(x, noise_level=0.05, rng=rng),
    f_high=lambda x: forrester_high(x, noise_level=0.02, rng=rng),
    f_low_true=lambda x: forrester_low(x),
    f_high_true=lambda x: forrester_high(x),
    bounds=[(0.0, 1.0)],
    mfbo_cfg=mfbo.MFBOConfig(n_init_L=8, n_init_H=3, n_iter=12, random_state=1234),
    gp_cfg=mfbo.GPConfig(l_lf=0.2, sigma_lf=5.0, l_delta=0.2, sigma_delta=3.0,
                         sigma_L=0.05, sigma_H=0.02, optimize_hyperparams=True),
    acq_cfg=mfbo.AcqConfig(xi=0.01),
    optim_cfg=mfbo.OptimConfig(method="refined", n_raw_samples=1000),
    fidelity_cfg=mfbo.FidelityConfig(cost_low=1.0, cost_high=10.0),
    save_cfg=mfbo.SaveConfig(out_path=str(out), plt_all=True, plot_every=1, dpi=200),
)
grab(run_dir(out), "GIF", [11], "mfbo")

# ------------------------------------------------------------------- mobo
out = TMP / "mobo"
mobo.multi_objective_bayesian_optimization(
    f=lambda x: schaffer_n1(x),
    f_true=lambda x: schaffer_n1(x),
    bounds=[(-4.0, 4.0)],
    mobo_cfg=mobo.MOBOConfig(n_init=6, n_iter=15, random_state=1234),
    gp_cfg=mobo.GPConfig(length_scales=0.2, L_B_diag=1.0, L_B_offdiag=0.0,
                         sigma_n=0.02, optimize_hyperparams=True),
    acq_cfg=mobo.AcqConfig(n_mc=512, ref_margin=0.10),
    optim_cfg=mobo.OptimConfig(method="refined", n_raw_samples=1000),
    save_cfg=mobo.SaveConfig(out_path=str(out), plt_all=True, plot_every=1, dpi=200),
)
r = run_dir(out)
grab(r, "GIF", [14], "mobo")
if (r / "plots" / "GIF_pareto").exists() or (r / "GIF_pareto").exists():
    grab(r, "GIF_pareto", [14], "mobo_pareto")

shutil.rmtree(TMP, ignore_errors=True)
print("figures written to", FIG)
