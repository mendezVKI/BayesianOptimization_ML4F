"""
Tests for pyRAMBO.sbo plotting behaviour: show_plots flag and saved figures.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyRAMBO import sbo as bo


def _f(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return float((x[0] - 0.7) ** 2)


def _f_true(X):
    return np.array([_f(x) for x in np.atleast_2d(X)])


def _run(tmp_path, show_plots=None):
    kwargs = {} if show_plots is None else dict(show_plots=show_plots)
    return bo.bayesian_optimization(
        f=_f,
        bounds=[(-2.0, 2.0)],
        bo_cfg=bo.BOConfig(n_init=4, n_iter=3, random_state=0),
        gp_cfg=bo.GPConfig(),
        acq_cfg=bo.AcqConfig(),
        optim_cfg=bo.OptimConfig(n_raw_samples=100),
        save_cfg=bo.SaveConfig(out_path=str(tmp_path), plt_all=True, log_enabled=False, **kwargs),
        f_true=_f_true,
    )


def test_plots_are_saved_but_not_shown_by_default(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))
    res = _run(tmp_path)

    assert shown == []                       # no pop-up window
    assert plt.get_fignums() == []           # and no figure left open
    plots = os.path.join(res.out_path, "plots")
    assert os.path.exists(os.path.join(plots, "conv.png"))
    assert os.path.exists(os.path.join(plots, "hist.png"))
    assert len(os.listdir(os.path.join(plots, "GIF"))) == 3


def test_show_plots_true_shows_every_figure(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))
    _run(tmp_path, show_plots=True)
    assert len(shown) == 3 + 1 + 1           # 3 state plots + conv + hist
    assert plt.get_fignums() == []


@pytest.mark.parametrize("name", ["conv.png", "hist.png"])
def test_conv_and_hist_are_not_cropped(tmp_path, name):
    """A cropped image has content (labels) touching the left/bottom edge; a
    correct one keeps a blank margin there."""
    res = _run(tmp_path)
    img = plt.imread(os.path.join(res.out_path, "plots", name))[..., :3]
    assert img[:, 0, :].min() > 0.99         # leftmost column blank
    assert img[-1, :, :].min() > 0.99        # bottom row blank


def test_make_gif_assembles_the_saved_frames(tmp_path):
    from PIL import Image

    res = _run(tmp_path)                       # plt_all=True -> 3 state frames
    gif_path = bo.make_gif(res.out_path, fps=4)

    assert gif_path == os.path.join(res.out_path, "plots", "evolution.gif")
    with Image.open(gif_path) as gif:
        assert gif.n_frames == 3               # one frame per iteration
        assert gif.info["loop"] == 0
        sizes = set()
        for k in range(gif.n_frames):
            gif.seek(k)
            sizes.add(gif.size)
    assert len(sizes) == 1                     # uniform canvas, no jitter


def test_make_gif_without_frames_raises_a_clear_error(tmp_path):
    res = bo.bayesian_optimization(
        f=_f, bounds=[(-2.0, 2.0)],
        bo_cfg=bo.BOConfig(n_init=4, n_iter=2, random_state=0),
        gp_cfg=bo.GPConfig(), acq_cfg=bo.AcqConfig(), optim_cfg=bo.OptimConfig(n_raw_samples=100),
        save_cfg=bo.SaveConfig(out_path=str(tmp_path), log_enabled=False),   # plots disabled
    )
    with pytest.raises(FileNotFoundError, match="plt_state_enabled"):
        bo.make_gif(res.out_path)
