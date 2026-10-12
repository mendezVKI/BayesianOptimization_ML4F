"""
Tests for the four run-folder naming modes of pyRAMBO.sbo (SaveConfig.run_naming).
"""

import os
import re

import numpy as np
import pytest

from pyRAMBO import sbo as bo
from pyRAMBO.sbo.saving import setup_experiment_folder


def _f(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return float((x[0] - 0.7) ** 2)


def _run(out_path, n_init=4, n_iter=2, xi=0.01, **save_kwargs):
    return bo.bayesian_optimization(
        f=_f,
        bounds=[(-2.0, 2.0)],
        bo_cfg=bo.BOConfig(n_init=n_init, n_iter=n_iter, random_state=0),
        gp_cfg=bo.GPConfig(optimize_hyperparams=False),
        acq_cfg=bo.AcqConfig(xi=xi),
        optim_cfg=bo.OptimConfig(n_raw_samples=100),
        save_cfg=bo.SaveConfig(out_path=str(out_path), log_enabled=False, **save_kwargs),
    )


def test_default_is_timestamp(tmp_path):
    res = _run(tmp_path)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", os.path.basename(res.out_path))


def test_run_id_increments(tmp_path):
    names = [os.path.basename(_run(tmp_path, run_naming="run_id").out_path) for _ in range(3)]
    assert names == ["run_1", "run_2", "run_3"]


def test_params_name_from_n_init_n_iter_xi(tmp_path):
    res = _run(tmp_path, n_init=4, n_iter=2, xi=0.01, run_naming="params")
    assert os.path.basename(res.out_path) == "ninit_4_niter_2_xi_0.01"
    assert os.path.isdir(os.path.join(res.out_path, "res"))


def test_params_name_never_overwrites(tmp_path):
    _run(tmp_path, run_naming="params")
    with pytest.raises(FileExistsError, match="never overwritten"):
        _run(tmp_path, run_naming="params")                       # same case twice
    other = _run(tmp_path, n_iter=3, run_naming="params")          # a different case is fine
    assert os.path.basename(other.out_path) == "ninit_4_niter_3_xi_0.01"


def test_custom_name(tmp_path):
    res = _run(tmp_path, run_naming="custom", run_name="my_experiment")
    assert os.path.basename(res.out_path) == "my_experiment"
    with pytest.raises(FileExistsError):
        _run(tmp_path, run_naming="custom", run_name="my_experiment")


@pytest.mark.parametrize("bad", [None, "", "  ", "a/b", "a" + chr(92) + "b", "..", "."])  # chr(92) = backslash
def test_custom_name_must_be_a_plain_folder_name(tmp_path, bad):
    with pytest.raises(ValueError):
        setup_experiment_folder(bo.SaveConfig(out_path=str(tmp_path), run_naming="custom", run_name=bad))


def test_unknown_naming_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="run_naming"):
        setup_experiment_folder(bo.SaveConfig(out_path=str(tmp_path), run_naming="nope"))


def test_params_without_name_parts_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="params"):
        setup_experiment_folder(bo.SaveConfig(out_path=str(tmp_path), run_naming="params"))


def test_caller_save_cfg_is_not_mutated(tmp_path):
    cfg = bo.SaveConfig(out_path=str(tmp_path), run_naming="params", log_enabled=False)
    bo.bayesian_optimization(
        f=_f, bounds=[(-2.0, 2.0)], bo_cfg=bo.BOConfig(n_init=4, n_iter=1, random_state=0),
        gp_cfg=bo.GPConfig(optimize_hyperparams=False), acq_cfg=bo.AcqConfig(), optim_cfg=bo.OptimConfig(n_raw_samples=100),
        save_cfg=cfg,
    )
    assert cfg.out_path == str(tmp_path)
