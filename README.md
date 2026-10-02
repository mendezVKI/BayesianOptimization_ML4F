# pyRAMBO

**R**ank-One **A**ccelerated **M**ulti-objective/fidelity/output for **B**ayesian **O**ptimization.

A lightweight Bayesian Optimization (BO) framework developed within the
Machine Learning for Fluid Systems group (ML4F): https://www.mendezma.com/

It implements a Gaussian-Process-based BO loop (EI/PI/UCB acquisition,
rank-1 Cholesky updates, optional hyperparameter optimization) plus two
optional extensions: diverse batch acquisition (propose several candidates
per iteration) and gradient-based local refinement of each proposal via
projected ADAM, for objectives where a true gradient (adjoint/AD) is
available.

## Installation

```bash
git clone https://github.com/mendezVKI/BayesianOptimization_ML4F.git
cd BayesianOptimization_ML4F
pip install -e ".[test]"
```

This installs the `pyRAMBO` package (with its `sbo`, `mfbo` and `mobo`
subpackages) in editable mode, plus `pytest` for running the test suite.

## Quickstart

```python
import numpy as np
from pyRAMBO import sbo as bo

f = lambda x: (x[0] - 0.7) ** 2  # objective to minimize

res = bo.bayesian_optimization(
    f=f,
    bounds=[(-2.0, 2.0)],
    bo_cfg=bo.BOConfig(n_init=5, n_iter=15, random_state=0),
    gp_cfg=bo.GPConfig(),
    acq_cfg=bo.AcqConfig(),
    optim_cfg=bo.OptimConfig(),
    save_cfg=bo.SaveConfig(),
)

print(res.best_x, res.best_y)
```

The multi-fidelity and multi-objective drivers follow the same pattern:
`from pyRAMBO import mfbo` (`mfbo.multi_fidelity_bayesian_optimization`) and
`from pyRAMBO import mobo` (`mobo.multi_objective_bayesian_optimization`).

See [`examples/`](examples/) for full, runnable tutorials (1D/2D benchmark
functions, plotting, two-tier persistence, hyperparameter optimization).

## Persistence

`SaveConfig` controls two tiers of run persistence, both numpy/JSON-based
(no HDF5, no pickle):

- **Tier 1 (trace, always on):** a cheap per-iteration record -- proposed x,
  observed y, incumbent best x/y, wall-clock time, acquisition value -- kept
  in memory (`BOResult.trace`) and written to `trace.npz` if `out_path` is set.
- **Tier 2 (snapshots, opt-in via `snapshot_enabled=True`):** everything
  needed to refit the GP exactly at a given iteration (design set,
  hyperparameters, kernel id, normalization state) -- never the dense
  covariance/Cholesky factors. Reload with `bo.load_run(out_path)` and call
  `.posterior(it).predict(Xgrid)` to reconstruct predictions bit-for-bit
  without re-running the objective. See
  [`examples/sbo/main_persistence_demo.py`](examples/sbo/main_persistence_demo.py).

## Project layout

One package, `pyRAMBO`, with three sibling subpackages, one per flavour of
Bayesian optimization. They share a deliberately identical module structure
but no code: each is independent, with no common base module and no imports
between them.

```
src/pyRAMBO/
    __init__.py     exposes the three subpackages below
    sbo/            single-fidelity, single-objective BO
    mfbo/           multi-fidelity BO (AR1 / Kennedy & O'Hagan)
    mobo/           multi-objective BO (ICM GP + Monte-Carlo EHVI)
        core.py         configs, GP model, acquisition, BO driver
        saving.py       experiment folders, run logging
        persistence.py  two-tier (trace / snapshot) run persistence and replay
        plotting.py     all matplotlib-based visualization (imported lazily)
legacy/         the exploratory scripts the packages were ported from
examples/       runnable tutorials -- see examples/README.md
tests/          pytest test suite
paper/          JOSS paper (paper.md, paper.bib)
```

In every subpackage, `core.py` has no dependency on matplotlib/logging beyond
the `saving`/`plotting`/`persistence` modules it composes; `plotting.py` and
`persistence.py` are only imported the first time a run actually needs them.

The example folders are named after the subpackages they demonstrate
(`examples/mobo/` for `pyRAMBO.mobo`). The examples only ever add the
repository **root** to `sys.path` and reach shared code as
`examples.benchmarks`.
[`examples/README.md`](examples/README.md) explains the rule and how to run
the examples.

## Running the tests

```bash
pytest
```

## License

MIT, see [LICENSE](LICENSE).

## Citation

If you use this software, please cite it as described in
[`paper/paper.md`](paper/paper.md) (JOSS submission in preparation).
