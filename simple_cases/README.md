# Simple cases

Runnable tutorials for pyRAMBO on analytical benchmark functions. Each one is
a self-contained script, split into `#%%` cells so it can also be stepped
through in Spyder or VS Code.

| Example | Subpackage | Shows |
| --- | --- | --- |
| [`sbo/main_1D_sin.py`](sbo/main_1D_sin.py) | `sbo` | The basic loop on a noisy 1D objective |
| [`sbo/main_1D_sin_large_scale.py`](sbo/main_1D_sin_large_scale.py) | `sbo` | Input/output normalization on a large-scale objective |
| [`sbo/main_2D_branin.py`](sbo/main_2D_branin.py) | `sbo` | 2D Branin, with posterior/acquisition contour maps |
| [`sbo/main_2D_rosenbrock.py`](sbo/main_2D_rosenbrock.py) | `sbo` | 2D Rosenbrock, a harder anisotropic valley |
| [`sbo/main_persistence_demo.py`](sbo/main_persistence_demo.py) | `sbo` | Two-tier persistence: reload a run from disk and re-plot it |

## Running them

From the repository root:

```bash
python -m simple_cases.sbo.main_1D_sin_large_scale
```

Running the file directly also works — `python simple_cases/sbo/main_1D_sin_large_scale.py`,
or the "Run file" button in Spyder / VS Code — from any working directory. No
installation step is needed: the scripts import pyRAMBO straight out of
`src/`. If you have run `pip install -e .`, that keeps working too, with the
source checkout taking precedence.

The scripts save figures, so they need a LaTeX installation (TeX Live or
MiKTeX, with `dvipng`); see the note in the main README.

## Layout

```
simple_cases/
    __init__.py      puts src/ on sys.path; checks pyRAMBO comes from the checkout
    _common.py       shared plot style and output locations
    benchmarks.py    the objective functions every example draws from
    sbo/             the tutorials
    _output/         all generated figures, logs and traces (gitignored)
```

Every example writes to `simple_cases/_output/<example_name>/` via
`_common.output_dir()`. That path is absolute, so results land in the same
place no matter where the interpreter was started.

## The one rule: never put `simple_cases/` on `sys.path`

The scripts add only the repository **root** to `sys.path`:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

and then reach shared code through the package -- `from simple_cases.benchmarks
import ...`, never a bare `import benchmarks` -- and the library through its
namespace -- `from pyRAMBO import sbo`. Since the library lives under
`pyRAMBO`, the example folder `sbo/` cannot shadow it.

`simple_cases/__init__.py` also verifies, on import, that `pyRAMBO` really did
resolve to `src/pyRAMBO`, and raises a pointed `ImportError` if not (for
instance if a stale installed copy was imported first).

## Adding a new example

1. Put it in the folder for the subpackage it demonstrates.
2. Copy the import header from any existing example, changing only the
   `python -m ...` line, the benchmark names and the library import.
3. Take the output path from `_common.output_dir("<name>")` — never a
   relative `"./..."` path.
4. Call `_common.use_paper_style()` instead of repeating the `plt.rc` block.
5. Put any new objective function in `benchmarks.py` rather than defining it
   inline, so other examples and tests can reuse it.
