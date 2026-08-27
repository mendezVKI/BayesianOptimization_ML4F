# Examples

Runnable tutorials for the three Bayesian-optimization packages in this
repository. Each one is a self-contained script, split into `#%%` cells so it
can also be stepped through in Spyder or VS Code.

| Example | Package | Shows |
| --- | --- | --- |
| [`sbo/main_1D_sin.py`](sbo/main_1D_sin.py) | `sbo` | The basic loop on a noisy 1D objective |
| [`sbo/main_1D_sin_large_scale.py`](sbo/main_1D_sin_large_scale.py) | `sbo` | Input/output normalization on a large-scale objective |
| [`sbo/main_2D_branin.py`](sbo/main_2D_branin.py) | `sbo` | 2D Branin, with posterior/acquisition contour maps |
| [`sbo/main_2D_rosenbrock.py`](sbo/main_2D_rosenbrock.py) | `sbo` | 2D Rosenbrock, a harder anisotropic valley |
| [`sbo/main_persistence_demo.py`](sbo/main_persistence_demo.py) | `sbo` | Two-tier persistence: reload a run from disk and re-plot it |
| [`mfbo/main_mfbo_forrester.py`](mfbo/main_mfbo_forrester.py) | `mfbo` | Two-fidelity BO on the Forrester benchmark |
| [`mobo/main_mobo_schaffer.py`](mobo/main_mobo_schaffer.py) | `mobo` | Bi-objective BO in 1D: the whole state in one figure |
| [`mobo/main_mobo_binh_korn.py`](mobo/main_mobo_binh_korn.py) | `mobo` | Bi-objective BO in 2D, with a learned output correlation |

## Running them

From the repository root:

```bash
python -m examples.mobo.main_mobo_schaffer
```

Running the file directly also works — `python examples/mobo/main_mobo_schaffer.py`,
or the "Run file" button in Spyder / VS Code — from any working directory. No
installation step is needed: the scripts import the packages straight out of
`src/`. If you have run `pip install -e .`, that keeps working too, with the
source checkout taking precedence.

## Layout

```
examples/
    __init__.py      puts src/ on sys.path; guards against the shadowing trap below
    _common.py       shared plot style and output locations
    benchmarks.py    the objective functions every example draws from
    sbo/  mfbo/  mobo/
    _output/         all generated figures, logs and traces (gitignored)
```

Every example writes to `examples/_output/<example_name>/` via
`_common.output_dir()`. That path is absolute, so results land in the same
place no matter where the interpreter was started — previously the output
paths were relative, which is how run folders ended up scattered inside
`src/`.

## The one rule: never put `examples/` on `sys.path`

The example folders are named after the packages they demonstrate, so
`examples/mobo/` has the same top-level name as the real `mobo` package in
`src/`. If `examples/` is on `sys.path`, then `import mobo` resolves to the
example folder rather than the library — and it does so **silently**: the
import succeeds, and the failure only surfaces later as
`AttributeError: module 'mobo' has no attribute ...`.

So the scripts add only the repository **root** to `sys.path`:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

and then reach shared code through the package — `from examples.benchmarks
import ...`, never a bare `import benchmarks`. `examples.mobo` cannot collide
with `mobo`, so the trap is closed by construction.

`examples/__init__.py` also verifies, on import, that `sbo`, `mfbo` and
`mobo` really did resolve to `src/`, and raises a pointed `ImportError` if
not. If you ever see that error, something added `examples/` to the path.

## Adding a new example

1. Put it in the folder for the package it demonstrates.
2. Copy the import header from any existing example, changing only the
   `python -m ...` line, the benchmark names and the library import.
3. Take the output path from `_common.output_dir("<name>")` — never a
   relative `"./..."` path.
4. Call `_common.use_paper_style()` instead of repeating the `plt.rc` block.
5. Put any new objective function in `benchmarks.py` rather than defining it
   inline, so other examples and tests can reuse it.
