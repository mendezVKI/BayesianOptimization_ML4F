"""
Runnable examples / tutorials for pyRAMBO (pyRAMBO.sbo, pyRAMBO.mfbo and
pyRAMBO.mobo).

This file is the SINGLE place where the examples are made importable. The
rule it relies on:

    Only the repository ROOT ever goes on sys.path -- never ``examples/``.

The example folders are then only ever reachable as ``examples.sbo`` etc.,
and shared code is imported as ``examples.benchmarks`` rather than as a bare
top-level ``benchmarks`` module. Because the libraries live under the single
``pyRAMBO`` namespace, the example folders (named sbo/, mfbo/, mobo/) cannot
shadow them. Importing anything from this package prepends ``src/`` to
sys.path, so the scripts work straight out of a fresh clone with no install
step; if you do ``pip install -e .`` that keeps working too, with the
checkout taking precedence (which is what you want while developing).

``_assert_library_from_checkout`` turns any mix-up (e.g. a stale installed
copy of pyRAMBO winning over src/) into a loud, actionable error.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
EXAMPLES_ROOT = REPO_ROOT / "simple_cases"

# The library package demonstrated here.
LIBRARY_PACKAGE = "pyRAMBO"

__all__ = ["REPO_ROOT", "SRC_ROOT", "EXAMPLES_ROOT", "LIBRARY_PACKAGE"]


def _prepend_src_to_sys_path() -> None:
    """Put ``src/`` first on sys.path, so a source checkout always wins over
    an installed copy of the same package."""

    if not SRC_ROOT.is_dir():
        return  # installed-only layout: rely on the installed package

    entry = str(SRC_ROOT)
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)


def _assert_library_from_checkout() -> None:
    """Fail loudly if ``import pyRAMBO`` would not resolve to ``src/pyRAMBO``.

    Cannot happen through this package's own bootstrap; it can still happen
    if pyRAMBO was already imported (from somewhere else) before importing
    ``examples``."""

    if not SRC_ROOT.is_dir():
        return

    name = LIBRARY_PACKAGE
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return

    if spec is None:
        return

    if spec.origin is None:
        # A namespace package: a bare directory with no __init__.py won the
        # lookup.
        locations = list(getattr(spec, "submodule_search_locations", []) or [])
        raise ImportError(
            f"'{name}' resolved to a namespace package at {locations}, not to the "
            f"library in {SRC_ROOT / name}. Something put a directory named "
            f"'{name}' on sys.path ahead of src/."
        )

    origin = Path(spec.origin).resolve()
    if SRC_ROOT not in origin.parents:
        raise ImportError(
            f"'{name}' resolved to {origin}, not to the library in "
            f"{SRC_ROOT / name}. It was probably imported before 'examples'; "
            f"import 'examples' first. See examples/README.md."
        )


_prepend_src_to_sys_path()
_assert_library_from_checkout()
