"""
Runnable examples / tutorials for the sbo, mfbo and mobo packages.

This file is the SINGLE place where the examples are made importable, and it
exists to close one specific trap. The example folders are named after the
libraries they demonstrate -- examples/sbo, examples/mfbo, examples/mobo --
so they have exactly the same top-level names as the real packages in src/.
Putting ``examples/`` itself on sys.path therefore makes ``import mobo``
resolve to the (empty) example folder rather than to the library, and the
failure is silent: the import succeeds and only blows up later with an
AttributeError on the first library call.

The rule that avoids this permanently:

    Only the repository ROOT ever goes on sys.path -- never ``examples/``.

The example folders are then only ever reachable as ``examples.mobo``, which
cannot collide with ``mobo``, and shared code is imported as
``examples.benchmarks`` rather than as a bare top-level ``benchmarks``
module. Importing anything from this package prepends ``src/`` to sys.path,
so the scripts work straight out of a fresh clone with no install step; if
you do ``pip install -e .`` that keeps working too, with the checkout taking
precedence (which is what you want while developing).

``_assert_library_not_shadowed`` turns the trap into a loud, actionable
error should it ever be re-introduced.

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
EXAMPLES_ROOT = REPO_ROOT / "examples"

# The library packages demonstrated here. Each one has a same-named folder
# under examples/, which is precisely why the shadowing guard below exists.
LIBRARY_PACKAGES = ("sbo", "mfbo", "mobo")

__all__ = ["REPO_ROOT", "SRC_ROOT", "EXAMPLES_ROOT", "LIBRARY_PACKAGES"]


def _prepend_src_to_sys_path() -> None:
    """Put ``src/`` first on sys.path, so a source checkout always wins over
    an installed copy of the same packages."""

    if not SRC_ROOT.is_dir():
        return  # installed-only layout: rely on the installed packages

    entry = str(SRC_ROOT)
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)


def _assert_library_not_shadowed() -> None:
    """Fail loudly if ``import sbo`` / ``mfbo`` / ``mobo`` would resolve to
    an example folder instead of the real package.

    Cannot happen through this package's own bootstrap; it can still happen
    if a script adds ``examples/`` to sys.path by hand, or imports a
    shadowed package before importing ``examples``."""

    if not SRC_ROOT.is_dir():
        return

    for name in LIBRARY_PACKAGES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue

        if spec is None:
            continue

        if spec.origin is None:
            # A namespace package: a bare directory with no __init__.py won
            # the lookup. That is the shadowing failure mode itself.
            locations = list(getattr(spec, "submodule_search_locations", []) or [])
            raise ImportError(
                f"'{name}' resolved to a namespace package at {locations}, not to the "
                f"library in {SRC_ROOT / name}. Something put a directory named "
                f"'{name}' on sys.path ahead of src/ -- most likely 'examples/'. "
                f"Import 'examples.{name}' instead of adding 'examples/' to sys.path."
            )

        origin = Path(spec.origin).resolve()
        if SRC_ROOT not in origin.parents:
            raise ImportError(
                f"'{name}' resolved to {origin}, not to the library in "
                f"{SRC_ROOT / name}. If that path is under 'examples/', an example "
                f"folder is shadowing the package: only the repository root may go "
                f"on sys.path, never 'examples/' itself. See examples/README.md."
            )


_prepend_src_to_sys_path()
_assert_library_not_shadowed()
