"""
Shared helpers for the example scripts: plot styling and output locations.

Everything here used to be copy-pasted into the top of every example, which
is how the two drifted apart (some scripts set the matplotlib style, some
did not; every script wrote its results to a *relative* path, so the output
landed wherever the interpreter happened to be started -- which is how
run folders ended up inside src/).

@authors: Yannick Lecomte and Miguel A. Mendez
"""

from __future__ import annotations

import shutil
import warnings

import matplotlib.pyplot as plt

from . import EXAMPLES_ROOT

# All example output goes here, and nowhere else. Absolute, so it does not
# depend on the working directory the script was launched from. Gitignored.
OUTPUT_ROOT = EXAMPLES_ROOT / "_output"

__all__ = ["OUTPUT_ROOT", "output_dir", "latex_available", "use_paper_style"]


def output_dir(name: str, clean: bool = False) -> str:
    """Absolute path of the output folder for one example, created on demand.

    ``clean=True`` removes any previous contents first, for the examples that
    need to start from an empty folder to demonstrate something (the
    persistence demo reloads a run from disk, so a stale folder would make
    the result ambiguous).

    Returns a ``str`` because the SaveConfig containers and the os.path.join
    calls throughout the examples expect one."""

    path = OUTPUT_ROOT / name

    if clean and path.is_dir():
        # Empty the folder rather than removing and recreating it: deleting
        # the directory itself intermittently raises PermissionError on
        # Windows when it sits in a synced folder (OneDrive) or is open in a
        # file browser, and re-creating it buys nothing over emptying it.
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass

    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def latex_available() -> bool:
    """Whether matplotlib's usetex backend can actually run: it shells out to
    a real LaTeX installation, and fails at DRAW time (not at import) when
    one is missing."""

    return all(shutil.which(exe) is not None for exe in ("latex", "dvipng"))


def use_paper_style(usetex: bool | None = None) -> None:
    """Apply the serif / LaTeX plot style shared by all examples.

    ``usetex=None`` (the default) enables it only if a LaTeX toolchain is
    actually on PATH, so a fresh clone still runs on a machine without one.
    """

    if usetex is None:
        usetex = latex_available()
        if not usetex:
            warnings.warn(
                "No LaTeX toolchain found on PATH (needs 'latex' and 'dvipng'), so "
                "the examples fall back to matplotlib's built-in mathtext. NOTE: the "
                "sbo/mfbo/mobo plotting modules set usetex=True unconditionally when "
                "they are imported, so figures drawn by the library itself will still "
                "attempt LaTeX. Install TeX Live or MiKTeX, or disable the library "
                "plots via SaveConfig(plt_all=False).",
                RuntimeWarning,
                stacklevel=2,
            )

    plt.rc("text", usetex=usetex)
    plt.rc("font", family="serif")
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    plt.rc("axes", labelsize=12)
