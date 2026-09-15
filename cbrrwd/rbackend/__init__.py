"""Optional R-backed functionality (r x c Fisher's exact test, tumgr growth/decay model).

Everything here needs the ``r`` extra (``pip install cbrrwd[r]``) plus the
relevant R packages. Imports are deferred to call time so that ``import cbrrwd``
works without R installed.
"""

from __future__ import annotations

import functools

_R_EXTRA_HINT = (
    "This feature needs the 'r' extra: pip install 'cbrrwd[r]'  "
    "(and the R package(s): {pkgs})."
)


def require_rpy2():
    """Import and return the rpy2 pieces used across the R backend."""
    try:
        from rpy2 import robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_R_EXTRA_HINT.format(pkgs="(base R only)")) from exc
    return ro, numpy2ri, pandas2ri, localconverter, importr


@functools.lru_cache(maxsize=None)
def r_package(name: str):
    """Import an R package by name, with a helpful error if it is missing."""
    _, _, _, _, importr = require_rpy2()
    try:
        return importr(name)
    except Exception as exc:  # rpy2 raises PackageNotInstalledError (subclass of Exception)
        raise ImportError(_R_EXTRA_HINT.format(pkgs=name)) from exc
