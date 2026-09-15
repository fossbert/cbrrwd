"""Small matplotlib helpers used across RWD figures: a square-root axis scale
and heatmap cell-value labels with automatic light/dark text colour.

Needs the ``plotting`` extra: ``pip install cbrrwd[plotting]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["sqrt_ax", "label_heatmap"]


def _fwd_sqrt(vals):
    return np.sqrt(vals)


def _rev_sqrt(vals):
    return np.power(vals, 2)


def sqrt_ax(axis: str, ax=None):
    """Apply a square-root scale to one or both axes of ``ax`` (default: current axes)."""

    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    axis_options = ["x", "y", "xy"]

    if axis not in axis_options:
        raise ValueError(f"axis must be one of {' '.join(axis_options)}")

    if axis == "x":
        ax.set_xscale("function", functions=(_fwd_sqrt, _rev_sqrt))
    elif axis == "y":
        ax.set_yscale("function", functions=(_fwd_sqrt, _rev_sqrt))
    else:
        ax.set_xscale("function", functions=(_fwd_sqrt, _rev_sqrt))
        ax.set_yscale("function", functions=(_fwd_sqrt, _rev_sqrt))

    return ax


def _color_light_or_dark(rgba_in: np.ndarray) -> str:
    """Determine whether an RGBA colour is light or dark, for contrasting text colour.

    See https://stackoverflow.com/questions/22603510/is-this-possible-to-detect-a-colour-is-a-light-or-dark-colour
    """
    r, g, b, _ = rgba_in * 255
    hsp = np.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
    return "k" if hsp > 127.5 else "w"


def label_heatmap(mesh, ncols: int, nrows: int, labels: pd.DataFrame, ax=None):
    """Draw ``labels`` as text on top of a heatmap ``mesh`` (e.g. from ``pcolormesh``),
    picking black or white text per cell based on that cell's fill colour.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    mesh.update_scalarmappable()

    xpos, ypos = np.meshgrid(np.arange(ncols), np.arange(nrows))

    for j, i, rgba_in in zip(xpos.flat, ypos.flat, mesh.get_facecolors()):

        lbl = labels.iloc[i, j]

        col = _color_light_or_dark(rgba_in)
        ax.text(j + 0.5, i + 0.5, lbl, ha="center", va="center", color=col, fontsize="x-small")
