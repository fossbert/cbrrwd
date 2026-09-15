import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from cbrrwd.plotting import _color_light_or_dark, sqrt_ax


def test_sqrt_ax_sets_function_scale_on_requested_axis():
    fig, ax = plt.subplots()
    sqrt_ax("y", ax=ax)
    assert ax.get_yscale() == "function"
    assert ax.get_xscale() == "linear"
    plt.close(fig)


def test_sqrt_ax_rejects_invalid_axis():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="axis must be one of"):
        sqrt_ax("z", ax=ax)
    plt.close(fig)


def test_color_light_or_dark():
    white = np.array([1.0, 1.0, 1.0, 1.0])
    black = np.array([0.0, 0.0, 0.0, 1.0])
    assert _color_light_or_dark(white) == "k"
    assert _color_light_or_dark(black) == "w"
