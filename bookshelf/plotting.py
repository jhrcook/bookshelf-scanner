"""Plotting functions and helpers."""

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .image_analysis.models import Line


def plot_lines(img: np.ndarray, lines: Iterable[Line], ax: Axes | None = None) -> None:
    """Plot lines on an image."""
    if ax is None:
        _, ax = plt.subplots()
        assert isinstance(ax, Axes)

    for line in lines:
        ax.axline(line.point.xy, slope=line.slope, color="blue", lw=1.5)

    ax.set_axis_off()
    ax.imshow(img)
