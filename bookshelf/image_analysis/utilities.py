"""General image analysis functions.."""

import numpy as np
import skimage as ski

from .models import Line, Point


def find_lines(
    prepped_img: np.ndarray, tested_angles: np.ndarray, min_dist: int
) -> list[Line]:
    """Find lines in an image.

    Args:
    ----
        prepped_img (np.ndarray): Pre-processed image.
        tested_angles (np.ndarray): Angles to test.
        min_dist (int): Minimum distance between lines in the final collection of best lines.

    Returns:
    -------
        list[Line]: List of best lines found.
    """
    hspace, angles, distances = ski.transform.hough_line(
        prepped_img, theta=tested_angles
    )
    hspace, angles, distances = ski.transform.hough_line_peaks(
        hspace, angles, distances, min_distance=min_dist
    )
    lines = []
    for theta, r in zip(angles, distances, strict=False):
        x, y = r * np.array([np.cos(theta), np.sin(theta)])
        slope = np.tan(theta + np.pi / 2)
        lines.append(Line(point=Point(row=y, col=x), slope=slope))
    return lines


def filter_otsu(img: np.ndarray) -> np.ndarray:
    """Otsu thresholding filter."""
    return img > ski.filters.threshold_otsu(img)
