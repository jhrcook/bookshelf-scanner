"""General image analysis functions.."""

from typing import Generator

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


def windows(
    img: np.ndarray, n_windows: int = 5, overlap: int | None = None
) -> Generator[tuple[np.ndarray, tuple[int, int]], None, None]:
    """Scan an image with horizontal windows.

    Args:
    ----
        img (np.ndarray): Original image.
        n_windows (int, optional): Number of windows. Defaults to 5.
        overlap (int | None, optional): Amount of overlap between windows. Defaults to None.

    Yields:
    ------
        Generator[np.ndarray, None, None]: Generator of windows over the image. The column boundaries for the window are also provided for each window.
    """
    W = img.shape[1]  # noqa: N806
    N = n_windows  # noqa: N806
    if overlap is None:
        p = round(0.2 * (W / N))  # Default of ~20% overlap with previous window.
    else:
        p = overlap
    w: int = round((W + (p * (N - 1))) / N)
    x1: int = 0
    for _ in range(N):
        x2 = x1 + w
        yield img.copy()[:, x1:x2], (x1, x2)
        x1 = x2 - p
