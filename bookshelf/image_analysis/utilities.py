"""General image analysis functions.."""

from dataclasses import dataclass
from typing import Generator

import numpy as np
import skimage as ski
from loguru import logger

from .models import Line, Point


@dataclass
class FoundLines:

    """Results from searching for lines in an image."""

    hspace: np.ndarray
    angles: np.ndarray
    distances: np.ndarray
    lines: list[Line]


def find_lines(
    prepped_img: np.ndarray,
    tested_angles: np.ndarray,
    *,
    min_dist: int,
    min_hspace_threshold: float | None = None,
) -> FoundLines:
    """Find lines in an image.

    Args:
    ----
        prepped_img (np.ndarray): Pre-processed image.
        tested_angles (np.ndarray): Angles to test.
        min_dist (int): Minimum distance between lines in the final collection of best lines.
        min_hspace_threshold (float | None, optional): Minimum threshold for intensity peaks (of `hspace`). The threshold used will be the maximum of this value and `0.5 max(hspace)` (which is the default in 'skimage').

    Returns:
    -------
        list[Line]: List of best lines found.
    """
    hspace, angles, distances = ski.transform.hough_line(
        prepped_img, theta=tested_angles
    )
    hspace_threshold: float | None = None
    if min_hspace_threshold is not None:
        hspace_threshold = max(hspace.max() * 0.5, min_hspace_threshold)

    hspace, angles, distances = ski.transform.hough_line_peaks(
        hspace, angles, distances, min_distance=min_dist, threshold=hspace_threshold
    )
    if hspace.size > 0:
        logger.debug(
            f"hspace avg: {hspace.mean().round(3)}  max: {hspace.max().round(3)}"
        )
    else:
        logger.debug("No lines found.")

    lines = []
    for theta, r in zip(angles, distances, strict=False):
        x, y = r * np.array([np.cos(theta), np.sin(theta)])
        slope = np.tan(theta + np.pi / 2)
        lines.append(Line(point=Point(row=y, col=x), slope=slope))
    return FoundLines(hspace=hspace, angles=angles, distances=distances, lines=lines)


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


def _trim_zero_cols_from_left(ary: np.ndarray) -> np.ndarray:
    i = 0
    while i < (ary.shape[1] - 1) and (ary[:, i] == 0).all():
        i += 1
    return ary[:, i:]


def drop_zero_rows_cols(ary: np.ndarray) -> np.ndarray:
    """Drop columns and rows that are all 0s and at the edge of the image."""
    new_ary = ary.copy()
    for _ in range(4):
        new_ary = _trim_zero_cols_from_left(new_ary)
        new_ary = np.rot90(new_ary)
    return new_ary
