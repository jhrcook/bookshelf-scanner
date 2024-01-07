"""Isolate shelves in an image."""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

from ..plotting import plot_lines
from .models import Line, Point, Shelf
from .utilities import find_lines


def find_shelf_lines(img: np.ndarray) -> list[Line]:
    """Find the horizontal lines that separate shelves."""
    prepped_img = ski.color.rgb2gray(img)
    prepped_img = ski.exposure.equalize_adapthist(prepped_img)
    prepped_img = ski.filters.sobel_h(prepped_img)
    prepped_img = ski.feature.canny(prepped_img, sigma=1)
    tested_angles = np.linspace(np.pi * 4 / 3, np.pi * 5 / 3, 101, endpoint=True)
    return find_lines(prepped_img, tested_angles, min_dist=100)


def isolate_shelf(img: np.ndarray, top: Line, bottom: Line, key: str) -> Shelf:
    """Isolate a shelf from the original image given its top and bottom lines."""
    width = img.shape[1]
    dst = np.array(
        [
            [0, top.row_intercept],
            [width, top.row(width)],
            [width, bottom.row(width)],
            [0, bottom.row_intercept],
        ]
    )
    height = np.mean([dst[3, 1] - dst[0, 1], dst[2, 1] - dst[1, 1]]).round()
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    tform = ski.transform.ProjectiveTransform()
    tform.estimate(src, dst)
    warped: np.ndarray = ski.transform.warp(img, tform, output_shape=(height, width))
    return Shelf(key=key, top=top, bottom=bottom, image=warped)


def _plot_shelf(shelf: Shelf, output_dir: Path | None) -> None:
    if output_dir is None:
        return
    ...


def _plot_shelf_lines(
    img: np.ndarray, lines: list[Line], output_dir: Path | None
) -> None:
    if output_dir is None:
        return
    width = 9
    height = width / img.shape[1] * img.shape[0]
    fig, ax = plt.subplots(figsize=(width, height))
    plot_lines(img, lines=lines, ax=ax)
    ax.set_title("results")
    fig.savefig(output_dir / "shelves.jpeg", dpi=300)


def find_shelves(img: np.ndarray, output_dir: Path | None = None) -> list[Shelf]:
    """Isolate the shelves from an image of a bookshelf.

    Args:
    ----
        img (np.ndarray): Image of a bookshelf.
        output_dir (Path | None, optional): Output directory where intermediate results can be saved. Defaults to None.

    Returns:
    -------
        list[Shelf]: Isolated shelves.
    """
    shelf_lines = find_shelf_lines(img)
    shelf_lines.sort(key=lambda line: line.row_intercept)
    shelf_lines.insert(0, Line(point=Point(row=0, col=0), slope=0))
    shelf_lines.append(Line(point=Point(row=img.shape[0], col=0), slope=0))
    _plot_shelf_lines(img, lines=shelf_lines, output_dir=output_dir)

    shelves = []
    for i, (top, bottom) in enumerate(itertools.pairwise(shelf_lines)):
        shelf = isolate_shelf(img, top, bottom, key=f"shelf-{i}")
        _plot_shelf(shelf, output_dir)
        shelves.append(shelf)
    return shelves
