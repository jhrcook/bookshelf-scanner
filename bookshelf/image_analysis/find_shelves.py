"""Isolate shelves in an image."""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from loguru import logger

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


def _plot_shelves(shelves: list[Shelf], output_dir: Path | None) -> None:
    if output_dir is None:
        return
    width = 9
    img_heights = [s.image.shape[0] for s in shelves]
    img_height = sum(img_heights)
    img_width = shelves[0].image.shape[1]
    height = width / img_width * img_height
    fig, axes = plt.subplots(
        nrows=len(shelves), figsize=(width, height), height_ratios=img_heights
    )
    axes = axes.flatten()
    for ax, shelf in zip(axes, shelves, strict=False):
        ax.imshow(shelf.image)
        ax.set_axis_off()
    fig.savefig(output_dir / "shelves-isolated.jpeg", dpi=300)


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


def find_shelves(
    img: np.ndarray, output_dir: Path | None = None, min_shelf_height: int = 300
) -> list[Shelf]:
    """Isolate the shelves from an image of a bookshelf.

    Args:
    ----
        img (np.ndarray): Image of a bookshelf.
        output_dir (Path | None, optional): Output directory where intermediate results can be saved. Defaults to None.
        min_shelf_height (int): Minimum height for a shelf. Defaults to 300.

    Returns:
    -------
        list[Shelf]: Isolated shelves.
    """
    shelf_lines = find_shelf_lines(img)
    shelf_lines.sort(key=lambda line: line.row_intercept)
    shelf_lines.insert(0, Line(point=Point(row=0, col=0), slope=0))
    shelf_lines.append(Line(point=Point(row=img.shape[0], col=0), slope=0))
    _plot_shelf_lines(img, lines=shelf_lines, output_dir=output_dir)

    shelves = [
        isolate_shelf(img, t, b, key=f"shelf-{i}")
        for i, (t, b) in enumerate(itertools.pairwise(shelf_lines))
    ]
    shelves = list(filter(lambda s: s.image.shape[0] >= min_shelf_height, shelves))
    logger.info(f"Found {len(shelves)} shelves.")
    _plot_shelves(shelves, output_dir)
    return shelves
