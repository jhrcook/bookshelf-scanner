"""Find books on a book shelf."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

from ..plotting import plot_lines, set_axes_off
from .models import Book, Line, Shelf
from .utilities import find_lines, windows


def find_book_lines(img: np.ndarray) -> tuple[list[Line], list[np.ndarray]]:
    """Find the horizontal lines that separate shelves."""
    gray = ski.color.rgb2gray(img)
    equalized = ski.exposure.equalize_adapthist(gray)
    sobel = ski.filters.sobel_v(equalized)
    canny = ski.feature.canny(sobel, sigma=1.5)
    tested_angles = np.linspace(np.pi * 5 / 6, np.pi * 7 / 6, 101, endpoint=True)
    return find_lines(canny, tested_angles, min_dist=20), [
        gray,
        equalized,
        sobel,
        canny,
    ]


def _plot_book_lines(
    img: np.ndarray,
    processed_imgs: list[np.ndarray],
    lines: list[Line],
    key: str,
    output_dir: Path | None,
) -> None:
    if output_dir is None:
        return
    width = 9
    n_imgs = len(processed_imgs) + 1
    height = width / (n_imgs * img.shape[1]) * img.shape[0]
    fig, axes = plt.subplots(ncols=n_imgs, figsize=(width, height))
    axes = axes.flatten()
    for i in range(len(processed_imgs)):
        axes[i].imshow(processed_imgs[i])
        axes[i].set_title(f"step {i+1}")
    plot_lines(img, lines=lines, ax=axes[-1])
    axes[-1].set_title("lines")
    set_axes_off(axes)
    fig.savefig(output_dir / f"{key}_books.jpeg", dpi=300)


def isolate_book(img: np.ndarray, left: Line, right: Line, key: str) -> Book:
    """Isolate a shelf from the original image given its top and bottom lines."""
    width = img.shape[1]
    # dst = np.array(
    #     [
    #         [0, top.row_intercept],
    #         [width, top.row(width)],
    #         [width, bottom.row(width)],
    #         [0, bottom.row_intercept],
    #     ]
    # )
    # TODO: figure out boundaries for original image.
    dst = np.array([])
    height = np.mean([dst[3, 1] - dst[0, 1], dst[2, 1] - dst[1, 1]]).round()
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    tform = ski.transform.ProjectiveTransform()
    tform.estimate(src, dst)
    warped: np.ndarray = ski.transform.warp(img, tform, output_shape=(height, width))
    return Book(key=key, left=left, right=right, image=warped)


def find_books(shelf: Shelf, output_dir: Path | None = None) -> list[Book]:
    """Find books on a shelf.

    Note there may be duplicates because of windowed search.

    Args:
    ----
        shelf (Shelf): Shelf extracted from an image.
        output_dir (Path | None, optional): Output directory where intermediate results can be saved. Defaults to None.

    Returns:
    -------
        list[Book]: List of books isolated from the shelf.
    """
    for i, (window, _) in enumerate(windows(shelf.image)):
        book_lines, prepped_img = find_book_lines(window)
        book_lines.sort(key=lambda line: line.row_intercept)
        _plot_book_lines(
            window,
            prepped_img,
            lines=book_lines,
            key=f"{shelf.key}_{i}",
            output_dir=output_dir,
        )
    return []
