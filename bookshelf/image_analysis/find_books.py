"""Find books on a book shelf."""
from collections.abc import Sequence
from itertools import pairwise, product
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from loguru import logger

from ..plotting import plot_lines, set_axes_off
from .models import Book, Line, Shelf
from .utilities import drop_zero_rows_cols, find_lines, windows


def find_book_lines(img: np.ndarray) -> tuple[list[Line], list[np.ndarray]]:
    """Find the horizontal lines that separate shelves."""
    gray = ski.color.rgb2gray(img)
    equalized = ski.exposure.equalize_adapthist(gray)
    sobel = ski.filters.sobel_v(equalized)
    canny = ski.feature.canny(sobel, sigma=1.5)
    tested_angles = np.linspace(np.pi * 5 / 6, np.pi * 7 / 6, 101, endpoint=True)
    return find_lines(
        canny,
        tested_angles,
        min_dist=20,
        min_hspace_threshold=200,
    ).lines, [
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
    fig, axes = plt.subplots(ncols=n_imgs, figsize=(width, height), squeeze=False)
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
    mask = img.copy()
    logger.trace("Isolating book.")
    for r, c in product(range(img.shape[0]), range(img.shape[1])):
        if not (left.col(r) < c < right.col(r)):
            mask[r, c] = 0
    mask = drop_zero_rows_cols(mask)
    return Book(key=key, left=left, right=right, image=mask)


def _plot_books(books: Sequence[Book], output_fname: Path) -> None:
    ncols = min(5, len(books))
    nrows = ceil(len(books) / ncols)
    width = 10
    height = width / ncols * nrows
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(width, height), squeeze=False
    )
    set_axes_off(axes)
    for ax, book in zip(axes.flatten(), books, strict=False):
        ax.imshow(book.image)
    fig.savefig(output_fname, dpi=300)


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
    logger.debug(f"Isolating books from shelf {shelf.key}.")
    books: list[Book] = []
    _half_row = shelf.image.shape[0] // 2
    for w, (window, _) in enumerate(windows(shelf.image)):
        book_lines, prepped_img = find_book_lines(window)
        logger.debug(f"Found {len(book_lines)} lines in window {w}.")
        if len(book_lines) < 2:  # noqa: PLR2004
            continue

        book_lines.sort(key=lambda line: line.col(_half_row))
        _plot_book_lines(
            window,
            prepped_img,
            lines=book_lines,
            key=f"{shelf.key}_{w}",
            output_dir=output_dir,
        )
        for b, (left, right) in enumerate(pairwise(book_lines)):
            book = isolate_book(window, left, right, key=f"{shelf.key}_w{w}_b{b}")
            if book.image.size > 0:
                books.append(book)
    if output_dir is not None and len(books) > 0:
        _plot_books(books, output_dir / f"{shelf.key}_all-books.jpeg")
    logger.info(f"Found {len(books)} books for shelf {shelf.key}")
    return books
