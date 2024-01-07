"""Bookshelf scanning library and application."""

from pathlib import Path

import numpy as np
import skimage as ski

from .image_analysis import scan_bookshelf

__version__ = "0.0.0.9000"


def scan(
    img: np.ndarray | Path, output_dir: Path | None = None
) -> list[scan_bookshelf.BookData]:
    """Scan an image of a bookshelf for book information.

    Args:
    ----
        img (np.ndarray | Path): Input image of bookshelf.
        output_dir (Path | None, optional): Output directory where intermediate results can be saved to. Defaults to None.

    Returns:
    -------
        list[scan_bookshelf.BookData]: Book data.
    """
    if isinstance(img, Path):
        img = ski.io.imread(img)
    assert isinstance(img, np.ndarray)
    return scan_bookshelf.extract_books_from_bookshelf_image(img, output_dir)
