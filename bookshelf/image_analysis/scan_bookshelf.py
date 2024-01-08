"""Identify books on a bookshelf."""

from pathlib import Path

import numpy as np
from more_itertools import flatten

from . import find_shelves
from .find_books import find_books
from .models import Book, BookData


def read_book_data(book: Book, output_dir: Path | None = None) -> BookData:
    """Read book data from a book isolated from a shelf.

    # TODO

    Args:
    ----
        book (Book): Isolated book.
        output_dir (Path | None, optional): Output directory where intermediate results can be saved. Defaults to None.

    Returns:
    -------
        BookData: Extracted book data.
    """
    return BookData(key="key")


def extract_books_from_bookshelf_image(
    img: np.ndarray, output_dir: Path | None = None
) -> list[BookData]:
    """Extract book data from a picture of a bookshelf.

    Args:
    ----
        img (np.ndarray): Image of a bookshelf.
        output_dir (Path | None, optional): Output directory where intermediate results can be saved. Defaults to None.

    Returns:
    -------
        list[BookData]: Isolated book data.
    """
    shelves = find_shelves.find_shelves(img, output_dir)
    _books = flatten([find_books(s, output_dir) for s in shelves])
    # book_info = [read_book_data(b, output_dir) for b in books]
    return []
