"""Identify books on a bookshelf."""

from pathlib import Path

import numpy as np
from loguru import logger
from more_itertools import flatten

from bookshelf.image_analysis.scan_book import read_book_data

from . import find_shelves
from .find_books import find_books
from .models import Book, BookData


def isolate_books_from_bookshelf_image(
    img: np.ndarray, output_dir: Path | None = None
) -> list[Book]:
    """Isolate images of books from a picture of a bookshelf.

    Args:
    ----
        img (np.ndarray): Image of a bookshelf.
        output_dir (Path | None, optional): Output directory where intermediate results can be saved. Defaults to None.

    Returns:
    -------
        list[Book]: Isolated book images.
    """
    logger.info("Finding shelves.")
    shelves = find_shelves.find_shelves(img, output_dir)
    logger.info("Finding books in shelves.")
    return list(flatten([find_books(s, output_dir) for s in shelves]))


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
        list[BookData]: Book data.
    """
    books = isolate_books_from_bookshelf_image(img, output_dir)
    logger.info("Running OCR on books.")
    books_info = []
    for book in books:
        img_outfile = (
            None
            if output_dir is None
            else output_dir / f"{book.key}_ocr-processed.jpeg"
        )
        books_info.append(read_book_data(book, img_outfile))
    return books_info
