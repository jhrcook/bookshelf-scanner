"""Command line interface."""

import shutil
from pathlib import Path
from typing import Annotated, Optional

import more_itertools
import skimage as ski
from loguru import logger
from rich import print as rprint
from typer import Argument, Option, Typer

from . import fuzzy_search, scan
from .book_database import BookDatabase
from .image_analysis.scan_book import read_book_data
from .image_analysis.scan_bookshelf import isolate_books_from_bookshelf_image

app = Typer()


@app.command(name="scan")
def scan_books(
    image_file: Annotated[
        Path, Argument(help="Image file.", dir_okay=False, file_okay=True, exists=True)
    ],
    output_dir: Annotated[
        Optional[Path],
        Option(
            help="Directory to which output is saved.", dir_okay=True, file_okay=False
        ),
    ] = None,
) -> None:
    """Identify books."""
    if output_dir is not None:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir()
    results = scan(image_file, output_dir=output_dir)
    for book_data in results:
        if len(book_data.ocr_results) == 0:
            rprint("No text identifed for [blue]{book_data.key}[/blue].")
            continue
        rprint(f"Results for [blue]{book_data.key}[/blue]:")
        for ocr in book_data.ocr_results:
            rprint(f" {ocr.formatted_text}")


@app.command()
def isolate_books(
    image_file: Annotated[
        Path, Argument(help="Image file.", dir_okay=False, file_okay=True, exists=True)
    ],
    output_dir: Annotated[
        Path,
        Argument(
            help="Directory to which output is saved.", dir_okay=True, file_okay=False
        ),
    ],
) -> None:
    """Isolate books as separate image files."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    image = ski.io.imread(image_file)
    books = isolate_books_from_bookshelf_image(image, output_dir=output_dir)
    for i, book in enumerate(books):
        ski.io.imsave(
            output_dir / f"book-{i:03d}.jpeg", ski.util.img_as_ubyte(book.image)
        )


@app.command()
def ocr(
    book: Annotated[
        list[str],
        Argument(
            help="Image of an isolated book.",
            file_okay=True,
            dir_okay=False,
            exists=True,
        ),
    ],
    output_dir: Annotated[
        Optional[Path],
        Option(
            help="Directory to which output is saved.", dir_okay=True, file_okay=False
        ),
    ] = None,
    book_db: Annotated[
        Optional[Path],
        Option(
            help="Search the results against titles and authors in a database of books.",
            file_okay=True,
            dir_okay=False,
            exists=True,
        ),
    ] = None,
    min_fuzz_matching_score: int = 50,
    top_fuzz_matches: int = 10,
) -> None:
    """Run OCR on an isolated book."""
    if output_dir is not None and not output_dir.exists():
        output_dir.mkdir()

    fpaths = more_itertools.flatten([Path().glob(b) for b in book])
    book_database = BookDatabase(book_db) if book_db is not None else None
    for fpath in fpaths:
        logger.info(f"Scanning image '{fpath.name}'")
        name = fpath.name.removesuffix(fpath.suffix)
        image = ski.io.imread(fpath)
        processing_out_path = None
        if output_dir:
            processing_out_path = output_dir / f"{name}_processed.jpeg"
        result = read_book_data(
            image,
            processing_out_path,
            min_conf=1,
            key=fpath.name.removesuffix(fpath.suffix),
        )
        if len(result.ocr_results) == 0:
            logger.warning("No text found.")
            continue

        for ocr_result in result.ocr_results:
            logger.info(f"text: '{ocr_result.formatted_text}'")

        if book_database is not None:
            logger.info("Search book database...")
            db_matches = fuzzy_search.fuzzy_search(
                result, book_database, min_score=min_fuzz_matching_score
            )
            merged_db_matches = fuzzy_search.summarize_matches(db_matches)
            logger.info("Done.")
            if len(db_matches) == 0:
                logger.warning("No matches in book database. ")
                continue

            logger.info(f"Found {len(merged_db_matches)} matches in book database.")
            _n = min(len(merged_db_matches), top_fuzz_matches)
            for db_match, hits in merged_db_matches[:_n]:
                top_score = max(h.score for h in hits)
                logger.info(f"{db_match} (top score: {top_score})")
