"""Command line interface."""

import shutil
from pathlib import Path
from typing import Annotated, Optional

import more_itertools
import skimage as ski
from loguru import logger
from rich import print as rprint
from typer import Argument, Option, Typer

from . import scan
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
) -> None:
    """Run OCR on an isolated book."""
    if output_dir is not None and not output_dir.exists():
        output_dir.mkdir()

    fpaths = more_itertools.flatten([Path().glob(b) for b in book])
    for fpath in fpaths:
        # fpath = Path(fpath_str)
        logger.info(f"Scanning image '{fpath.name}'")
        name = fpath.name.removesuffix(fpath.suffix)
        image = ski.io.imread(fpath)
        processing_out_path = None
        if output_dir:
            processing_out_path = output_dir / f"{name}_processed.jpeg"
        result = read_book_data(
            image,
            processing_out_path,
            min_conf=-1,
            key=fpath.name.removesuffix(fpath.suffix),
        )
        if len(result.ocr_results):
            for ocr_result in result.ocr_results:
                print(f"text: '{ocr_result.formatted_text}'")
        else:
            logger.warning("No text found.")
