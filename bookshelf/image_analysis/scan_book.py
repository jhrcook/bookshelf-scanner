"""Extract text from a book."""

from pathlib import Path

import numpy as np
import polars as pl
import pytesseract
import skimage as ski
from loguru import logger
from matplotlib import pyplot as plt

from . import utilities
from .models import Book, BookData, OCRResult


def _plot_processing(images: dict[str, np.ndarray], out_file: Path) -> None:
    img_shape = images["original"].shape
    nrows, ncols = len(images), 1
    fig_w = 5.0
    fig_h = len(images) * (0.5 + (fig_w / img_shape[0] * img_shape[1]))
    if img_shape[1] > img_shape[0]:
        nrows, ncols = ncols, nrows
        fig_w, fig_h = fig_h, fig_w
        plot_images = {n: i.copy() for n, i in images.items()}
    else:
        plot_images = {n: np.rot90(i.copy()) for n, i in images.items()}

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False
    )
    axes = axes.flatten()
    for ax, (name, img) in zip(axes.flatten(), plot_images.items(), strict=False):
        ax.set_title(name)
        ax.set_axis_off()
        ax.imshow(img)
    fig.tight_layout()
    fig.savefig(out_file, dpi=400)


def read_book_data(
    book: Book | np.ndarray, processing_out_path: Path | None = None, min_conf: int = 10
) -> BookData:
    """Read book data from a book isolated from a shelf.

    Args:
    ----
        book (Book | np.ndarray): Isolated book.
        processing_out_path (Path | None, optional): File path for where to write the processing pipeline images. Defaults to `None`.
        min_conf (int, optional): Minimum confidence threshold for detected text. The range is [-1, 100]. Defaults to 10.

    Returns:
    -------
        BookData: Extracted book data.
    """
    image: np.ndarray
    if isinstance(book, Book):
        image = book.image.copy()
    else:
        image = book.copy()

    gray = ski.color.rgb2gray(image)
    # exp_gamma = ski.exposure.adjust_gamma(gray)
    exp_log = ski.exposure.adjust_log(gray)
    exp_eq = ski.exposure.equalize_adapthist(exp_log)
    gauss = ski.filters.gaussian(exp_eq)
    laplace = ski.filters.laplace(gauss)  # WORKS REALLY WELL!
    # meijering = ski.filters.meijering(gauss, sigmas=(0.1, 0.5, 1), black_ridges=True)
    # sobel = ski.filters.sobel(gauss)
    # skeleton = ski.morphology.skeletonize(gauss)
    # canny = ski.feature.canny(sobel, sigma=0.5)
    otsu = utilities.filter_otsu(laplace)
    # flipped = utilities.flip_background(otsu)
    # closed = ski.morphology.area_closing(otsu, area_threshold=10)
    # border = ski.segmentation.clear_border(otsu)
    ocr_input = ski.util.img_as_ubyte(otsu)

    if processing_out_path:
        _plot_processing(
            {
                "original": image,
                "gray": gray,
                # "exp. Gamma": exp_gamma,
                "exp. log": exp_log,
                "eq. adapt. hist.": exp_eq,
                "Gaussian": gauss,
                # "Sobel": sobel,
                "Laplace": laplace,
                # "meijering": meijering,
                # "Canny": canny,
                "Otsu": otsu,
                # "flipped": flipped,
                # "closed": closed,
                # "border": border,
                "OCR input": ocr_input,
            },
            processing_out_path,
        )

    ocr_results: list[OCRResult] = []
    for k in range(4):
        _rot_img = np.rot90(ocr_input, k=k)
        logger.trace(f"Running OCR at angle {90*k}")
        res = (
            pl.DataFrame(
                pytesseract.image_to_data(
                    image=_rot_img, lang="eng", output_type=pytesseract.Output.DATAFRAME
                )
            )
            .cast({"text": pl.Utf8})
            .filter(~pl.col("text").is_null())
            .with_columns(pl.col("text").str.strip())
            .filter((pl.col("conf") >= min_conf) & (pl.col("text").str.lengths() > 0))
        )
        if res.shape[0] > 0:
            ocr_results.append(OCRResult(data_frame=res))
        else:
            logger.trace("No text found.")
    return BookData(key="key", ocr_results=ocr_results)
