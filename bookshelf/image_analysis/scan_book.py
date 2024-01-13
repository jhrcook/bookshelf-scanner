"""Extract text from a book."""

from pathlib import Path

import numpy as np
import polars as pl
import pytesseract
import skimage as ski
from loguru import logger
from matplotlib import pyplot as plt

from ..models import Book, BookData, OCRResult
from ..plotting import set_axes_off
from . import utilities


def _plot_processing(
    processing_intermediates: list[dict[str, np.ndarray]], out_file: Path
) -> None:
    img_shape = processing_intermediates[0]["original"].shape
    nrows, ncols = (
        max(len(d) for d in processing_intermediates),
        len(processing_intermediates),
    )
    fig_w = 5.0 * len(processing_intermediates)
    fig_h = nrows * (0.5 + (fig_w / img_shape[0] * img_shape[1]))

    fig, grid_axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False
    )

    for axes, images in zip(grid_axes.T, processing_intermediates, strict=False):
        plot_images = {n: np.rot90(i.copy()) for n, i in images.items()}
        for ax, (name, img) in zip(axes.flatten(), plot_images.items(), strict=False):
            ax.set_title(name)
            ax.imshow(img)

    set_axes_off(grid_axes)
    fig.tight_layout()
    fig.savefig(out_file, dpi=400)
    plt.close()


def _preprocessing_pipeline_1(
    image: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    gray = ski.color.rgb2gray(image)
    exp_eq = ski.exposure.equalize_adapthist(gray)
    gauss = ski.filters.gaussian(exp_eq)
    laplace = ski.filters.laplace(gauss)
    otsu = utilities.filter_otsu(laplace)
    trimmed = utilities.drop_zero_rows_cols(otsu)
    ocr_input = ski.util.img_as_ubyte(otsu)
    intermediates = {
        "original": image,
        "gray": gray,
        "eq. adapt. hist.": exp_eq,
        "Gaussian": gauss,
        "Laplace": laplace,
        "Otsu": otsu,
        "trimmed": trimmed,
        "OCR input": ocr_input,
    }
    return ocr_input, intermediates


def _preprocessing_pipeline_2(
    image: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    gray = ski.color.rgb2gray(image)
    exp_log = ski.exposure.adjust_log(gray)
    exp_eq = ski.exposure.equalize_adapthist(exp_log)
    gauss = ski.filters.gaussian(exp_eq)
    meijering = ski.filters.meijering(gauss, sigmas=(1,), black_ridges=True)
    otsu = utilities.filter_otsu(meijering).astype(float)
    flipped = utilities.flip_background(otsu)
    trimmed = utilities.drop_zero_rows_cols(flipped)
    ocr_input = ski.util.img_as_ubyte(flipped)
    intermediates = {
        "original": image,
        "gray": gray,
        "exp. log": exp_log,
        "eq. adapt. hist.": exp_eq,
        "Gaussian": gauss,
        "meijering": meijering,
        "Otsu": otsu,
        "flipped": flipped,
        "trimmed": trimmed,
        "OCR input": ocr_input,
    }
    return ocr_input, intermediates


def read_book_data(
    book: Book | np.ndarray,
    processing_out_path: Path | None = None,
    min_conf: int = 10,
    key: str = "",
) -> BookData:
    """Read book data from a book isolated from a shelf.

    Args:
    ----
        book (Book | np.ndarray): Isolated book.
        processing_out_path (Path | None, optional): File path for where to write the processing pipeline images. Defaults to `None`.
        min_conf (int, optional): Minimum confidence threshold for detected text. The range is [-1, 100]. Defaults to 10.
        key (str): Identifying key to use if an image is passed.

    Returns:
    -------
        BookData: Extracted book data.
    """
    image = book.image.copy() if isinstance(book, Book) else book.copy()
    ocr_results: list[OCRResult] = []
    pipeline_res: list[dict[str, np.ndarray]] = []
    for i, preprocessing_pipeline in enumerate(
        (
            _preprocessing_pipeline_1,
            _preprocessing_pipeline_2,
        )
    ):
        logger.debug(f"Preprocessing pipeline #{i}.")
        ocr_input, _pipeline_res = preprocessing_pipeline(image)
        pipeline_res.append(_pipeline_res)
        for k in range(4):
            _rot_img = np.rot90(ocr_input, k=k)
            logger.trace(f"Running OCR at angle {90*k}")
            res = (
                pl.DataFrame(
                    pytesseract.image_to_data(
                        image=_rot_img,
                        lang="eng",
                        output_type=pytesseract.Output.DATAFRAME,
                    )
                )
                .cast({"text": pl.Utf8})
                .filter(~pl.col("text").is_null())
                .with_columns(pl.col("text").str.strip())
                .filter(
                    (pl.col("conf") >= min_conf) & (pl.col("text").str.lengths() > 0)
                )
            )
            if res.shape[0] > 0:
                ocr_results.append(OCRResult(data_frame=res))
            else:
                logger.trace("No text found.")
    if processing_out_path:
        _plot_processing(pipeline_res, processing_out_path)
    _key = book.key if isinstance(book, Book) else key
    return BookData(key=_key, ocr_results=ocr_results)
