"""Command line interface."""

from pathlib import Path
from typing import Annotated

from typer import Argument, Typer

from . import scan

app = Typer()


@app.command()
def main(
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
    """Identify books."""
    if not output_dir.exists():
        output_dir.mkdir()
    results = scan(image_file, output_dir=output_dir)
    print(results)
