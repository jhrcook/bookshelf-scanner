"""Command line interface."""

from typer import Typer

app = Typer()


@app.command()
def main() -> None:
    """Identify books."""
    ...
