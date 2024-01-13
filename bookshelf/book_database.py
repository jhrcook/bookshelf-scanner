"""Book database."""

from pathlib import Path

import polars as pl

from .models import KnownBook


class BookDatabase:

    """Book database."""

    def __init__(self, db_file: Path) -> None:
        """Create a book database object."""
        self.db_file = db_file
        self._known_books: list[KnownBook] | None = None

    @property
    def known_books(self) -> list[KnownBook]:
        """Iterate through known books."""
        if self._known_books is None:
            db = pl.read_csv(self.db_file)
            _known_books = []
            for title, authors_str, isbn in zip(
                db["Title"], db["Authors"], db["ISBN/UID"], strict=False
            ):
                authors = [a.strip() for a in authors_str.split(",")]
                _known_books.append(KnownBook(title=title, authors=authors, uuid=isbn))
            self._known_books = _known_books
        return self._known_books.copy()
