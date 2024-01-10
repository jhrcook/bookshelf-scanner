"""Data models."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Point:

    """Point in an image."""

    row: float
    col: float

    @property
    def xy(self) -> tuple[float, float]:
        """Corresponding coordinates for (x,y) axes."""
        return (self.col, self.row)


@dataclass
class Line:

    """Line in an image."""

    point: Point
    slope: float

    @property
    def row_intercept(self) -> float:
        """Row intercept of the line."""
        return self.slope * (-self.point.col) + self.point.row

    def is_above(self, point: tuple[float, float]) -> bool:
        """Is the line above a given point."""
        y = self.slope * point[1] + self.row_intercept
        return point[0] > y

    def is_right_of(self, point: Point) -> bool:
        """Is the line to the right of a given point."""
        col = self.col(at_row=point.row)
        return point.col < col

    def row(self, at_col: float) -> float:
        """Row of a line at a given column."""
        return self.slope * at_col + self.row_intercept

    def col(self, at_row: float) -> float:
        """Column of a line at a given row."""
        return self.point.col - ((self.point.row - at_row) / self.slope)

    def __hash__(self) -> int:
        """Generate unique hash."""
        return hash(f"point: {self.point}  --  slope: {self.slope}")


@dataclass
class Shelf:

    """Isolated bookshelf."""

    key: str
    top: Line
    bottom: Line
    image: np.ndarray


@dataclass
class Book:

    """Isolated book."""

    key: str
    left: Line
    right: Line
    image: np.ndarray


@dataclass
class BookData:

    """Extracted book information."""

    key: str
