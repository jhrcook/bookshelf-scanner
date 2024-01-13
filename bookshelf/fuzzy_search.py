"""Use fuzzy matching to search a book database."""

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable

from more_itertools import flatten
from thefuzz import fuzz, process

from .book_database import BookDatabase
from .models import BookData, KnownBook


@dataclass
class FuzzyMatch:

    """Fuzzy search match."""

    input_book_key: str
    query: str
    known_book: KnownBook | None
    match: str
    score: int
    fuzz_scorer: str


Scorer = Callable[[str, str], int]


def _fuzzy_search(
    query: str,
    input_book_key: str,
    choices: dict[str, KnownBook] | Iterable[str],
    scorer: Scorer,
) -> list[FuzzyMatch]:
    _choices = choices.keys() if isinstance(choices, dict) else choices
    fuzzy_results: list[tuple[str, int]] = process.extract(
        query, _choices, scorer=scorer
    )
    matches: list[FuzzyMatch] = []
    for matching_pattern, score in fuzzy_results:
        known_book = choices[matching_pattern] if isinstance(choices, dict) else None
        fuzz_match = FuzzyMatch(
            input_book_key=input_book_key,
            query=query,
            known_book=known_book,
            match=matching_pattern,
            score=score,
            fuzz_scorer=scorer.__name__,
        )
        matches.append(fuzz_match)
    return matches


def fuzzy_search(
    book_data: BookData,
    book_db: BookDatabase,
    min_score: int = 0,
    min_query_length: int = 4,
) -> list[FuzzyMatch]:
    """Perform fuzzy searching of the OCR results in a book database.

    Args:
    ----
        book_data (BookData): Book data.
        book_db (BookDatabase): Book database.
        min_score (int, optional): Minimum fuzzy matching score. Defaults to 0.
        min_query_length (int, optional): Minimum query length. Defaults to 4.

    Returns:
    -------
        list[FuzzyMatch]: Ranked fuzzy search matches.
    """
    matches: list[FuzzyMatch] = []
    title_choices = {b.title: b for b in book_db.known_books}
    author_choices = set(flatten(b.authors for b in book_db.known_books))
    scorers = (
        fuzz.token_set_ratio,
        fuzz.token_sort_ratio,
    )
    for ocr_result, choices, scorer in product(
        book_data.ocr_results, (title_choices, author_choices), scorers
    ):
        query_str = ocr_result.formatted_text
        if len(query_str) < min_query_length:
            continue
        matches += _fuzzy_search(
            query_str, book_data.key, choices=choices, scorer=scorer
        )

    matches = [m for m in matches if m.score >= min_score]
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


def summarize_matches(matches: list[FuzzyMatch]) -> list[tuple[str, list[FuzzyMatch]]]:
    """Summarize matches by grouping those with the same matching string."""
    sum_matches_map: dict[str, list[FuzzyMatch]] = {}
    for f_match in matches:
        sum_match = sum_matches_map.get(f_match.match, [])
        sum_match.append(f_match)
        sum_matches_map[f_match.match] = sum_match

    sum_matches: list[tuple[str, list[FuzzyMatch]]] = list(sum_matches_map.items())
    sum_matches.sort(key=lambda a: max(m.score for m in a[1]), reverse=True)
    return sum_matches
