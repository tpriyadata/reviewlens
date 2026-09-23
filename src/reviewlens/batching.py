"""Request-size limits for the synchronous Language APIs."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")

# Max documents per synchronous request differs by operation (Azure AI Language data limits).
# PII and NER accept only 5; sending 10 fails the whole batch with HTTP 400.
STAGE_BATCH_LIMITS = {
    "language": 1000,
    "sentiment": 10,
    "key_phrases": 10,
    "pii": 5,
    "entities": 5,
}
MAX_DOCS_PER_REQUEST = 10  # upper bound for the user-configurable batch_size
# Per-document limit is 5,120 text elements. Python len() counts code points, which is
# always >= grapheme count, so truncating on len() is conservative (never over the limit).
MAX_CHARS_PER_DOC = 5120


def chunked(items: Sequence[T], size: int = MAX_DOCS_PER_REQUEST) -> Iterator[list[T]]:
    if size < 1:
        raise ValueError("size must be >= 1")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def truncate(text: str, limit: int = MAX_CHARS_PER_DOC) -> tuple[str, bool]:
    """Return (text, was_truncated). Cuts on the last whitespace before the limit when possible."""
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.8:
        cut = cut[:space]
    return cut, True
