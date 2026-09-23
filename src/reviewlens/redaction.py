"""Local, auditable redaction from the service's PII entity offsets."""
from __future__ import annotations

from collections.abc import Iterable

# Categories the service detects that are not personal data for this use case.
# PersonType = roles and job titles ("customer support", "courier", "manager").
DEFAULT_IGNORED_PII = frozenset({"PersonType"})


def mask_spans(text: str, spans: Iterable[tuple[int | None, int | None]]) -> str:
    """Mask each (offset, length) span with '*', char for char, like the service does.
    Raises ValueError on any span that does not fit the text, so the caller can fail closed."""
    chars = list(text)
    for offset, length in spans:
        if offset is None or length is None or offset < 0 or length <= 0 or offset + length > len(text):
            raise ValueError(f"PII span out of range: offset={offset} length={length} text_len={len(text)}")
        chars[offset:offset + length] = "*" * length
    return "".join(chars)
