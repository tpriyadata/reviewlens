"""Aggregate per-review results into a summary."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from .models import ReviewResult


def summarize(results: Sequence[ReviewResult], top_n: int = 10) -> dict:
    status = Counter(r.status for r in results)
    analyzed = [r for r in results if r.sentiment]
    sentiment = Counter(r.sentiment for r in analyzed)

    aspect_counts: dict[str, Counter] = defaultdict(Counter)
    for r in analyzed:
        for a in r.aspects:
            aspect_counts[a.target.lower()][a.sentiment] += 1

    complaints = sorted(
        ((t, c["negative"], sum(c.values())) for t, c in aspect_counts.items() if c["negative"]),
        key=lambda x: (-x[1], x[0]),
    )[:top_n]
    phrases = Counter(p.lower() for r in analyzed for p in r.key_phrases).most_common(top_n)

    return {
        "total": len(results),
        "status": dict(status),
        "sentiment": dict(sentiment),
        "top_complaints": [{"aspect": t, "negative": n, "mentions": m} for t, n, m in complaints],
        "top_key_phrases": [{"phrase": p, "count": c} for p, c in phrases],
        "reviews_with_pii": sum(1 for r in results if r.pii_categories),
        "flagged_mixed_signal": [r.review_id for r in results
                                 if "overall positive but has negative aspects" in r.warnings],
    }
