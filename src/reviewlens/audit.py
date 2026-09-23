"""Reverse-check audit: work backwards from the written outputs to the input and prove they agree.

Two outputs:
- a stage trace per review (which stage passed, failed, skipped, or never ran, and why)
- a list of reconciliation checks that must all pass for the run to be trusted

The audit never writes raw review text or a detected PII value. It reports review ids only.
"""
from __future__ import annotations

import csv
import io
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .models import ReviewResult

STAGES = ("input", "language", "pii", "sentiment", "key_phrases", "entities")
MIXED_SIGNAL = "overall positive but has negative aspects"


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class StageState:
    state: str  # pass | fail | skip | not_run
    detail: str = ""


# --------------------------------------------------------------- stage trace
def trace(r: ReviewResult) -> dict[str, StageState]:
    errors = {e.stage: e.message for e in r.errors}
    out: dict[str, StageState] = {}

    out["input"] = (StageState("fail", errors["input"]) if "input" in errors
                    else StageState("pass", "; ".join(r.warnings) if "text truncated" in " ".join(r.warnings) else ""))

    if "language" in errors:
        out["language"] = StageState("fail", errors["language"])
    elif r.skipped_reason:
        out["language"] = StageState("skip", r.skipped_reason)
    elif r.language:
        out["language"] = StageState("pass", r.language)
    else:
        out["language"] = StageState("not_run", "blocked upstream")

    pii_ok = r.redacted_text is not None
    if "pii" in errors:
        out["pii"] = StageState("fail", errors["pii"])
    elif pii_ok:
        out["pii"] = StageState("pass", ", ".join(r.pii_categories) or "none found")
    else:
        out["pii"] = StageState("not_run", "blocked upstream")

    detail = {
        "sentiment": r.sentiment or "",
        "key_phrases": f"{len(r.key_phrases)} phrases",
        "entities": f"{len(r.entities)} entities",
    }
    for stage in ("sentiment", "key_phrases", "entities"):
        if stage in errors:
            out[stage] = StageState("fail", errors[stage])
        elif pii_ok:
            out[stage] = StageState("pass", detail[stage])
        else:
            out[stage] = StageState("not_run", "blocked by PII gate" if r.language else "blocked upstream")
    return out


# ------------------------------------------------------------ reverse checks
def _masked_values(raw: str, redacted: str) -> list[str] | None:
    """Recover what the service masked by diffing raw vs redacted text (masking is char-for-char).
    Returns None if the texts can't be aligned."""
    if len(raw) != len(redacted):
        return None
    values, start = [], None
    for i, (a, b) in enumerate(zip(raw, redacted)):
        masked = b == "*" and a != "*"
        if masked and start is None:
            start = i
        elif not masked and start is not None:
            values.append(raw[start:i]); start = None
    if start is not None:
        values.append(raw[start:])
    return [v.strip() for v in values if len(v.strip()) >= 3]


def run_checks(
    inputs: Sequence[tuple[str, str]],
    results: Sequence[ReviewResult],
    summary: dict,
    written_outputs: dict[str, str],
) -> list[Check]:
    checks: list[Check] = []
    by_id = {r.review_id: r for r in results}
    input_ids = [str(i).strip() for i, _ in inputs]

    # 1. Every input row is accounted for exactly once, and nothing extra appeared.
    missing = sorted(set(input_ids) - set(by_id))
    extra = sorted(set(by_id) - set(input_ids))
    checks.append(Check("every input row has exactly one result",
                        not missing and not extra and len(results) == len(inputs),
                        f"inputs={len(inputs)} results={len(results)} missing={missing} extra={extra}"))

    # 2. Summary counts reconcile with the per-review rows.
    recount = dict(Counter(r.status for r in results))
    checks.append(Check("summary status counts match per-review rows",
                        summary.get("total") == len(results) and summary.get("status") == recount,
                        f"summary={summary.get('status')} recomputed={recount}"))

    # 3. Every complaint in the summary traces back to real reviews with that negative aspect.
    bad = []
    for c in summary.get("top_complaints", []):
        ids = [r.review_id for r in results
               if any(a.target.lower() == c["aspect"] and a.sentiment == "negative" for a in r.aspects)]
        if len(ids) != c["negative"]:
            bad.append(f"{c['aspect']}: summary={c['negative']} traced={len(ids)}")
    checks.append(Check("top complaints trace back to source reviews", not bad,
                        "; ".join(bad) or f"{len(summary.get('top_complaints', []))} complaints traced"))

    # 4. Mixed-signal flags match what the data says.
    expected = sorted(r.review_id for r in results
                      if r.sentiment == "positive" and any(a.sentiment == "negative" for a in r.aspects))
    got = sorted(summary.get("flagged_mixed_signal", []))
    checks.append(Check("mixed-signal flags match data", expected == got, f"expected={expected} got={got}"))

    # 5. Fail-closed: a review that did not pass PII has no downstream analysis at all.
    leaks = [r.review_id for r in results if r.redacted_text is None
             and (r.sentiment or r.key_phrases or r.entities or r.aspects)]
    checks.append(Check("PII gate is fail-closed", not leaks, f"violations={leaks}"))

    # 6. No masked PII value appears anywhere in the written outputs.
    raw_by_id = {str(i).strip(): (t or "").strip() for i, t in inputs}
    leaked, unverifiable, n_values = [], [], 0
    for r in results:
        if not r.redacted_text:
            continue
        values = _masked_values(raw_by_id.get(r.review_id, ""), r.redacted_text)
        if values is None:
            unverifiable.append(r.review_id); continue
        n_values += len(values)
        for v in values:
            for fname, text in written_outputs.items():
                if v in text:
                    leaked.append(f"{r.review_id} in {fname}")  # never print the value itself
    checks.append(Check("no redacted PII value appears in any output file", not leaked,
                        f"checked {n_values} masked values; leaks={leaked}; unverifiable={unverifiable}"))

    # 7. No raw review text is written for reviews that contained PII.
    raw_found = [f"{r.review_id} in {fname}" for r in results if r.pii_categories
                 for fname, text in written_outputs.items()
                 if raw_by_id.get(r.review_id) and raw_by_id[r.review_id] in text]
    checks.append(Check("raw text of PII reviews is never written", not raw_found, f"found={raw_found}"))
    return checks


# ------------------------------------------------------------------ writers
_ICON = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP", "not_run": "--"}


def to_csv(results: Sequence[ReviewResult]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["review_id", "status", *STAGES, "pii_categories", "sentiment_label", "warnings"])
    for r in results:
        t = trace(r)
        w.writerow([r.review_id, r.status, *(t[s].state for s in STAGES),
                    "|".join(r.pii_categories), r.sentiment or "", "|".join(r.warnings)])
    return buf.getvalue()


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown(results: Sequence[ReviewResult], checks: Sequence[Check]) -> str:
    ok = all(c.passed for c in checks)
    lines = ["# ReviewLens audit", "",
             f"**Verdict: {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED - do not trust this run'}**", "",
             "## Reverse checks", "", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {_cell(c.detail)} |" for c in checks]
    lines += ["", "## Stage trace", "",
              "| review | status | " + " | ".join(STAGES) + " |",
              "|---|---|" + "---|" * len(STAGES)]
    traces = {r.review_id: trace(r) for r in results}
    for r in results:
        lines.append(f"| {r.review_id} | {r.status} | "
                     + " | ".join(_ICON[traces[r.review_id][s].state] for s in STAGES) + " |")
    lines += ["", "## Per-review detail", ""]
    for r in results:
        t = traces[r.review_id]
        lines.append(f"### {r.review_id} - {r.status}")
        for s in STAGES:
            d = f" - {t[s].detail}" if t[s].detail else ""
            lines.append(f"- **{s}**: {t[s].state}{d}")
        if r.redacted_text:
            lines.append(f"- redacted text: `{r.redacted_text}`")
        if r.aspects:
            lines.append("- aspects: " + ", ".join(f"{a.target} -> {a.sentiment}" for a in r.aspects))
        if r.warnings:
            lines.append("- warnings: " + "; ".join(r.warnings))
        lines.append("")
    return "\n".join(lines)
