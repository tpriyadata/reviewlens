"""ReviewLens final milestone: PersonType-aware redaction, regression tests, README, CI. Run from the repo root."""
EDITS = [
 ("src/reviewlens/pipeline.py",
  '''- The three analysis stages run independently on the redacted text, so a sentiment
  failure does not cost you key phrases or entities.
"""''',
  '''- The three analysis stages run independently on the redacted text, so a sentiment
  failure does not cost you key phrases or entities.
- Redaction is done locally from the service's entity offsets, so we control which
  categories count as personal data (PersonType - roles like "customer support" - does not).
  Every character we mask must also be masked by the service; any disagreement fails closed.
"""'''),
 ("src/reviewlens/pipeline.py",
  "from .models import Assessment, AspectOpinion, Entity, ReviewResult\n",
  "from .models import Assessment, AspectOpinion, Entity, ReviewResult\nfrom .redaction import DEFAULT_IGNORED_PII, mask_spans\n"),
 ("src/reviewlens/pipeline.py",
  '''        min_entity_confidence: float = 0.6,
    ) -> None:''',
  '''        min_entity_confidence: float = 0.6,
        pii_ignore_categories: Iterable[str] = DEFAULT_IGNORED_PII,
    ) -> None:'''),
 ("src/reviewlens/pipeline.py",
  '''        self.min_entity_confidence = min_entity_confidence
''',
  '''        self.min_entity_confidence = min_entity_confidence
        self.pii_ignore_categories = frozenset(pii_ignore_categories)
'''),
 ("src/reviewlens/pipeline.py",
  '''    @staticmethod
    def _make_apply_pii(texts: dict[str, str]) -> StageApply:
        def apply(doc, result: ReviewResult) -> None:
            if doc.redacted_text is None:
                raise ValueError("no redacted_text returned")
            result.redacted_text = doc.redacted_text
            result.pii_categories = sorted({str(e.category) for e in doc.entities})
            texts[doc.id] = doc.redacted_text  # downstream stages only ever see redacted text
        return apply
''',
  '''    def _make_apply_pii(self, texts: dict[str, str]) -> StageApply:
        def apply(doc, result: ReviewResult) -> None:
            raw = texts[doc.id]
            service = doc.redacted_text
            if service is None or len(service) != len(raw):
                raise ValueError("service redaction missing or not aligned with input")
            kept = [e for e in doc.entities if str(e.category) not in self.pii_ignore_categories]
            redacted = mask_spans(raw, [(e.offset, e.length) for e in kept])
            # Cross-check: anything we mask, the service must have masked too.
            if any(r == "*" and raw[i] != "*" and service[i] != "*" for i, r in enumerate(redacted)):
                raise ValueError("local redaction disagrees with service redaction")
            result.redacted_text = redacted
            result.pii_categories = sorted({str(e.category) for e in kept})
            texts[doc.id] = redacted  # downstream stages only ever see redacted text
        return apply
'''),
 ("tests/fakes.py",
  '''EMAIL = re.compile(r"\\S+@\\S+")
''',
  '''EMAIL = re.compile(r"\\S+@\\S+")
ROLE = re.compile(r"Customer support")
'''),
 ("tests/fakes.py",
  '''    def __init__(self, fail_ids: dict[str, set[str]] | None = None, raise_on: dict | None = None):
        self.calls: list[tuple[str, list[dict]]] = []
''',
  '''    def __init__(self, fail_ids: dict[str, set[str]] | None = None, raise_on: dict | None = None,
                 pii_mode: str = "normal"):
        self.pii_mode = pii_mode         # normal | bad_offset | service_unmasked
        self.calls: list[tuple[str, list[dict]]] = []
'''),
 ("tests/fakes.py",
  '''            ents = [PiiEntity(text=m.group(), category="Email", confidence_score=0.9)
                    for m in EMAIL.finditer(d["text"])]
            redacted = EMAIL.sub(lambda m: "*" * len(m.group()), d["text"])
''',
  '''            text = d["text"]
            ents = [PiiEntity(text=m.group(), category=cat, confidence_score=0.9,
                              offset=m.start(), length=len(m.group()))
                    for rx, cat in ((EMAIL, "Email"), (ROLE, "PersonType")) for m in rx.finditer(text)]
            redacted = text
            for e in ents:  # like the real service: masks every detected entity, char for char
                redacted = redacted[:e.offset] + "*" * e.length + redacted[e.offset + e.length:]
            if self.pii_mode == "bad_offset" and ents:
                ents[0].offset = len(text) + 5
            if self.pii_mode == "service_unmasked":
                redacted = text
'''),
]

NEW_FILES = {}

NEW_FILES["src/reviewlens/redaction.py"] = r'''"""Local, auditable redaction from the service's PII entity offsets."""
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
'''

NEW_FILES["tests/test_redaction.py"] = r'''import pytest

from reviewlens.pipeline import ReviewPipeline
from reviewlens.redaction import mask_spans

from .fakes import FakeClient

REVIEW = "Customer support never replied. Email jo@x.com please."


def test_mask_spans_masks_char_for_char():
    assert mask_spans("call 555-1234 now", [(5, 8)]) == "call ******** now"


@pytest.mark.parametrize("span", [(-1, 3), (10, 50), (0, 0), (None, 3)])
def test_mask_spans_rejects_bad_spans(span):
    with pytest.raises(ValueError):
        mask_spans("short text", [span])


def test_person_type_is_kept_but_email_is_masked():
    client = FakeClient()
    res = ReviewPipeline(client).run([("a", REVIEW)])[0]
    assert res.status == "ok"
    assert res.redacted_text.startswith("Customer support never replied.")
    assert "jo@x.com" not in res.redacted_text
    assert res.pii_categories == ["Email"]
    sent = [d["text"] for s, docs in client.calls if s == "sentiment" for d in docs]
    assert "Customer support" in sent[0]  # the aspect survives to sentiment analysis


def test_person_type_can_be_redacted_if_configured():
    res = ReviewPipeline(FakeClient(), pii_ignore_categories=()).run([("a", REVIEW)])[0]
    assert "Customer support" not in res.redacted_text
    assert res.pii_categories == ["Email", "PersonType"]


def test_bad_offset_fails_closed():
    client = FakeClient(pii_mode="bad_offset")
    res = ReviewPipeline(client).run([("a", REVIEW)])[0]
    assert res.status == "failed" and res.sentiment is None
    assert not any(s == "sentiment" for s, _ in client.calls)


def test_disagreement_with_service_fails_closed():
    res = ReviewPipeline(FakeClient(pii_mode="service_unmasked")).run([("a", REVIEW)])[0]
    assert res.status == "failed"
    assert "disagrees" in res.errors[0].message
'''

NEW_FILES["README.md"] = r'''# ReviewLens

Customer-review analysis on **Azure AI Language** (`azure-ai-textanalytics`), built to be trusted rather than demoed:
PII is redacted before any analysis, every review gets a traceable status, and every run is reverse-checked against its own input.

```
CSV ──► input check ──► language ──► PII redaction (gate) ──┬─► sentiment + opinion mining
                                                            ├─► key phrases
                                                            └─► entities
                                   ──► results.jsonl + summary.json ──► audit (reverse checks)
```

## What it produces

| File | Contents |
|---|---|
| `output/results.jsonl` | One row per review: status, language, **redacted** text, PII categories, sentiment, aspects, key phrases, entities, warnings, per-stage errors |
| `output/summary.json` | Status counts, sentiment mix, top complaints (negative aspects), top key phrases, reviews with PII, mixed-signal flags |
| `output/audit.md` | Verdict, reverse checks, a stage-by-stage trace for every review |
| `output/audit.csv` | The same trace, spreadsheet-friendly |

The runner exits non-zero if any audit check fails.

## Design decisions

**PII redaction is a fail-closed gate.** A review whose redaction fails goes no further. Raw text is never sent to the analysis stages or written to disk.

**Redaction is done locally from entity offsets, and cross-checked against the service.** The service's own `redacted_text` masks every category it detects, including `PersonType` (roles like "customer support", "courier"). That silently deleted real complaints from the analysis. ReviewLens masks from the entity offsets itself, skipping configurable non-identifying categories (default: `PersonType`). Every character it masks must also be masked by the service; any disagreement, or an offset that doesn't fit the text, fails the review closed.

**Failures are per review, per stage.** One bad document never fails a batch; one failed batch never fails the run. Status is `ok`, `partial` (an analysis stage failed but others succeeded), `failed` (a gating stage failed), or `skipped` (language not in the allowed set). Authentication errors abort immediately, since every call would fail.

**Per-stage batch limits.** Synchronous requests accept different document counts per operation: PII and NER take 5, sentiment and key phrases take 10. Sending 10 to PII fails the *whole batch* with HTTP 400. Batch errors record status code, service error code and message.

**Retries** come from azure-core's `RetryPolicy` (429/5xx, honours `Retry-After`).

**Pydantic schemas** (`extra="forbid"`, `validate_assignment=True`) reject malformed service responses at the point they're mapped; a validation failure becomes a stage error, not a crash.

## Reverse-check audit

After writing outputs, the runner re-reads them and reconciles against the input:

1. Every input row has exactly one result (nothing lost, duplicated or invented)
2. Summary status counts match the per-review rows
3. Every top complaint traces back to source reviews with that negative aspect
4. Mixed-signal flags match the data
5. The PII gate is fail-closed (no analysis exists for unredacted reviews)
6. No masked PII value appears in any output file (values recovered by diffing raw vs redacted text, then searched for)
7. Raw text of PII-containing reviews is never written

The audit reports review ids only. It never writes a PII value, even when reporting a leak.

## Findings from real runs

- **Overall sentiment can hide complaints.** "The battery dies fast but the screen is gorgeous." was scored `positive` (0.99) and opinion mining missed the battery complaint entirely. ReviewLens flags `positive` reviews that contain negative aspects, but can't recover an aspect the service never extracted.
- **Negation lives on the assessment.** "Customer support never replied" returns the assessment word `replied`; the negation is only visible through `is_negated`, which ReviewLens keeps in the output.
- **Over-redaction damages analytics silently.** With service-side redaction, "Customer support" (`PersonType`) and "Courier" (`Person`) were masked, removing both complaints from the report. `PersonType` is now kept; see limitations for `Person`.
- **Batch limits differ by operation** (see above). The first real run failed every PII batch until limits were set per stage.

## Known limitations

- **`Person` false positives are accepted.** The model tagged "Courier" as a `Person`. Missing a real name is worse than masking a role word, so `Person` stays redacted.
- **English only by default** (`--languages en`). Other languages are marked `skipped`, not analysed.
- **Opinion-mining recall.** Aspects the service doesn't extract never reach the report; overall labels on mixed reviews are unreliable.
- **Audit check 6 is conservative.** It searches outputs for masked values as plain strings, so a common word masked in one review but present in another will be reported as a leak. Treat a failure as "investigate", not proof of a leak.
- **Free tier (F0)**: 5,000 text records/month; each stage call on each document counts.
- No labelled evaluation set yet; accuracy figures would need one.

## Setup (Windows PowerShell)

```powershell
git clone https://github.com/tpriyadata/reviewlens.git
cd reviewlens
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env   # then fill in LANGUAGE_ENDPOINT and LANGUAGE_KEY
```

Create an Azure **Language** resource (Free F0 tier is enough) and copy the endpoint and KEY 1 from **Keys and Endpoint**.

## Run

```powershell
python scripts\smoke_test.py          # confirms credentials
python scripts\run_pipeline.py        # data\sample_reviews.csv -> output\
python scripts\run_pipeline.py --input my_reviews.csv --languages en,es
```

Input CSV needs `review_id` and `text` columns. The sample data is synthetic; the contact details in it are fictional.

## Tests

```powershell
pytest -q
```

Tests use a fake client that returns real SDK model objects, so no Azure calls or keys are needed. They cover batching limits, the fail-closed PII gate, local-vs-service redaction disagreement, bad offsets, whole-batch HTTP errors, auth aborts, and each audit check (including planted leaks and a tampered summary). CI runs them on every push.

## Project layout

```
src/reviewlens/
  client.py      client factory, config validation
  batching.py    per-stage request limits, truncation
  redaction.py   offset-based masking
  pipeline.py    stages, gate, per-document error handling
  models.py      Pydantic output schemas
  report.py      summary aggregation
  audit.py       stage trace + reverse checks
scripts/         smoke_test.py, run_pipeline.py
tests/           fake client + tests
data/            synthetic sample reviews
```
'''

NEW_FILES[".github/workflows/tests.yml"] = r'''name: tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest -q   # uses the fake client; no Azure keys needed
'''

from pathlib import Path
for path, old, new in EDITS:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        print("already applied:", path); continue
    if text.count(old) != 1:
        raise SystemExit(f"could not find expected code in {path} - stopping")
    p.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    print("patched", path)
for path, content in NEW_FILES.items():
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8", newline="\n")
    print("wrote", p)