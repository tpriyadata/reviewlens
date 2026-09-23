# ReviewLens

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
