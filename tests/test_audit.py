from reviewlens.audit import _masked_values, run_checks, to_csv, to_markdown, trace
from reviewlens.pipeline import ReviewPipeline
from reviewlens.report import summarize

from .fakes import FakeClient

INPUTS = [
    ("a", "Battery dies. Mail me at jo@x.com"),
    ("b", ""),
    ("c", "El producto llegó roto"),
    ("d", "Great screen"),
]


def _run(client=None):
    results = ReviewPipeline(client or FakeClient()).run(INPUTS)
    summary = summarize(results)
    written = {"results.jsonl": "\n".join(r.model_dump_json() for r in results)}
    return results, summary, written


def test_masked_values_recovers_what_was_redacted():
    assert _masked_values("call 555-1234 now", "call ******** now") == ["555-1234"]
    assert _masked_values("abc", "abcd") is None


def test_all_checks_pass_on_clean_run():
    results, summary, written = _run()
    checks = run_checks(INPUTS, results, summary, written)
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]


def test_detects_pii_leak_in_output_without_printing_it():
    results, summary, written = _run()
    written["results.jsonl"] += "\nleaked jo@x.com"
    leak = next(c for c in run_checks(INPUTS, results, summary, written) if "PII value" in c.name)
    assert not leak.passed and "a in results.jsonl" in leak.detail
    assert "jo@x.com" not in leak.detail


def test_detects_tampered_summary():
    results, summary, written = _run()
    summary["status"]["ok"] += 1
    failed = [c.name for c in run_checks(INPUTS, results, summary, written) if not c.passed]
    assert failed == ["summary status counts match per-review rows"]


def test_detects_missing_result():
    results, summary, written = _run()
    failed = [c.name for c in run_checks(INPUTS + [("z", "x")], results, summary, written) if not c.passed]
    assert "every input row has exactly one result" in failed


def test_trace_shows_where_each_review_stopped():
    results, _, _ = _run(FakeClient(fail_ids={"pii": {"d"}}))
    t = {r.review_id: trace(r) for r in results}
    assert t["a"]["entities"].state == "pass"
    assert t["b"]["input"].state == "fail" and t["b"]["language"].state == "not_run"
    assert t["c"]["language"].state == "skip" and t["c"]["pii"].state == "not_run"
    assert t["d"]["pii"].state == "fail" and t["d"]["sentiment"].detail == "blocked by PII gate"


def test_reports_contain_no_raw_pii():
    results, summary, written = _run()
    md = to_markdown(results, run_checks(INPUTS, results, summary, written))
    assert "jo@x.com" not in md and "jo@x.com" not in to_csv(results)
    assert "ALL CHECKS PASSED" in md
