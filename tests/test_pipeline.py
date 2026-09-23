import pytest
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

from reviewlens.pipeline import ReviewPipeline
from reviewlens.report import summarize

from .fakes import FakeClient


def by_id(results):
    return {r.review_id: r for r in results}


def test_happy_path_and_redaction():
    client = FakeClient()
    res = by_id(ReviewPipeline(client).run([("a", "Battery dies. Mail me at x@y.com")]))["a"]
    assert res.status == "ok"
    assert res.pii_categories == ["Email"]
    assert "x@y.com" not in res.redacted_text
    assert res.aspects[0].target == "battery"
    assert "overall positive but has negative aspects" in res.warnings
    assert [e.text for e in res.entities] == ["Acme"]  # low-confidence entity filtered


def test_raw_pii_never_reaches_analysis_stages():
    client = FakeClient()
    ReviewPipeline(client).run([("a", "email me: secret@corp.com")])
    for stage, docs in client.calls:
        if stage in {"sentiment", "key_phrases", "entities"}:
            assert all("secret@corp.com" not in d["text"] for d in docs)


def test_pii_failure_is_fail_closed():
    client = FakeClient(fail_ids={"pii": {"a"}})
    res = by_id(ReviewPipeline(client).run([("a", "text"), ("b", "text")]))
    assert res["a"].status == "failed" and res["a"].sentiment is None
    assert res["b"].status == "ok"
    later = [d["id"] for s, docs in client.calls if s == "sentiment" for d in docs]
    assert "a" not in later


def test_empty_and_non_english_are_handled():
    res = by_id(ReviewPipeline(FakeClient()).run([("e", "   "), ("s", "El producto llegó roto")]))
    assert res["e"].status == "failed" and res["e"].errors[0].stage == "input"
    assert res["s"].status == "skipped" and res["s"].language == "es"


def test_batches_respect_per_stage_limits():
    client = FakeClient()
    ReviewPipeline(client).run([(str(i), f"review {i}") for i in range(25)])
    for stage, docs in client.calls:
        assert len(docs) <= (5 if stage in {"pii", "entities"} else 10), stage


def test_batch_error_message_includes_status():
    client = FakeClient(raise_on={"pii": HttpResponseError(message="Max 5 records are permitted.")})
    res = ReviewPipeline(client).run([("a", "x")])[0]
    assert res.status == "failed"
    assert "Max 5 records" in res.errors[0].message


def test_stage_failure_is_partial_not_fatal():
    client = FakeClient(fail_ids={"sentiment": {"a"}})
    res = by_id(ReviewPipeline(client).run([("a", "battery ok")]))["a"]
    assert res.status == "partial"
    assert res.key_phrases == ["battery"]  # other stages still ran


def test_whole_batch_http_error_marks_batch_and_continues():
    client = FakeClient(raise_on={"key_phrases": HttpResponseError(message="503")})
    res = ReviewPipeline(client).run([("a", "x"), ("b", "y")])
    assert all(r.status == "partial" for r in res)
    assert all(r.sentiment == "positive" for r in res)


def test_auth_error_aborts_run():
    client = FakeClient(raise_on={"language": ClientAuthenticationError(message="401")})
    with pytest.raises(ClientAuthenticationError):
        ReviewPipeline(client).run([("a", "x")])


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        ReviewPipeline(FakeClient()).run([("a", "x"), ("a", "y")])


def test_summary():
    results = ReviewPipeline(FakeClient()).run([("a", "battery bad"), ("b", "")])
    s = summarize(results)
    assert s["total"] == 2 and s["status"] == {"ok": 1, "failed": 1}
    assert s["top_complaints"][0]["aspect"] == "battery"
