import pytest

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
