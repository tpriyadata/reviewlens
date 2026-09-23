"""ReviewLens pipeline.

    input check -> language -> PII redaction -> { sentiment+opinions, key phrases, entities }

Design rules:
- PII redaction is a gate. If redaction fails for a review, that review goes no further,
  so raw text is never sent to the other stages or written to output.
- Failures are recorded per review and per stage. One bad document never fails a batch,
  and one failed batch never fails the run. The exception is authentication, which aborts.
- The three analysis stages run independently on the redacted text, so a sentiment
  failure does not cost you key phrases or entities.
- Redaction is done locally from the service's entity offsets, so we control which
  categories count as personal data (PersonType - roles like "customer support" - does not).
  Every character we mask must also be masked by the service; any disagreement fails closed.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ServiceRequestError,
)

from .batching import MAX_DOCS_PER_REQUEST, STAGE_BATCH_LIMITS, chunked, truncate
from .models import Assessment, AspectOpinion, Entity, ReviewResult
from .redaction import DEFAULT_IGNORED_PII, mask_spans

log = logging.getLogger(__name__)

StageCall = Callable[[list[dict[str, str]]], Sequence[Any]]
StageApply = Callable[[Any, ReviewResult], None]


def _describe(exc: Exception) -> str:
    """Status code + service error code + first line of the message, so failures are debuggable."""
    status = getattr(exc, "status_code", None)
    code = getattr(getattr(exc, "error", None), "code", None)
    message = (getattr(exc, "message", None) or str(exc)).splitlines()[0][:200]
    parts = [type(exc).__name__]
    if status:
        parts.append(f"status={status}")
    if code:
        parts.append(f"code={code}")
    return " ".join(parts) + f": {message}"


class ReviewPipeline:
    def __init__(
        self,
        client: Any,
        allowed_languages: Iterable[str] = ("en",),
        batch_size: int = MAX_DOCS_PER_REQUEST,
        min_entity_confidence: float = 0.6,
        pii_ignore_categories: Iterable[str] = DEFAULT_IGNORED_PII,
    ) -> None:
        if not 1 <= batch_size <= MAX_DOCS_PER_REQUEST:
            raise ValueError(f"batch_size must be 1..{MAX_DOCS_PER_REQUEST}")
        self.client = client
        self.allowed_languages = frozenset(allowed_languages)
        self.batch_size = batch_size
        self.min_entity_confidence = min_entity_confidence
        self.pii_ignore_categories = frozenset(pii_ignore_categories)

    # ------------------------------------------------------------------ public
    def run(self, reviews: Iterable[tuple[str, str]]) -> list[ReviewResult]:
        results: dict[str, ReviewResult] = {}
        texts: dict[str, str] = {}

        for review_id, raw in reviews:
            review_id = str(review_id).strip()
            if not review_id:
                raise ValueError("review_id must be non-empty")
            if review_id in results:
                raise ValueError(f"duplicate review_id: {review_id}")
            result = ReviewResult(review_id=review_id)
            results[review_id] = result

            text = (raw or "").strip()
            if not text:
                result.fail("input", "empty text")
                continue
            text, was_truncated = truncate(text)
            if was_truncated:
                result.warnings = [*result.warnings, "text truncated to service limit"]
            texts[review_id] = text

        ids = list(texts)
        ids = self._run_stage("language", ids, texts, results, self._call_language,
                              self._apply_language, send_language=False)
        ids = [i for i in ids if not results[i].skipped_reason]

        ids = self._run_stage("pii", ids, texts, results, self._call_pii, self._make_apply_pii(texts))

        for name, call, apply in (
            ("sentiment", self._call_sentiment, self._apply_sentiment),
            ("key_phrases", self._call_key_phrases, self._apply_key_phrases),
            ("entities", self._call_entities, self._apply_entities),
        ):
            self._run_stage(name, ids, texts, results, call, apply)

        return list(results.values())

    # ------------------------------------------------------------ stage runner
    def _run_stage(
        self,
        stage: str,
        ids: list[str],
        texts: dict[str, str],
        results: dict[str, ReviewResult],
        call: StageCall,
        apply: StageApply,
        send_language: bool = True,
    ) -> list[str]:
        """Run one stage over ids in batches. Returns the ids that succeeded."""
        succeeded: list[str] = []
        size = min(self.batch_size, STAGE_BATCH_LIMITS[stage])
        for batch in chunked(ids, size):
            docs = []
            for rid in batch:
                doc = {"id": rid, "text": texts[rid]}
                if send_language and results[rid].language:
                    doc["language"] = results[rid].language
                docs.append(doc)

            try:
                responses = call(docs)
            except ClientAuthenticationError:
                raise  # wrong key/endpoint: every call will fail, so stop now
            except (HttpResponseError, ServiceRequestError) as exc:
                detail = _describe(exc)
                log.error("stage=%s batch of %d failed: %s", stage, len(batch), detail)
                for rid in batch:
                    results[rid].fail(stage, f"request failed: {detail}")
                continue

            seen: set[str] = set()
            for doc in responses:
                result = results.get(doc.id)
                if result is None:
                    log.warning("stage=%s unexpected id in response: %s", stage, doc.id)
                    continue
                seen.add(doc.id)
                if doc.is_error:
                    result.fail(stage, f"{doc.error.code}: {doc.error.message}")
                    continue
                try:
                    apply(doc, result)
                except ValueError as exc:  # includes pydantic ValidationError
                    result.fail(stage, f"invalid response: {exc}")
                    continue
                succeeded.append(doc.id)

            for rid in set(batch) - seen:
                results[rid].fail(stage, "missing from service response")

        log.info("stage=%s ok=%d/%d", stage, len(succeeded), len(ids))
        return succeeded

    # ------------------------------------------------------------ service calls
    def _call_language(self, docs):
        return self.client.detect_language(docs)

    def _call_pii(self, docs):
        return self.client.recognize_pii_entities(docs)

    def _call_sentiment(self, docs):
        return self.client.analyze_sentiment(docs, show_opinion_mining=True)

    def _call_key_phrases(self, docs):
        return self.client.extract_key_phrases(docs)

    def _call_entities(self, docs):
        return self.client.recognize_entities(docs)

    # ---------------------------------------------------------- result mappers
    def _apply_language(self, doc, result: ReviewResult) -> None:
        lang = (doc.primary_language.iso6391_name or "").lower()
        result.language = lang or None
        if lang not in self.allowed_languages:
            result.skipped_reason = f"language '{lang or 'unknown'}' not in allowed set"

    def _make_apply_pii(self, texts: dict[str, str]) -> StageApply:
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

    @staticmethod
    def _apply_sentiment(doc, result: ReviewResult) -> None:
        result.sentiment = doc.sentiment
        s = doc.confidence_scores
        result.confidence = {"positive": s.positive, "neutral": s.neutral, "negative": s.negative}
        aspects = []
        for sentence in doc.sentences:
            for op in sentence.mined_opinions or []:
                aspects.append(AspectOpinion(
                    target=op.target.text,
                    sentiment=op.target.sentiment,
                    assessments=[
                        Assessment(text=a.text, sentiment=a.sentiment, is_negated=bool(a.is_negated))
                        for a in op.assessments
                    ],
                ))
        result.aspects = aspects
        if doc.sentiment == "positive" and any(a.sentiment == "negative" for a in aspects):
            result.warnings = [*result.warnings, "overall positive but has negative aspects"]

    @staticmethod
    def _apply_key_phrases(doc, result: ReviewResult) -> None:
        result.key_phrases = list(doc.key_phrases)

    def _apply_entities(self, doc, result: ReviewResult) -> None:
        result.entities = [
            Entity(text=e.text, category=str(e.category), confidence=e.confidence_score)
            for e in doc.entities
            if e.confidence_score >= self.min_entity_confidence
        ]
