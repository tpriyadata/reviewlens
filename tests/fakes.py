"""A fake TextAnalyticsClient that returns real SDK model objects, so no Azure calls are made."""
from __future__ import annotations

import re

from azure.ai.textanalytics import (
    AnalyzeSentimentResult,
    AssessmentSentiment,
    CategorizedEntity,
    DetectedLanguage,
    DetectLanguageResult,
    DocumentError,
    ExtractKeyPhrasesResult,
    MinedOpinion,
    PiiEntity,
    RecognizeEntitiesResult,
    RecognizePiiEntitiesResult,
    SentenceSentiment,
    SentimentConfidenceScores,
    TargetSentiment,
    TextAnalyticsError,
)

EMAIL = re.compile(r"\S+@\S+")


def _err(doc_id: str) -> DocumentError:
    return DocumentError(id=doc_id, error=TextAnalyticsError(code="InvalidDocument", message="bad doc"),
                         is_error=True)


class FakeClient:
    def __init__(self, fail_ids: dict[str, set[str]] | None = None, raise_on: dict | None = None):
        self.calls: list[tuple[str, list[dict]]] = []
        self.fail_ids = fail_ids or {}   # stage -> ids that return DocumentError
        self.raise_on = raise_on or {}   # stage -> exception raised for the whole batch

    def _record(self, stage, docs):
        self.calls.append((stage, docs))
        if stage in self.raise_on:
            raise self.raise_on[stage]

    def _bad(self, stage, doc_id):
        return doc_id in self.fail_ids.get(stage, set())

    def detect_language(self, docs):
        self._record("language", docs)
        out = []
        for d in docs:
            if self._bad("language", d["id"]):
                out.append(_err(d["id"])); continue
            iso = "es" if "producto" in d["text"] else "en"
            out.append(DetectLanguageResult(id=d["id"], is_error=False, primary_language=DetectedLanguage(
                name=iso, iso6391_name=iso, confidence_score=1.0)))
        return out

    def recognize_pii_entities(self, docs):
        self._record("pii", docs)
        out = []
        for d in docs:
            if self._bad("pii", d["id"]):
                out.append(_err(d["id"])); continue
            ents = [PiiEntity(text=m.group(), category="Email", confidence_score=0.9)
                    for m in EMAIL.finditer(d["text"])]
            redacted = EMAIL.sub(lambda m: "*" * len(m.group()), d["text"])
            out.append(RecognizePiiEntitiesResult(id=d["id"], is_error=False, entities=ents,
                                                  redacted_text=redacted))
        return out

    def analyze_sentiment(self, docs, show_opinion_mining=False):
        self._record("sentiment", docs)
        out = []
        for d in docs:
            if self._bad("sentiment", d["id"]):
                out.append(_err(d["id"])); continue
            neg = "battery" in d["text"].lower()
            op = MinedOpinion(
                target=TargetSentiment(text="battery", sentiment="negative"),
                assessments=[AssessmentSentiment(text="fast", sentiment="negative", is_negated=False)],
            ) if neg else None
            out.append(AnalyzeSentimentResult(
                id=d["id"], is_error=False, sentiment="positive",
                confidence_scores=SentimentConfidenceScores(positive=0.9, neutral=0.05, negative=0.05),
                sentences=[SentenceSentiment(text=d["text"], mined_opinions=[op] if op else [])],
            ))
        return out

    def extract_key_phrases(self, docs):
        self._record("key_phrases", docs)
        return [ExtractKeyPhrasesResult(id=d["id"], is_error=False, key_phrases=["battery"])
                for d in docs]

    def recognize_entities(self, docs):
        self._record("entities", docs)
        return [RecognizeEntitiesResult(id=d["id"], is_error=False, entities=[
            CategorizedEntity(text="Acme", category="Organization", confidence_score=0.95),
            CategorizedEntity(text="thing", category="Product", confidence_score=0.2),
        ]) for d in docs]
