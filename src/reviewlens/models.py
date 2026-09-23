"""Output schemas. Pydantic enforces the contract on every field we write."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

DocSentiment = Literal["positive", "neutral", "negative", "mixed"]
TargetPolarity = Literal["positive", "mixed", "negative"]
Status = Literal["ok", "partial", "failed", "skipped"]

# Stages that must succeed for any downstream analysis to exist.
GATING_STAGES = frozenset({"input", "language", "pii"})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class StageError(_Strict):
    stage: str
    message: str


class Assessment(_Strict):
    text: str
    sentiment: TargetPolarity
    is_negated: bool = False


class AspectOpinion(_Strict):
    target: str
    sentiment: TargetPolarity
    assessments: list[Assessment] = Field(default_factory=list)


class Entity(_Strict):
    text: str
    category: str
    confidence: float = Field(ge=0.0, le=1.0)


class ReviewResult(_Strict):
    """One row of output. Never holds the raw review text, only the redacted version."""

    review_id: str
    language: str | None = None
    skipped_reason: str | None = None
    redacted_text: str | None = None
    pii_categories: list[str] = Field(default_factory=list)
    sentiment: DocSentiment | None = None
    confidence: dict[str, float] = Field(default_factory=dict)
    aspects: list[AspectOpinion] = Field(default_factory=list)
    key_phrases: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[StageError] = Field(default_factory=list)

    def fail(self, stage: str, message: str) -> None:
        self.errors = [*self.errors, StageError(stage=stage, message=message)]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> Status:
        if self.skipped_reason:
            return "skipped"
        if not self.errors:
            return "ok"
        if any(e.stage in GATING_STAGES for e in self.errors):
            return "failed"
        return "partial"
