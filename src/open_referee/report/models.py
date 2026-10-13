"""Report models. Schema mirrors the anchored-comment style of modern AI
referees: overall markdown report + inline comments anchored to exact quotes."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ReviewComment(BaseModel):
    id: str
    title: str
    section: str | None = None
    block_id: str | None = None  # anchor block in the ingested document
    paragraph_anchor: str | None = None  # exact document text the comment attaches to
    quote: str | None = None  # the specific span being criticized
    message: str
    score: float = Field(ge=0.0, le=1.0)  # severity
    category: str = "general"  # math | consistency | evidence | references | clarity | novelty ...
    anchor_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class UsageSummary(BaseModel):
    tokens_by_role: dict[str, dict[str, float]] = Field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class ReviewReport(BaseModel):
    run_id: str
    paper_title: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    paper_summary: str = ""
    overall_feedback: str = ""  # markdown
    comments: list[ReviewComment] = Field(default_factory=list)
    usage: UsageSummary = Field(default_factory=UsageSummary)
    model_roles: dict[str, str] = Field(default_factory=dict)  # role -> provider/model
    validator_notes: str | None = None  # notes from the post-meta validation pass

    def sorted_comments(self) -> list[ReviewComment]:
        return sorted(self.comments, key=lambda c: c.score, reverse=True)
