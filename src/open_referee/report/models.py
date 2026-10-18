"""Report models. Schema mirrors the anchored-comment style of modern AI
referees: overall markdown report + inline comments anchored to exact quotes."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

TOOL_NAME = "open-referee"
TOOL_URL = "https://github.com/paluigi/open-referee"


class Attribution(BaseModel):
    tool: str = TOOL_NAME
    url: str = TOOL_URL
    text: str = "Built with open-referee"


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


class PlanSummary(BaseModel):
    """How the run was budgeted and how the spend actually went."""

    mode: str = "preset"  # auto | fast | standard | deep
    budget_usd: float | None = None  # hard ceiling
    target_usd: float | None = None  # soft plan target
    planned_usd: float | None = None
    spent_usd: float = 0.0
    control_spent_usd: float = 0.0  # triage/inventory/planner overhead
    projected_final_usd: float | None = None
    planner_calls_used: int = 0
    adjustments: list[str] = Field(default_factory=list)  # human-readable change log
    notes: str | None = None  # e.g. gate stops ("claim verification stopped at 18/40")


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
    plan_summary: PlanSummary | None = None  # budget-mode telemetry
    attribution: Attribution = Field(default_factory=Attribution)

    def sorted_comments(self) -> list[ReviewComment]:
        return sorted(self.comments, key=lambda c: c.score, reverse=True)


# ------------------------------------------------------- desk-reject estimator --


class FitFactor(BaseModel):
    """One dimension of the journal-fit assessment: the LLM editor's textual
    analysis (capped for brevity), which the decision model then judges."""

    name: str  # machine name, e.g. scope_fit
    label: str  # human label, e.g. "Scope fit"
    analysis: str = ""  # the editor's reasoning, ~500-600 tokens max


class FitDecision(BaseModel):
    """The decision model's verdict over the LLM's factor analyses."""

    probability: float = Field(ge=0.0, le=1.0)  # P(pass the desk screen)
    confidence: float | None = None
    question: str = ""
    source: str = "decision-model"  # decision-model | llm-fallback


class FitReport(BaseModel):
    run_id: str
    paper_title: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    journal_profile: str = ""  # the free-form venue description, echoed
    decision: FitDecision | None = None
    probability_not_desk_rejected: float = Field(default=0.0, ge=0.0, le=1.0)
    factors: list[FitFactor] = Field(default_factory=list)
    reasoning: str = ""  # markdown
    risks: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    usage: UsageSummary = Field(default_factory=UsageSummary)
    model_roles: dict[str, str] = Field(default_factory=dict)
    plan_summary: PlanSummary | None = None
    attribution: Attribution = Field(default_factory=Attribution)
