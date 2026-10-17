"""Renderer metadata: build-metadata table, budget lines, attribution."""

from __future__ import annotations

from open_referee.report.models import (
    FitFactor,
    FitReport,
    PlanSummary,
    ReviewComment,
    ReviewReport,
    UsageSummary,
)
from open_referee.report.render import (
    _MD_LINK_RE,
    render_fit_markdown,
    render_markdown,
)


def _review_report() -> ReviewReport:
    return ReviewReport(
        run_id="r1",
        paper_title="Widget Networks",
        paper_summary="A paper.",
        overall_feedback="## Verdict\n\nSolid.",
        comments=[
            ReviewComment(
                id="c001",
                title="Gap",
                message="No proof.",
                score=0.8,
                paragraph_anchor="Theorem 1. Every widget network is stable.",
            )
        ],
        usage=UsageSummary(
            tokens_by_role={
                "strong": {
                    "input_tokens": 12000,
                    "output_tokens": 3400,
                    "cost_usd": 0.0921,
                    "calls": 4,
                },
                "small": {
                    "input_tokens": 8000,
                    "output_tokens": 2100,
                    "cost_usd": 0.0078,
                    "calls": 6,
                },
            },
            estimated_cost_usd=0.0999,
            total_input_tokens=20000,
            total_output_tokens=5500,
        ),
        model_roles={
            "strong": "openrouter/claude-sonnet-4.5",
            "small": "ollama/llama3.1:8b",
        },
        plan_summary=PlanSummary(
            mode="auto",
            budget_usd=10.0,
            target_usd=4.5,
            planned_usd=4.62,
            spent_usd=4.31,
            control_spent_usd=0.21,
            projected_final_usd=4.55,
            planner_calls_used=2,
            adjustments=["[steering] drop_lenses: Skip the per-section lens grid"],
            notes="claim_verify: 3 unit(s) skipped by the budget gate",
        ),
    )


def test_review_markdown_has_build_metadata_table():
    md = render_markdown(_review_report())
    assert "## Build metadata" in md
    assert "| strong | `openrouter/claude-sonnet-4.5` | 4 | 12,000 | 3,400 | $0.0921 |" in md
    assert "| small | `ollama/llama3.1:8b` | 6 | 8,000 | 2,100 | $0.0078 |" in md
    assert "**Total**" in md and "$0.0999" in md


def test_review_markdown_has_budget_lines_and_attribution():
    md = render_markdown(_review_report())
    assert "Budget: $10.00" in md
    assert "planned target $4.50" in md
    assert "control-plane overhead $0.21" in md
    assert "2 planner call(s)" in md
    assert "adjustment: [steering] drop_lenses" in md
    assert "skipped by the budget gate" in md
    assert "Built with [open-referee](https://github.com/paluigi/open-referee)" in md


def test_review_markdown_without_plan_omits_budget_block():
    report = _review_report()
    report.plan_summary = None
    md = render_markdown(report)
    assert "Budget:" not in md
    assert "## Build metadata" in md  # usage table still present


def _fit_report() -> FitReport:
    return FitReport(
        run_id="f1",
        paper_title="Widget Networks",
        journal_profile="Top field journal, high novelty bar.",
        probability_not_desk_rejected=0.62,
        llm_holistic_estimate=0.55,
        factors=[
            FitFactor(
                name="scope_fit",
                label="Scope fit",
                score=0.8,
                rationale="Squarely in scope.",
                evidence="widgets",
            ),
            FitFactor(name="aim_alignment", label="Aim alignment", score=0.7, rationale="ok"),
        ],
        calibration={"bias": -3.6, "scope_fit": 2.2, "aim_alignment": 1.8},
        reasoning="## Weighing\n\nScope is fine.",
        risks=["Novelty borderline"],
        suggestions=["Sharpen contribution"],
        usage=UsageSummary(
            tokens_by_role={
                "strong": {
                    "input_tokens": 9000,
                    "output_tokens": 1500,
                    "cost_usd": 0.05,
                    "calls": 1,
                }
            },
            estimated_cost_usd=0.05,
            total_input_tokens=9000,
            total_output_tokens=1500,
        ),
        model_roles={"strong": "openrouter/claude-sonnet-4.5"},
    )


def test_fit_markdown_renders_probability_factors_calibration():
    md = render_fit_markdown(_fit_report())
    assert "# Journal fit estimate — Widget Networks" in md
    assert "**62%** (calibrated decision model)" in md
    assert "LLM holistic estimate: **55%**" in md
    assert "sigmoid(-3.6" in md and "2.2·scope_fit" in md
    assert "### Scope fit — 0.80 / 1.00" in md
    assert "> widgets" in md
    assert "## Key risks" in md and "- Novelty borderline" in md
    assert "## Venue profile (as provided)" in md
    assert "## Build metadata" in md
    assert "Built with [open-referee](https://github.com/paluigi/open-referee)" in md


def test_reportlab_fallback_link_conversion():
    assert (
        _MD_LINK_RE.sub(
            r"\1 (\2)", "Built with [open-referee](https://github.com/paluigi/open-referee)"
        )
        == "Built with open-referee (https://github.com/paluigi/open-referee)"
    )
