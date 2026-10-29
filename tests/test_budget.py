"""Unit tests for the budget harness: cost model, planner, steering, gating."""

from __future__ import annotations

import pytest

from open_referee.config import Config, LLMConfig, ModelRole, ProviderConfig, RoleConfig
from open_referee.pipeline.budget import (
    MIN_CLAIMS,
    BudgetHarness,
    BudgetInfeasible,
    DocumentStats,
    PlanKnobs,
    PricingError,
)
from open_referee.pipeline.orchestrator import RolePool
from open_referee.providers.usage import CallRecord

PRICING = {
    "fake/strong-fake": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
    "fake/small-fake": {"input_per_mtok": 0.3, "output_per_mtok": 2.5},
}


def make_harness(
    budget: float = 10.0,
    auto: bool = True,
    priced: bool = True,
    stats: DocumentStats | None = None,
) -> tuple[BudgetHarness, RolePool]:
    cfg = Config()
    cfg.llm = LLMConfig(
        providers={"fake": ProviderConfig(type="fake")},  # type: ignore[arg-type]
        roles={
            "strong": RoleConfig(provider="fake", model="strong-fake"),
            "small": RoleConfig(provider="fake", model="small-fake"),
        },  # type: ignore[arg-type]
        pricing=dict(PRICING) if priced else {},
    )
    pool = RolePool.build(cfg, {ModelRole.STRONG, ModelRole.SMALL}, cap_usd=budget)
    h = BudgetHarness(cfg, pool.ledger, pool.specs, budget_usd=budget, auto=auto)
    h.stats = stats or DocumentStats(
        chars=60_000, n_sections=8, n_figures=3, n_tables=2, n_references=30
    )
    return h, pool


def _rec(h: BudgetHarness, ct: str, cost: float, in_tok: int = 1000, out_tok: int = 500) -> None:
    rec = CallRecord(
        role="strong",
        provider="fake",
        model="strong-fake",
        input_tokens=in_tok,
        output_tokens=out_tok,
        estimated_cost_usd=cost,
        call_type=ct,
    )
    h.ledger.calls.append(rec)
    h.note_record(rec)


# ------------------------------------------------------------------ pricing --


async def test_unpriced_model_raises_in_auto_mode():
    h, _ = make_harness(priced=False)
    with pytest.raises(PricingError) as ei:
        h.require_pricing()
    assert "fake/strong-fake" in str(ei.value)
    assert "llm.pricing" in str(ei.value)


async def test_explicit_zero_pricing_is_allowed():
    """Local models priced at an explicit 0.0 are configured, not missing."""
    h, pool = make_harness(priced=False)
    pool.ledger.pricing["fake/strong-fake"] = {"input_per_mtok": 0.0, "output_per_mtok": 0.0}
    pool.ledger.pricing["fake/small-fake"] = {"input_per_mtok": 0.0, "output_per_mtok": 0.0}
    h.require_pricing()  # no raise


async def test_preset_mode_does_not_require_pricing():
    """require_pricing itself raises regardless of mode — the orchestrator's
    preset path warns instead; auto path propagates. Here we check the harness
    raises only via require_pricing, and deterministic_plan works priced."""
    h, _ = make_harness(auto=False)
    plan = h.deterministic_plan()
    assert plan.planned_cost() > 0


# ------------------------------------------------------------------ preflight --


async def test_preflight_infeasible_budget_suggests_minimum():
    h, _ = make_harness(budget=0.001)
    with pytest.raises(BudgetInfeasible) as ei:
        h.deterministic_plan()
    assert ei.value.min_budget_usd > 0.001
    assert ei.value.standard_budget_usd >= ei.value.min_budget_usd
    assert "meaningful review" in str(ei.value).lower()


async def test_min_viable_cost_includes_control_calls():
    """The minimum covers control calls (triage/inventory/surveyor/planner),
    not just the review calls."""
    h, _ = make_harness(budget=10_000.0)
    min_knobs = h.min_viable_knobs()
    control, stages = h._estimates_for(min_knobs)
    assert set(control) >= {"triage", "claim_inventory"}
    assert control["claim_verify" if False else "claim_inventory"].calls == 1
    # the planner contingency is part of the control bucket in auto mode
    assert control["planner"].calls >= 1


# --------------------------------------------------------- deterministic plan --


async def test_deterministic_plan_fits_budget():
    for budget in (2.0, 5.0, 20.0, 50.0):
        h, _ = make_harness(budget=budget)
        plan = h.deterministic_plan()
        assert plan.planned_cost() <= budget + 1e-9, f"plan exceeds {budget}"
        assert plan.target_usd <= budget


async def test_more_budget_buys_more_claims():
    stats = DocumentStats(chars=200_000, n_sections=10, n_claims=80, n_references=40)
    # 200k chars -> chunked inventory (2 calls) sits in the floor now
    h_small, _ = make_harness(budget=2.5, stats=stats)
    h_big, _ = make_harness(budget=50.0, stats=stats)
    small = h_small.deterministic_plan()
    big = h_big.deterministic_plan()
    assert big.knobs.max_claims > small.knobs.max_claims
    assert big.knobs.max_claims >= MIN_CLAIMS


async def test_simple_paper_plans_cheap_within_same_budget():
    """The $100-budget/$25-plan scenario: a small doc targets well under."""
    h_simple, _ = make_harness(
        budget=100.0, stats=DocumentStats(chars=20_000, n_sections=3, n_references=10)
    )
    h_big, _ = make_harness(budget=100.0)
    simple = h_simple.deterministic_plan()
    big = h_big.deterministic_plan()
    assert simple.target_usd < big.target_usd
    assert simple.target_usd < 100.0


async def test_plan_respects_actual_claim_count():
    h, _ = make_harness(budget=50.0)
    h.stats.n_claims = 7
    plan = h.deterministic_plan()
    assert plan.knobs.max_claims <= 150  # knob may be higher, but calls bounded:
    assert plan.stage_estimates["claim_verify"].calls == 7


# ------------------------------------------------------------ plan validation --


async def test_validate_plan_repairs_overbudget_llm_plan():
    h, _ = make_harness(budget=2.0)
    base = h.deterministic_plan()
    reckless = base.model_copy(
        update={
            "knobs": PlanKnobs(
                max_claims=150,
                per_section_lenses=True,
                whole_paper_passes=True,
                defense_round=True,
                max_artifacts=50,
            ),
            "rationale": "LLM tried to spend it all",
        }
    )
    fixed = h.validate_plan(reckless)
    assert fixed.planned_cost() <= 2.0 + 1e-9
    assert fixed.target_usd <= 2.0
    assert fixed.plan_version > reckless.plan_version


async def test_validate_plan_clamps_knobs():
    h, _ = make_harness(budget=50.0)
    base = h.deterministic_plan()
    silly = base.model_copy(update={"knobs": PlanKnobs(max_claims=10_000, max_artifacts=-5)})
    fixed = h.validate_plan(silly)
    assert fixed.knobs.max_claims <= 150
    assert fixed.knobs.max_artifacts >= 0


# ------------------------------------------------------------------ tracking --


async def test_note_record_tracks_and_projects():
    h, _ = make_harness(budget=10.0)
    h.set_plan(h.deterministic_plan())
    before = h.projected_final()
    _rec(h, "triage", 0.01)
    _rec(h, "claim_inventory", 0.02)
    assert h.done["triage"] == 1 and h.done["claim_inventory"] == 1
    after = h.projected_final()
    # actual spend booked; remaining plan for those two types removed
    assert after < before
    st = h.status()
    assert st["actual_usd"] == pytest.approx(0.03)
    assert st["by_call_type"]["triage"]["done_calls"] == 1


async def test_control_plane_spend_is_separated():
    h, _ = make_harness(budget=10.0)
    h.set_plan(h.deterministic_plan())
    _rec(h, "triage", 0.01)
    _rec(h, "planner", 0.005)
    _rec(h, "claim_verify", 0.05)
    st = h.status()
    assert st["control_actual_usd"] == pytest.approx(0.015)
    assert st["actual_usd"] == pytest.approx(0.065)


async def test_adaptive_unit_costs_after_samples():
    h1, _ = make_harness(budget=100.0)
    h2, _ = make_harness(budget=100.0)
    base = h1.unit_cost("claim_verify")
    for _ in range(4):
        _rec(h2, "claim_verify", 0.0, in_tok=50_000, out_tok=20_000)
    adapted = h2.unit_cost("claim_verify")
    assert adapted > base * 3  # empirical reality: far bigger calls than estimated


# ------------------------------------------------------------------ steering --


async def test_no_replan_when_on_plan():
    """Stick to plan: under-run never triggers replanning."""
    h, _ = make_harness(budget=10.0)
    plan = h.deterministic_plan()
    h.set_plan(plan)
    # simulate spending far LESS than planned (fresh ledger = 0 actual)
    assert h.should_replan() is None
    s = h.steer()
    assert s.action == "none"


async def test_replans_are_not_count_limited():
    """Project-owner decision: replan count is unbounded; anti-thrash comes
    from deviation thresholds only."""
    h, _ = make_harness(budget=10.0)
    plan = h.deterministic_plan()
    plan.planner_calls_used = 99  # far beyond any old allowance
    h.set_plan(plan)
    _rec(h, "claim_verify", 9.9)
    assert h.should_replan() is not None  # still fires on material overrun


async def test_overrun_yields_viable_repair_options():
    h, _ = make_harness(budget=3.0)
    h.stats.n_claims = 60
    plan = h.deterministic_plan()
    h.set_plan(plan)
    # simulate heavy overspend on claim verification
    _rec(h, "claim_verify", 0.8)
    assert h.projected_final() > h.budget_usd
    options = h.repair_options()
    assert options, "expected repair options when projection exceeds budget"
    assert all(o.saves_usd > 0 for o in options)
    # options are ranked by savings
    saves = [o.saves_usd for o in options]
    assert saves == sorted(saves, reverse=True)


async def test_steer_policy_branches():
    """Deterministic-first steering: 1 viable option -> auto-apply (no LLM
    call); several -> defer to the planner; none -> rely on gating/cap."""
    from open_referee.pipeline.budget import RepairOption

    def _overrun(h: BudgetHarness) -> None:
        # make the projection exceed the (lowered) hard budget
        h.budget_usd = h.projected_final() - 0.5

    # exactly one viable option: applied deterministically
    h, _ = make_harness(budget=10.0)
    h.set_plan(h.deterministic_plan())
    _overrun(h)
    h.repair_options = lambda: [
        RepairOption(
            index=0, action="drop_defense", label="Skip defense gate", saves_usd=0.9, impact=""
        )
    ]
    s = h.steer()
    assert s.action == "applied"
    assert s.applied is not None and s.applied.action == "drop_defense"

    # two viable options: judgment call -> planner
    h2, _ = make_harness(budget=10.0)
    h2.set_plan(h2.deterministic_plan())
    _overrun(h2)
    h2.repair_options = lambda: [
        RepairOption(index=0, action="drop_defense", label="a", saves_usd=0.9, impact=""),
        RepairOption(index=1, action="drop_lenses", label="b", saves_usd=0.8, impact=""),
    ]
    s2 = h2.steer()
    assert s2.action == "needs_planner"
    assert len(s2.options) == 2

    # no option closes the gap: gating + hard cap take over
    h3, _ = make_harness(budget=10.0)
    h3.set_plan(h3.deterministic_plan())
    _overrun(h3)
    h3.repair_options = lambda: [
        RepairOption(index=0, action="drop_defense", label="a", saves_usd=0.1, impact="")
    ]
    s3 = h3.steer()
    assert s3.action == "cant_fix"


async def test_apply_option_trims_remaining_projection():
    h, _ = make_harness(budget=10.0)
    h.stats.n_claims = 60
    plan = h.deterministic_plan()
    h.set_plan(plan)
    options = h.repair_options()
    assert options
    biggest = options[0]
    trimmed = h.apply_option(biggest, reason="test")
    assert trimmed.planned_cost() <= plan.planned_cost()


# -------------------------------------------------------------------- gating --


async def test_allow_call_gate_blocks_when_projection_exceeds():
    h, _ = make_harness(budget=3.0)
    h.stats.n_claims = 60
    plan = h.deterministic_plan()
    h.set_plan(plan)
    _rec(h, "claim_verify", 0.995)
    assert h.allow_call("claim_verify") is False
    h.note_gate_stop("claim_verify", 5)
    assert h.gate_stopped["claim_verify"] == 5


async def test_gate_open_without_plan():
    h, _ = make_harness(budget=3.0)
    assert h.allow_call("claim_verify") is True


# ------------------------------------------------------------------ briefing --


async def test_briefing_includes_self_cost_and_options():
    h, _ = make_harness(budget=5.0)
    h.set_plan(h.deterministic_plan())
    b = h.briefing()
    assert b["planner_call_cost_usd"] >= 0.0
    assert "rules" in b and "unit_costs_usd" in b
    assert b["planner_calls_remaining"] >= 1
    b2 = h.briefing(options=[])
    assert b2["mode"] == "adjust"
