"""Tests for the planning committee: frontier feasibility, strategy protocol,
residual-frontier replanning, wrap-up, and the uncapped preset mode."""

from __future__ import annotations

import json

import pytest

from open_referee.config import Config, LLMConfig, ProviderConfig, RoleConfig
from open_referee.pipeline.budget import (
    PRESET_KNOBS,
    BudgetHarness,
    DocumentStats,
)
from open_referee.pipeline.orchestrator import RolePool
from open_referee.providers.base import ModelSpec
from open_referee.providers.decisions import DecisionAnswer, FakeDecisionProvider
from open_referee.providers.usage import CallRecord

PRICING = {
    "fake/strong-fake": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
    "fake/small-fake": {"input_per_mtok": 0.3, "output_per_mtok": 2.5},
    "fakedec/decider-fake": {"input_per_mtok": 0.02, "output_per_mtok": 0.0},
}


def make_pool(budget: float = 50.0):
    cfg = Config()
    cfg.llm = LLMConfig(
        providers={
            "fake": ProviderConfig(type="fake"),  # type: ignore[arg-type]
            "fakedec": ProviderConfig(  # type: ignore[arg-type]
                type="decision", base_url="http://fake.local"
            ),
        },
        roles={
            "strong": RoleConfig(provider="fake", model="strong-fake"),
            "small": RoleConfig(provider="fake", model="small-fake"),
        },
        pricing=dict(PRICING),
    )
    pool = RolePool.build(
        cfg,
        {__import__("open_referee.config", fromlist=["ModelRole"]).ModelRole.STRONG},
        cap_usd=budget,
    )
    decider = FakeDecisionProvider(model="decider-fake")
    decider.provider_name = "fakedec"
    pool.decider = decider
    pool.decider_spec = ModelSpec(
        provider_name="fakedec",
        provider_type="decision",
        model="decider-fake",
        base_url="http://fake.local",
    )
    return cfg, pool


def make_harness(pool, budget: float = 50.0, stats: DocumentStats | None = None) -> BudgetHarness:
    from open_referee.config import ModelRole

    h = BudgetHarness(pool.cfg, pool.ledger, pool.specs, budget_usd=budget, auto=True)
    h.specs[ModelRole.DECISION] = pool.decider_spec
    h.stats = stats or DocumentStats(
        chars=60_000, n_sections=8, n_figures=3, n_tables=2, n_references=30, n_claims=60
    )
    pool.harness = h
    return h


def _rec(h, ct, cost, in_tok=1000, out_tok=500):
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


# ------------------------------------------------------------------- frontier --


async def test_frontier_paths_are_feasible_by_construction():
    _, pool = make_pool()
    h = make_harness(pool, budget=2.0)
    frontier = h.frontier(limit=10)
    assert 1 <= len(frontier) <= 10
    assert len({p.name for p in frontier}) == len(frontier)
    for p in frontier:
        assert p.est_cost_usd <= h.effective_budget()
        assert p.claims_count <= 60  # capped by actual claim count
    # a generous budget keeps the depth-maximal and breadth-heavy ends
    _, pool2 = make_pool()
    h2 = make_harness(pool2, budget=50.0)
    names = {p.name for p in h2.frontier()}
    assert "maximal" in names and "lean-core" in names


async def test_frontier_empty_when_budget_too_small():
    _, pool = make_pool()
    h = make_harness(pool, budget=0.001)
    assert h.frontier() == []


async def test_preset_estimates_are_mechanical_and_ordered():
    _, pool = make_pool()
    h = make_harness(pool)
    est = h.preset_estimates()
    assert set(est) == {"fast", "standard", "deep"}
    assert 0 < est["fast"] < est["standard"]
    assert est["deep"] >= est["standard"]  # equal when claims/artifacts are capped out


# ------------------------------------------------------------ strategy committee --


class StubPipeline:
    """Minimal pipeline double for committee functions."""

    def __init__(self, pool):
        self._pool = pool
        self._triage_cache = {"domain": "economics"}
        from open_referee.pipeline.state import RunState

        self.state = RunState(run_id="c01", paper_path="/tmp/x")

        self.run_id = "c01"
        self._events = []

    async def _emit(self, *a, **k):
        self._events.append(a)

    def _sync_budget_to_state(self):
        h = self._pool.harness
        self.state.plan = h.plan.model_dump(mode="json") if h.plan else None


async def test_strategy_committee_commit_feasible_path():
    from open_referee.pipeline import committee

    _, pool = make_pool()
    h = make_harness(pool, budget=2.0)
    h.set_plan(h.deterministic_plan(), reason="preflight-v0")
    # script: reflection (strong), DM path choice, DM confirmation
    pool.providers[
        __import__("open_referee.config", fromlist=["ModelRole"]).ModelRole.STRONG
    ].queue(
        json.dumps(
            {
                "draft": "d",
                "critique": "c",
                "evaluation_summary": "lean",
                "preferred_path": "claims-light",
            }
        )
    )
    pool.decider.queue(
        [DecisionAnswer(type="choice", name="path_choice", choice="claims-light", confidence=0.9)]
    )
    pool.decider.queue([DecisionAnswer(type="predicate", name="confirm_path", probability=0.85)])
    stub = StubPipeline(pool)
    claims = [
        {
            "id": f"cl{i}",
            "type": "numerical",
            "importance": "central",
            "components": {"x": 1},
            "anchor": f"a{i}",
        }
        for i in range(20)
    ]
    await committee.strategy_committee(stub, pool, claims)
    assert h.plan is not None
    assert h.plan.based_on == "frontier"
    assert h.plan.planned_cost() <= h.effective_budget()
    assert any(a.reason == "strategy-committee" for a in h.plan.adjustments)
    # committee transcript is in the plan log
    assert any(e["reason"].startswith("frontier:") for e in h.plan_log)


async def test_rank_claims_uses_decision_model_order():
    from open_referee.pipeline import committee

    _, pool = make_pool()
    h = make_harness(pool)
    # cl02 scores critical, cl01 scores low -> order flips
    pool.decider.queue(
        [
            DecisionAnswer(type="score", name="claim_cl01", score=0, confidence=0.9),
            DecisionAnswer(type="score", name="claim_cl02", score=3, confidence=0.9),
        ]
    )
    stub = StubPipeline(pool)
    claims = [
        {
            "id": "cl01",
            "type": "numerical",
            "importance": "central",
            "components": {},
            "anchor": "a",
        },
        {
            "id": "cl02",
            "type": "theorem_proof",
            "importance": "supporting",
            "components": {},
            "anchor": "b",
        },
    ]
    ranked = await committee.rank_claims(stub, pool, claims)
    assert ranked[0]["id"] == "cl02"
    assert any(c.call_type == "dm_rank" for c in h.ledger.calls)


# ------------------------------------------------------------- replan + wrap-up --


async def test_residual_frontier_feasible_and_replan_applies():
    from open_referee.pipeline import committee

    _, pool = make_pool()
    h = make_harness(pool, budget=3.0)
    h.set_plan(h.deterministic_plan(), reason="preflight-v0")
    # simulate progress: control plane + 10 claims verified + heavy spend
    _rec(h, "triage", 0.02)
    _rec(h, "claim_inventory", 0.03)
    for _ in range(10):
        _rec(h, "claim_verify", 0.02)
    paths = h.residual_frontier()
    assert paths, "expected at least the lean continuation to fit"
    for p in paths:
        assert p.projected_final_usd <= h.effective_budget()
    stub = StubPipeline(pool)
    # single-option auto-apply (only 'lean-finish' fits) needs a tight budget:
    # force it by raising spend
    _rec(h, "claim_verify", 0.9)
    tight = h.residual_frontier()
    applied = await committee.replan_choice(stub, pool, "test overrun")
    assert applied is True
    assert h.plan.based_on == "residual-frontier"
    assert tight is not None


async def test_wrapup_when_nothing_is_feasible():
    from open_referee.pipeline import committee

    _, pool = make_pool()
    h = make_harness(pool, budget=2.0)
    h.set_plan(h.deterministic_plan(), reason="preflight-v0")
    _rec(h, "triage", 0.02)
    _rec(h, "claim_inventory", 0.03)
    for _ in range(12):
        _rec(h, "claim_verify", 0.15)  # spend past every continuation
    assert h.residual_frontier() == []
    stub = StubPipeline(pool)
    stub._ingested_doc = None  # no doc -> wrap-up raises BudgetExceeded path
    from open_referee.providers import BudgetExceeded

    with pytest.raises(BudgetExceeded):
        await committee.wrapup_incomplete(stub, pool, "test")
    assert stub.state.status == "incomplete"
    assert stub.state.shortfall_usd is not None


# ------------------------------------------------------------ uncapped presets --


async def test_preset_mode_is_uncapped():
    """Preset plans track spend but never enforce a cap."""
    _, pool = make_pool()
    h = BudgetHarness(pool.cfg, pool.ledger, pool.specs, budget_usd=0.05, auto=False)
    h.stats = DocumentStats(chars=60_000, n_claims=60)
    pool.harness = h
    import math

    pool.ledger.cap_usd = math.inf  # what _init_budget does in preset mode
    h.set_plan(h.preset_plan(PRESET_KNOBS["standard"], "standard"), reason="preset-tracking")
    _rec(h, "claim_verify", 5.0)  # wild overspend vs the 0.05 'budget'
    assert h.should_replan() is None  # no replanning in preset mode
    assert h.allow_call("claim_verify") is True  # gates off

    spec = pool.specs[__import__("open_referee.config", fromlist=["ModelRole"]).ModelRole.STRONG]
    await h.ledger.check_budget(spec, 1000, 1000)  # never trips at inf
