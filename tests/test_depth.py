"""Verification depth engine: multi-pass voting, arbitration, auditor routing,
per-call output budgets, reasoning-effort plumbing."""

from __future__ import annotations

import pytest

from open_referee.config import ModelRole
from open_referee.ingestion.readers import document_from_markdown
from open_referee.pipeline.budget import PlanKnobs, call_max_tokens
from open_referee.pipeline.claims import _aggregate_passes, claim_role

DOC = document_from_markdown(
    "# T\n\n## 1 Model\n\nThe theorem holds under mild assumptions. "
    + "body " * 80,
    source_format="md",
    title_hint="t",
)

CLAIM = {
    "id": "cl001",
    "type": "numerical",
    "importance": "central",
    "anchor": "The theorem holds",
    "components": {"claim": "0.313 is the pooled MAE"},
}


def _comment(title: str, score: float = 0.7) -> dict:
    return {"title": title, "message": f"Issue about {title}.", "score": score}


class StubPool:
    def __init__(self):
        self.arbitrate_verdicts: list[dict] = []

    async def complete_json(self, role, system, user, call_type="other", max_tokens=None):
        assert call_type == "claim_arbitrate"
        return self.arbitrate_verdicts.pop(0)


class StubPipeline:
    def budget_allows(self, ct):
        return True


# ------------------------------------------------------------- aggregation --


@pytest.mark.asyncio
async def test_reproduced_finding_confirmed_without_arbitration():
    pool = StubPool()
    out = await _aggregate_passes(
        StubPipeline(), pool, CLAIM,
        [[_comment("MAE aggregation undefined")], [_comment("MAE aggregation not defined")]],
        passes=2,
    )
    assert len(out) == 1
    assert out[0]["provenance"] == "claim_verify 2-of-2"
    assert pool.arbitrate_verdicts == []


@pytest.mark.asyncio
async def test_single_pass_finding_requires_arbitration_confirm():
    pool = StubPool()
    pool.arbitrate_verdicts = [
        {"confirmed": True, "message": "Refined finding.", "severity": 0.8}
    ]
    out = await _aggregate_passes(
        StubPipeline(), pool, CLAIM, [[_comment("Odd issue")], []], passes=2
    )
    assert len(out) == 1
    assert out[0]["provenance"] == "claim_verify 1-of-2 + arbitrated"
    assert out[0]["message"] == "Refined finding."
    assert out[0]["score"] == 0.8


@pytest.mark.asyncio
async def test_single_pass_finding_dismissed_by_arbitration():
    pool = StubPool()
    pool.arbitrate_verdicts = [{"confirmed": False, "message": "noise"}]
    out = await _aggregate_passes(
        StubPipeline(), pool, CLAIM, [[], [_comment("Odd issue")]], passes=2
    )
    assert out == []


# ----------------------------------------------------------------- routing --


def test_claim_routing_levels():
    assert claim_role("theorem_proof", "base") is ModelRole.STRONG
    assert claim_role("numerical", "base") is ModelRole.SMALL
    assert claim_role("numerical", "strong") is ModelRole.STRONG
    assert claim_role("table_text", "strong") is ModelRole.STRONG
    assert claim_role("figure_text", "strong") is ModelRole.SMALL
    assert claim_role("citation_claim", "all_strong") is ModelRole.STRONG
    assert claim_role("whatever", "all_strong") is ModelRole.STRONG


# ------------------------------------------------------------ output budgets --


def test_call_max_tokens_table():
    assert call_max_tokens("claim_verify", "low") == 4096
    assert call_max_tokens("claim_verify", "high") == 12288
    assert call_max_tokens("claim_inventory", "high") == 16384
    assert call_max_tokens("surveyor", "high") is None  # unlisted -> role default


def test_knobs_carry_depth_axes():
    k = PlanKnobs(verify_passes=3, claim_routing="all_strong", output_budget="high")
    assert k.verify_passes == 3
    assert PlanKnobs().verify_passes == 1  # default keeps single-pass behavior


def test_budget_counts_multiply_with_passes():
    from open_referee.pipeline.budget import DocumentStats
    from test_budget import make_harness

    stats = DocumentStats(chars=60_000, n_sections=6, n_claims=20, n_references=10)
    h1, _ = make_harness(budget=50.0, stats=stats)
    h2, _ = make_harness(budget=50.0, stats=stats)
    c1 = h1._call_counts(PlanKnobs(verify_passes=1))
    c2 = h2._call_counts(PlanKnobs(verify_passes=3))
    assert c2["claim_verify"] == 3 * c1["claim_verify"]
    assert c2["claim_arbitrate"] == 3  # 15% of 20 claims
    assert "claim_arbitrate" not in c1


# ------------------------------------------------------- max_tokens override --


@pytest.mark.asyncio
async def test_complete_json_applies_and_restores_max_tokens():
    from open_referee.pipeline.orchestrator import RolePool
    from open_referee.providers.base import ModelSpec
    from open_referee.providers.usage import UsageLedger

    seen: list[ModelSpec] = []

    class Recorder:
        def __init__(self, spec):
            self.spec = spec

        async def complete(self, messages, json_mode=False):
            seen.append(self.spec)
            from open_referee.providers.base import LLMResponse, Usage

            return LLMResponse(
                text='{"ok": true}', usage=Usage(input_tokens=1, output_tokens=1)
            )

        async def close(self):
            pass

    pool = RolePool.__new__(RolePool)
    pool.ledger = UsageLedger(100.0, {})
    pool.providers = {ModelRole.STRONG: Recorder(None)}
    pool.specs = {
        ModelRole.STRONG: ModelSpec(
            provider_name="x", provider_type="openai_compatible", model="m", max_tokens=4096
        )
    }
    pool.harness = None
    out = await pool.complete_json(
        ModelRole.STRONG, "s", "u", call_type="triage", max_tokens=16384
    )
    assert out == {"ok": True}
    assert seen[0].call_max_tokens == 16384
    assert seen[0].max_tokens == 4096
    # restored after the call
    assert pool.specs[ModelRole.STRONG].call_max_tokens is None
    assert pool.providers[ModelRole.STRONG].spec is None or (
        pool.providers[ModelRole.STRONG].spec.call_max_tokens is None
    )


# ------------------------------------------------- reasoning effort plumbing --


def test_openai_payload_includes_reasoning_effort():
    import asyncio

    from open_referee.providers.adapters import OpenAICompatibleProvider
    from open_referee.providers.base import ChatMessage, ModelSpec

    spec = ModelSpec(
        provider_name="vercel",
        provider_type="openai_compatible",
        model="m",
        base_url="https://x.example/v1",
        reasoning_effort="high",
    )
    provider = OpenAICompatibleProvider(spec)
    captured: dict = {}

    async def fake_post(url, json=None, headers=None):
        captured.update(json)

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }

        return R()

    provider._client.post = fake_post

    async def run():
        return await provider._complete_once(
            [ChatMessage(role="user", content="hi")], json_mode=False
        )

    asyncio.new_event_loop().run_until_complete(run())
    assert captured.get("reasoning_effort") == "high"
    assert captured.get("max_tokens") == 4096


# ------------------------------------------------------------- presets/frontier --


def test_preset_depth_axes():
    from open_referee.pipeline.budget import PRESET_KNOBS

    fast, std, deep = PRESET_KNOBS["fast"], PRESET_KNOBS["standard"], PRESET_KNOBS["deep"]
    assert fast.verify_passes == 1 and not fast.recompute and fast.output_budget == "low"
    assert std.verify_passes == 2 and std.claim_routing == "strong" and std.recompute
    assert deep.verify_passes == 3 and deep.claim_routing == "all_strong"
    assert deep.defense_second_round and deep.max_artifacts == 50
    # strict ordering of expected cost ingredients
    assert fast.verify_passes < std.verify_passes < deep.verify_passes


def test_frontier_includes_depth_archetypes():
    import sys

    sys.path.insert(0, "tests")
    from open_referee.pipeline.budget import DocumentStats
    from test_budget import make_harness

    stats = DocumentStats(chars=120_000, n_sections=8, n_claims=40, n_references=20)
    h, _ = make_harness(budget=500.0, stats=stats)
    names = [p.name for p in h.frontier(limit=20)]
    assert "cross-check" in names and "rigor-max" in names
    cross = next(p for p in h.frontier(limit=20) if p.name == "cross-check")
    assert cross.knobs.verify_passes == 2 and cross.knobs.claim_routing == "strong"
