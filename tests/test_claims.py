"""Tests for the claim-based verification engine: artifacts, presets,
dispatch, defense round."""

from __future__ import annotations

import json

import pytest

from open_referee.config import ReviewConfig
from open_referee.ingestion.artifacts import (
    build_definitions_pack,
    extract_tables,
    extract_theorems,
)
from open_referee.ingestion.readers import document_from_markdown
from open_referee.pipeline.claims import (
    CLAIM_TYPE_ROLES,
    _definitions_for,
    _passage_for,
    defense_round,
    find_mentions,
    verify_claims,
)
from open_referee.pipeline.orchestrator import ReviewPipeline, RolePool

# ------------------------------------------------------------------ presets --


def test_depth_presets():
    fast = ReviewConfig(depth="fast").depth_preset()
    std = ReviewConfig(depth="standard").depth_preset()
    deep = ReviewConfig(depth="deep").depth_preset()
    assert not fast.per_section_lenses and not fast.defense_round
    assert fast.max_claims == 20
    assert std.per_section_lenses and std.defense_round and std.max_claims == 60
    assert std.verify_passes == 2 and std.claim_routing == "strong"
    assert std.recompute and std.synthesis and not std.defense_second_round
    assert deep.max_claims == 150 and deep.max_artifacts == 50
    assert deep.verify_passes == 3 and deep.claim_routing == "all_strong"
    assert deep.defense_second_round and deep.output_budget == "high"
    assert fast.verify_passes == 1 and not fast.recompute and fast.output_budget == "low"
    with pytest.raises(ValueError):
        ReviewConfig(depth="ultra").depth_preset()


# ---------------------------------------------------------------- theorems ---

THEOREM_DOC = """# Math paper

## 1. Setup

Let n be the number of agents. Each agent i has weight w_i > 0. The total
mass is M = sum_i w_i.

Definition 1 (Widget). A widget is a pair (x, y) with x positive.

## 2. Results

Theorem 1. All widgets are stable.

Proof. Let (x, y) be a widget. Since x is positive, stability follows from
the definition of M. QED.

Lemma 2. Stability implies uniqueness.

Proof. Immediate.
"""


def test_extract_theorems_pairs_proofs():
    doc = document_from_markdown(THEOREM_DOC, source_format="md")
    doc.theorems = extract_theorems(doc)
    thm = [t for t in doc.theorems if t.kind == "theorem"]
    assert len(thm) == 1
    assert thm[0].label == "Theorem 1"
    assert thm[0].proof and "Since x is positive" in thm[0].proof
    lemma = [t for t in doc.theorems if t.kind == "lemma"]
    assert len(lemma) == 1
    assert "Immediate" in lemma[0].proof


def test_definitions_pack_includes_relevant_defs():
    doc = document_from_markdown(THEOREM_DOC, source_format="md")
    doc.theorems = extract_theorems(doc)
    thm = [t for t in doc.theorems if t.kind == "theorem"][0]
    pack = build_definitions_pack(doc, thm)
    # the theorem's symbols (x, y, widget) pull the Widget definition
    assert "Widget" in pack
    assert len(pack) < 4000  # compact


def test_claim_definitions_for_matches_theorem():
    doc = document_from_markdown(THEOREM_DOC, source_format="md")
    doc.theorems = extract_theorems(doc)
    claim = {
        "type": "theorem_proof",
        "anchor": "Theorem 1. All widgets are stable.",
        "components": {"statement": "Theorem 1. All widgets are stable."},
    }
    pack = _definitions_for(doc, claim)
    assert "widget" in pack.lower()


# ------------------------------------------------------------------ tables ---

TABLE_DOC = """# Empirical paper

## Results

We estimate the effect to be 0.25.

| var | coef | se |
|---|---|---|
| treatment | 0.25 | 0.10 |
| control | -0.02 | 0.08 |

Table 1: Treatment effects by specification.
"""


def test_extract_tables_markdown():
    doc = document_from_markdown(TABLE_DOC, source_format="md")
    tables = extract_tables(doc)
    assert len(tables) == 1
    assert "treatment" in tables[0].markdown
    assert tables[0].caption and "Table 1" in tables[0].caption


def test_find_mentions_figure_table():
    doc = document_from_markdown(TABLE_DOC, source_format="md")
    mentions = find_mentions(doc, "table", "tbl_p1_1")
    assert any("Table 1" in m or "estimate" in m for m in mentions)


# --------------------------------------------------------- dispatch & gates ---


def test_claim_type_role_mapping():
    assert CLAIM_TYPE_ROLES["theorem_proof"].value == "strong"
    assert CLAIM_TYPE_ROLES["numerical"].value == "small"
    assert CLAIM_TYPE_ROLES["cross_reference"].value == "small"


async def test_verify_claims_dispatch(test_config, sample_doc):
    """Claims route to the right role and comments inherit claim anchors."""
    from open_referee.config import ModelRole

    pipeline = ReviewPipeline.__new__(ReviewPipeline)
    pipeline.cfg = test_config
    pool = RolePool.build(test_config, {ModelRole.STRONG, ModelRole.SMALL})
    strong = pool.providers[ModelRole.STRONG]
    small = pool.providers[ModelRole.SMALL]

    claims = [
        {
            "id": "cl01",
            "type": "theorem_proof",
            "importance": "central",
            "components": {
                "statement": "Theorem 1. Every widget network is stable.",
                "proof": "Trivial.",
            },
            "anchor": "Theorem 1. Every widget network is stable.",
            "section": "3. Main result",
        },
        {
            "id": "cl02",
            "type": "cross_reference",
            "importance": "supporting",
            "components": {"passage": "Table 1 reports the outcomes.", "target": "Table 1"},
            "anchor": "Table 1 reports the outcomes.",
            "section": "3. Main result",
        },
    ]
    # standard preset verifies twice: the finding reproduces -> confirmed
    for _ in range(2):
        strong.queue(
            json.dumps(
                {
                    "comments": [
                        {
                            "title": "Proof gap",
                            "quote": "stable",
                            "message": "no proof",
                            "score": 0.8,
                            "category": "math",
                        }
                    ]
                }
            )
        )
    small.queue(json.dumps({"comments": []}))
    small.queue(json.dumps({"comments": []}))  # pass 2 for the small-model claim

    emitted = []

    async def _emit(*a, **k):
        emitted.append(a)

    pipeline._emit = _emit

    comments = await verify_claims(pipeline, pool, sample_doc, claims, test_config)
    assert len(comments) == 1
    assert comments[0]["claim_id"] == "cl01"
    assert comments[0]["provenance"] == "claim_verify 2-of-2"
    # anchor inherited from claim
    assert "Theorem 1" in comments[0]["paragraph_anchor"]
    assert len(strong.calls) == 2 and len(small.calls) == 2  # 2 passes per claim
    await pool.close()


async def test_defense_round_upholds_and_dismisses(test_config, sample_doc):
    from open_referee.config import ModelRole
    from open_referee.pipeline.orchestrator import RolePool

    pool = RolePool.build(test_config, {ModelRole.STRONG, ModelRole.SMALL})
    strong = pool.providers[ModelRole.STRONG]
    small = pool.providers[ModelRole.SMALL]

    pipeline = ReviewPipeline.__new__(ReviewPipeline)
    pipeline.cfg = test_config

    async def _emit(*a, **k):
        return None

    pipeline._emit = _emit

    candidates = [
        {
            "title": "Real issue",
            "paragraph_anchor": "Theorem 1. Every widget network is stable.",
            "quote": "Every widget network is stable",
            "message": "unproven",
            "score": 0.8,
            "category": "math",
        },
        {
            "title": "Fabricated",
            "paragraph_anchor": "Widget networks are everywhere.",
            "quote": "nowhere in text zzz",
            "message": "made up",
            "score": 0.7,
            "category": "clarity",
        },
    ]
    # defenses (small): first weak, second strong
    small.queue(json.dumps({"defense": "The proof exists elsewhere.", "defense_strength": "weak"}))
    small.queue(json.dumps({"defense": "Quote does not exist.", "defense_strength": "strong"}))
    # adjudications (strong): uphold first, dismiss second
    strong.queue(
        json.dumps(
            {
                "verdict": "upheld",
                "reason": "survives",
                "comment": {
                    "title": "Real issue",
                    "paragraph_anchor": "Theorem 1. Every widget network is stable.",
                    "quote": "Every widget network is stable",
                    "message": "unproven",
                    "score": 0.75,
                    "category": "math",
                },
            }
        )
    )
    strong.queue(json.dumps({"verdict": "dismissed", "reason": "fabricated"}))

    kept = await defense_round(pipeline, pool, sample_doc, candidates, test_config)
    assert len(kept) == 1
    assert kept[0]["title"] == "Real issue"
    assert kept[0]["score"] == 0.75  # adjudicator's recalibration wins
    await pool.close()


def test_passage_for_returns_context():
    doc = document_from_markdown(THEOREM_DOC, source_format="md")
    p = _passage_for(doc, "All widgets are stable")
    assert p and "widget" in p.lower()
    assert _passage_for(doc, "zzz not present") is None


@pytest.mark.asyncio
async def test_chunked_claim_inventory_merges_ids(test_config, monkeypatch):
    """Manuscripts beyond 70k chars are catalogued in two chunk calls whose
    claims merge under global ids; a failing chunk degrades, not kills."""
    from open_referee.config import ModelRole
    from open_referee.pipeline.claims import run_claim_inventory

    md = "# Big paper\n\n" + (
        "## Section one\n\n" + ("claim text one. " * 2400) + "\n\n"
        "## Section two\n\n" + ("claim text two. " * 2400) + "\n\n"
        "## Section three\n\n" + ("claim text three. " * 2400)
    )
    doc = document_from_markdown(md, source_format="md", title_hint="big")
    assert len(doc.full_text()) > 70_000

    pool = RolePool.build(test_config, {ModelRole.STRONG})
    strong = pool.providers[ModelRole.STRONG]
    strong.queue(
        json.dumps(
            {
                "claims": [
                    {
                        "id": "x1",
                        "type": "logical",
                        "components": {"s": "a"},
                        "anchor": "claim text one",
                        "section": "1",
                        "importance": "central",
                    },
                ]
            }
        )
    )
    strong.queue(
        json.dumps(
            {
                "claims": [
                    {"components": {"s": "b"}, "anchor": "claim text three"},
                ]
            }
        )
    )

    out = await run_claim_inventory(pool, doc, {"domain": "d"}, test_config)
    assert [c["id"] for c in out] == ["cl001", "cl002"]
    assert out[1]["anchor"] == "claim text three"
    assert len(strong.calls) == 2  # two chunk calls
    await pool.close()


# --------------------------------------------- decision-model defense gate --


@pytest.mark.asyncio
async def test_defense_round_dm_batches_with_passages(test_config, sample_doc):
    """The DM adjudication batches comments (no silent [:60k] context
    truncation) and carries the anchored manuscript passage next to each
    defense, so verdicts are checkable against what the paper says."""
    from open_referee.pipeline.claims import DM_BATCH_COMMENTS, _defense_round_dm
    from open_referee.providers.decisions import DecisionAnswer, DecisionResult

    dm_inputs: list[str] = []
    dm_qcounts: list[int] = []
    adjudication_users: list[str] = []

    class FakeDMPool:
        decider = object()  # presence routes defense_round to the DM path

        async def decide(self, input, questions, call_type="dm_defense"):
            assert call_type == "dm_defense"
            dm_inputs.append(input)
            dm_qcounts.append(len(questions))
            batch_no = len(dm_inputs)
            answers = []
            for q in questions:
                idx = int(q.name.split("_")[1])
                if q.name.startswith("verdict_"):
                    choice = "uphold"
                    if batch_no == 1 and idx == 2:
                        choice = "dismiss"
                    elif batch_no == 1 and idx == 3:
                        choice = "adjust"
                    answers.append(DecisionAnswer(type="choice", name=q.name, choice=choice))
                else:  # severity_<i>
                    answers.append(
                        DecisionAnswer(
                            type="score",
                            name=q.name,
                            probabilities=[{"label": "major", "probability": 0.9}],
                        )
                    )
            return DecisionResult(answers=answers)

        async def complete_json(self, role, system, user, call_type="other", **kw):
            if call_type == "defense":
                return {
                    "defense": "this is standard practice",
                    "defense_strength": "weak",
                    "concession": "",
                }
            assert call_type == "adjudication", f"unexpected {call_type}"
            adjudication_users.append(user)
            return {
                "comment": {
                    "title": "Adjusted",
                    "message": "reworked message",
                    "score": 0.4,
                    "category": "clarity",
                }
            }

    pipeline = ReviewPipeline.__new__(ReviewPipeline)
    pipeline.cfg = test_config

    async def _emit(*a, **k):
        return None

    pipeline._emit = _emit

    n = DM_BATCH_COMMENTS * 2 + 5  # 85 -> batches of 40/40/5
    candidates = [
        {
            "title": f"comment {i}",
            "paragraph_anchor": (
                "Theorem 1. Every widget network is stable." if i == 1 else "zzz nowhere"
            ),
            "quote": f"quote {i}",
            "message": f"message {i}",
            "score": 0.5,
            "category": "clarity",
            "provenance": "claim_verify 2-of-2",
        }
        for i in range(1, n + 1)
    ]
    kept = await _defense_round_dm(pipeline, FakeDMPool(), sample_doc, candidates, test_config)

    # three batches, every question count bounded, no input near the 60k cap
    assert dm_qcounts == [DM_BATCH_COMMENTS * 2, DM_BATCH_COMMENTS * 2, 10]
    assert all(len(x) < 60_001 for x in dm_inputs)
    # every listing entry carries a passage line; #1's is the real anchor text
    assert dm_inputs[0].count("passage:") == DM_BATCH_COMMENTS
    assert "Every widget network is stable" in dm_inputs[0]
    # dismissed batch-1 #2 gone, adjusted #3 rewritten, rest upheld
    assert len(kept) == n - 1
    assert all(c["title"] != "comment 2" for c in kept)
    adjusted = next(c for c in kept if c["title"] == "Adjusted")
    # adjust-merge keeps candidate keys the rewrite omitted -> provenance survives
    assert adjusted["provenance"] == "claim_verify 2-of-2"
    assert adjusted["quote"] == "quote 3"
    assert len(adjudication_users) == 1 and "standard practice" in adjudication_users[0]
    # severity recalibration applied (major -> 0.7)
    first = next(c for c in kept if c["title"] == "comment 1")
    assert first["score"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_defense_round_dm_batch_failure_upholds_that_batch(test_config, sample_doc):
    """A failed decide() call upholds only its own batch — the rest of the
    gate still runs (previously one monolithic failure nulled the gate)."""
    from open_referee.pipeline.claims import DM_BATCH_COMMENTS, _defense_round_dm
    from open_referee.providers.base import LLMError
    from open_referee.providers.decisions import DecisionAnswer, DecisionResult

    dm_calls = 0

    class FakeDMPool:
        decider = object()

        async def decide(self, input, questions, call_type="dm_defense"):
            nonlocal dm_calls
            dm_calls += 1
            if dm_calls == 2:
                raise LLMError("decision endpoint down")
            answers = [
                DecisionAnswer(type="choice", name=q.name, choice="uphold") for q in questions
            ]
            return DecisionResult(answers=answers)

        async def complete_json(self, role, system, user, call_type="other", **kw):
            return {"defense": "d", "defense_strength": "weak", "concession": ""}

    pipeline = ReviewPipeline.__new__(ReviewPipeline)
    pipeline.cfg = test_config

    async def _emit(*a, **k):
        return None

    pipeline._emit = _emit
    candidates = [
        {
            "title": f"c{i}",
            "paragraph_anchor": "zzz",
            "quote": f"q{i}",
            "message": "m",
            "score": 0.5,
            "category": "clarity",
        }
        for i in range(DM_BATCH_COMMENTS + 3)
    ]
    kept = await _defense_round_dm(pipeline, FakeDMPool(), sample_doc, candidates, test_config)
    assert dm_calls == 2
    assert len(kept) == len(candidates)  # batch 2 failed -> upheld, nothing lost
