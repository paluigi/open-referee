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
    assert deep.max_claims == 150 and deep.max_artifacts == 25
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

    emitted = []

    async def _emit(*a, **k):
        emitted.append(a)

    pipeline._emit = _emit

    comments = await verify_claims(pipeline, pool, sample_doc, claims, test_config)
    assert len(comments) == 1
    assert comments[0]["claim_id"] == "cl01"
    # anchor inherited from claim
    assert "Theorem 1" in comments[0]["paragraph_anchor"]
    assert len(strong.calls) == 1 and len(small.calls) == 1  # role dispatch correct
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
