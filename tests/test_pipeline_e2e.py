"""End-to-end pipeline run on a sample paper with scripted FakeProviders."""

from __future__ import annotations

import json

from conftest import overall_json, triage_json, validator_json
from open_referee.config import ModelRole
from open_referee.pipeline import Stage, StageStatus
from open_referee.pipeline.orchestrator import ReviewPipeline


def _script_comments(scope: str) -> str:
    return json.dumps(
        {
            "comments": [
                {
                    "title": f"{scope}: Theorem 1 lacks a proof",
                    "paragraph_anchor": "Theorem 1. Every widget network is stable.",
                    "quote": "Every widget network is stable",
                    "message": "State and prove the theorem or cite a source.",
                    "score": 0.8,
                    "category": "math",
                }
            ]
        }
    )


def _script_challenge(scope: str) -> str:
    return json.dumps(
        {
            "validated": [
                {
                    "title": f"{scope}: Theorem 1 lacks a proof",
                    "paragraph_anchor": "Theorem 1. Every widget network is stable.",
                    "quote": "Every widget network is stable",
                    "message": "State and prove the theorem or cite a source.",
                    "score": 0.75,
                    "category": "math",
                    "verdict": "kept",
                    "verdict_reason": "real gap",
                }
            ],
            "new_comments": [
                {
                    "title": f"{scope}: Overclaim in intro",
                    "paragraph_anchor": "Widget networks are everywhere.",
                    "quote": "Widget networks are everywhere",
                    "message": "Hedge or cite.",
                    "score": 0.45,
                    "category": "evidence",
                }
            ],
        }
    )


async def test_full_pipeline_e2e(test_config, sample_paper_file, monkeypatch, tmp_path):

    # isolate run storage + config dir
    home = tmp_path / "home"
    runs_dir = home / "runs"
    import open_referee.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_DIR", home)
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", home / "config.yaml")

    # stub the network-touching parts: no OpenAlex/Crossref/search in this test
    async def _fake_literature(self, pool, doc, triage, queries, key_citations, lit_docs):
        from open_referee.literature import ContextPack

        pack = ContextPack(paper_title=doc.title, field_hint="network science")
        pack.user_docs = []
        return pack

    async def _fake_scout(self, pool, title, key_citations, pack):
        return None

    async def _fake_bib(self, pool, doc, pack):
        return {"entries": [], "missing_key_references": []}

    monkeypatch.setattr(ReviewPipeline, "_literature", _fake_literature)
    monkeypatch.setattr(ReviewPipeline, "_scout", _fake_scout)
    monkeypatch.setattr(ReviewPipeline, "_bibliography", _fake_bib)

    pipeline = ReviewPipeline(test_config, run_id="e2e01")
    pool = await pipeline._ensure_pool()
    strong = pool.providers[ModelRole.STRONG]
    small = pool.providers[ModelRole.SMALL]
    # script the LLM responses in call order:
    # triage(strong) -> survey(small, may fail gracefully) -> verify(small x sections)
    # -> challenge(strong x sections) -> bibliography(small, stubbed)
    # -> meta(strong) -> validate(strong)
    doc = await __import__("asyncio").to_thread(
        __import__("open_referee.ingestion.readers", fromlist=["ingest_document"]).ingest_document,
        sample_paper_file,
    )
    sections = pipeline._reviewable_sections(doc)
    triage = json.loads(triage_json())
    lenses = pipeline._lenses_for(triage)
    n_verify = len(sections) * len(lenses)

    strong.queue(triage_json())
    # CLAIMS stage (strong)
    strong.queue(
        json.dumps(
            {
                "claims": [
                    {
                        "id": "cl01",
                        "type": "theorem_proof",
                        "importance": "central",
                        "components": {
                            "statement": "Theorem 1. Every widget network is stable.",
                            "proof": "(none given)",
                        },
                        "anchor": "Theorem 1. Every widget network is stable.",
                        "section": "3. Main result",
                    },
                ]
            }
        )
    )
    small.queue(
        json.dumps({"selected": [], "state_of_the_art_notes": "n/a", "missing_references": []})
    )
    # per-claim worker (small for cross_reference would be small; theorem_proof -> strong)
    strong.queue(_script_comments("claim"))
    for _ in range(n_verify):
        small.queue(_script_comments("lens"))
    # whole-paper coherence pass (strong) happens AFTER section lenses
    strong.queue(_script_comments("whole-paper"))
    for title, _ in sections:
        strong.queue(_script_challenge(title))
    # whole-paper challenge (strong) happens AFTER section challengers
    strong.queue(_script_challenge("whole-paper"))
    # defense gates: each surviving comment consumes small(defense)+strong(adjudication).
    # count comments produced by the scripted challenge responses:
    # per section: 1 validated + 1 new; whole-paper: 1 validated; claim: 1.
    n_sections = len(sections)
    n_defended = n_sections * 2 + 1 + 1  # + wp + claim comment
    for i in range(n_defended):
        small.queue(json.dumps({"defense": "weak", "defense_strength": "weak"}))
        strong.queue(
            json.dumps(
                {
                    "verdict": "upheld",
                    "reason": "ok",
                    "comment": {
                        "title": f"upheld-{i}",
                        "paragraph_anchor": "Theorem 1. Every widget network is stable.",
                        "quote": "Every widget network is stable",
                        "message": "State and prove the theorem or cite a source.",
                        "score": 0.75,
                        "category": "math",
                    },
                }
            )
        )
    strong.queue(overall_json())
    strong.queue(validator_json())

    events: list = []
    q = pipeline.subscribe()

    import asyncio

    async def collect():
        while True:
            ev = await q.get()
            events.append(ev)
            if ev.stage is Stage.ASSEMBLE and ev.status in (StageStatus.DONE, StageStatus.FAILED):
                return

    collect_task = asyncio.create_task(collect())
    report = await pipeline.run(sample_paper_file)
    await asyncio.wait_for(collect_task, timeout=5)

    # report assertions
    assert report.paper_title.startswith("On the Stability")
    assert report.overall_feedback and "validated" in report.overall_feedback
    assert report.comments, "expected anchored comments"
    top = report.sorted_comments()[0]
    assert top.score >= 0.7
    assert top.block_id is not None, "math comment should anchor to Theorem 1 block"
    assert top.anchor_confidence >= 0.6
    assert report.validator_notes  # validator pass contributed notes
    assert report.model_roles["strong"] == "fake/strong-fake"
    assert report.usage.total_input_tokens > 0

    # persistence assertions
    assert (runs_dir / "e2e01" / "report.json").exists()
    assert (runs_dir / "e2e01" / "report.md").exists()
    state = __import__("open_referee.pipeline.state", fromlist=["RunState"]).RunState.load(
        "e2e01", runs_dir=runs_dir
    )
    assert state.status == "completed"
    assert state.stage_status(Stage.VALIDATE) is StageStatus.DONE

    # every stage emitted events
    stages_seen = {ev.stage for ev in events}
    assert {
        Stage.INGEST,
        Stage.TRIAGE,
        Stage.VERIFY,
        Stage.CHALLENGE,
        Stage.META,
        Stage.VALIDATE,
    } <= stages_seen
