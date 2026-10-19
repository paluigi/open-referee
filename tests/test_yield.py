"""Yield & synthesis: checker findings join the comment stream, the synthesis
essay replaces the merger's overall feedback, dm gate screens all comments in
chunks, and deep plans run a second defense round."""

from __future__ import annotations

import pytest

from open_referee.config import Config
from open_referee.ingestion.readers import document_from_markdown
from open_referee.pipeline.claims import defense_round
from open_referee.pipeline.orchestrator import ReviewPipeline
from open_referee.pipeline.state import RunState


def _pipeline_with_findings(tmp_path) -> ReviewPipeline:
    cfg = Config()
    pipeline = ReviewPipeline.__new__(ReviewPipeline)
    pipeline.cfg = cfg
    pipeline.state = RunState(run_id="y1", paper_path="x")
    pipeline.state.checker_findings = [
        {
            "check": "consistency",
            "title": "Conflicting values for mae (0.313 vs 0.418)",
            "message": "The quantity described as mae is reported differently.",
            "quote": "overall MAE of 0.313",
            "severity": 0.7,
            "category": "consistency",
            "provenance": "deterministic",
        },
        {
            "check": "doi-resolution",
            "title": "Reference not resolvable",
            "message": "No Crossref or OpenAlex record matches.",
            "quote": "Berki, A. (2025). Machine learning nowcasting.",
            "severity": 0.4,
            "category": "reference",
            "provenance": "deterministic",
        },
    ]
    return pipeline


class StubPool:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []  # (call_type, user)

    async def complete_json(self, role, system, user, call_type="other", max_tokens=None):
        self.calls.append((call_type, user))
        if call_type == "meta":
            return {"paper_summary": "s", "overall_feedback": "## Merge", "comments": []}
        if call_type == "synthesis":
            return {"overall_feedback": "## Synth essay"}
        raise AssertionError(f"unexpected call {call_type}")


def test_checker_findings_become_candidate_comments(tmp_path):
    pipeline = _pipeline_with_findings(tmp_path)
    comments = pipeline._checker_comments()
    assert len(comments) == 2
    assert comments[0]["provenance"] == "deterministic"
    assert comments[0]["score"] == 0.7
    assert comments[0]["paragraph_anchor"].startswith("overall MAE")
    assert comments[1]["category"] == "reference"


@pytest.mark.asyncio
async def test_meta_merges_checker_comments_and_replaces_overall(tmp_path, monkeypatch):
    pipeline = _pipeline_with_findings(tmp_path)

    class Knobs:
        synthesis = True
        output_budget = "default"

    pipeline.plan_knobs = lambda: Knobs()  # type: ignore[method-assign]
    pipeline.budget_allows = lambda ct: True  # type: ignore[method-assign]
    pipeline._venue_block = ""
    pipeline._synthesis_spy = None

    doc = document_from_markdown("# T\n\nBody.", source_format="md", title_hint="t")

    class Pack:
        field_hint = "field"

    pool = StubPool()
    draft = await ReviewPipeline._meta(
        pipeline, pool, doc, {"domain": "d"}, [{"title": "LLM finding"}], {"entries": []}, Pack()
    )
    call_types = [c for c, _ in pool.calls]
    assert call_types == ["meta", "synthesis"]
    meta_user = pool.calls[0][1]
    assert "Conflicting values for mae" in meta_user  # checker comment merged in
    assert "LLM finding" in meta_user
    assert draft["overall_feedback"] == "## Synth essay"


@pytest.mark.asyncio
async def test_meta_skips_synthesis_when_knob_off(tmp_path):
    pipeline = _pipeline_with_findings(tmp_path)

    class Knobs:
        synthesis = False
        output_budget = "default"

    pipeline.plan_knobs = lambda: Knobs()  # type: ignore[method-assign]
    pipeline._venue_block = ""
    doc = document_from_markdown("# T\n\nBody.", source_format="md", title_hint="t")

    class Pack:
        field_hint = ""

    pool = StubPool()
    draft = await ReviewPipeline._meta(
        pipeline, pool, doc, {}, [], {"entries": []}, Pack()
    )
    assert [c for c, _ in pool.calls] == ["meta"]
    assert draft["overall_feedback"] == "## Merge"


@pytest.mark.asyncio
async def test_dm_validator_gate_chunks_all_comments():
    pipeline = ReviewPipeline.__new__(ReviewPipeline)

    async def emit(*a, **k):
        return None

    pipeline._emit = emit  # type: ignore[method-assign]
    doc = document_from_markdown("# T\n\nBody.", source_format="md", title_hint="t")

    calls: list[list] = []

    class Decider:
        async def decide(self, input, questions, call_type="dm"):
            calls.append(questions)
            from open_referee.providers.decisions import DecisionAnswer, DecisionResult

            first_chunk = len(calls) == 1
            answers = []
            for q in questions:
                drop = first_chunk and q.name == "comment_1"
                answers.append(
                    DecisionAnswer(
                        type="predicate",
                        name=q.name,
                        probability=0.1 if drop else 0.9,
                    )
                )
            return DecisionResult(answers=answers)

    decider = Decider()

    class Pool:
        pass

    Pool.decide = decider.decide  # bound method shim matching RolePool surface
    pool = Pool()
    pool.decider = decider

    comments = [
        {"title": f"c{i}", "quote": "q", "paragraph_anchor": "a"} for i in range(75)
    ]
    kept = await ReviewPipeline._dm_validator_gate(pipeline, pool, doc, comments)
    # two chunked calls, first comment of chunk 1 dropped, everything else kept
    assert len(calls) == 2
    assert len(calls[0]) == 60 and len(calls[1]) == 15
    assert len(kept) == 74


@pytest.mark.asyncio
async def test_second_defense_round_runs_on_survivors(monkeypatch):
    calls: list[list] = []

    async def fake_dm_round(pipeline, pool, doc, candidates, cfg):
        calls.append(list(candidates))
        # drop the first comment each round
        return candidates[1:]

    import open_referee.pipeline.claims as claims_mod

    monkeypatch.setattr(claims_mod, "_defense_round_dm", fake_dm_round)
    pipeline = ReviewPipeline.__new__(ReviewPipeline)

    class Knobs:
        defense_round = True
        defense_second_round = True

    pipeline.plan_knobs = lambda: Knobs()  # type: ignore[method-assign]

    async def emit(*a, **k):
        return None

    pipeline._emit = emit  # type: ignore[method-assign]

    class Pool:
        decider = object()

    out = await defense_round(pipeline, Pool(), None, [{"t": 1}, {"t": 2}, {"t": 3}], None)
    assert calls == [[{"t": 1}, {"t": 2}, {"t": 3}], [{"t": 2}, {"t": 3}]]
    assert out == [{"t": 3}]
