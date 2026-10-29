"""Copyedit annex: verbatim language-fix pass, deterministic quote filter,
budget plumbing, and rendering."""

from __future__ import annotations

import pytest

from open_referee.ingestion.readers import document_from_markdown
from open_referee.pipeline.orchestrator import ReviewPipeline
from open_referee.report.models import CopyeditItem

DOC_MD = """# Copyedit target

## 1 Introduction

Widget networks are everywere in modern sciance. The framework is applied
consistently across all the sections that follow, and the datas supports
the claim.
"""

DOC = document_from_markdown(DOC_MD, source_format="md", title_hint="t")


class StubPool:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[str] = []

    async def complete_json(self, role, system, user, call_type="other", max_tokens=None):
        self.calls.append(call_type)
        assert call_type == "copyedit"
        return self.payload


class StubPipeline:
    def plan_knobs(self):
        from open_referee.pipeline.budget import PlanKnobs

        return PlanKnobs(copyedit=True)

    def budget_allows(self, ct):
        return True

    def _note_gate_stop(self, ct):
        pass

    def _manuscript_trunc(self, doc):
        return doc.full_text()[:120_000]

    async def _emit(self, *a, **k):
        return None


@pytest.mark.asyncio
async def test_copyedit_keeps_locatable_quotes_and_drops_hallucinated():
    payload = {
        "items": [
            {
                "quote": "Widget networks are everywere in modern sciance.",
                "suggestion": "Widget networks are everywhere in modern science.",
                "kind": "typo",
            },
            {
                "quote": "This sentence does not exist in the manuscript at all.",
                "suggestion": "anything",
                "kind": "grammar",
            },
            {
                "quote": "the datas supports the claim",
                "suggestion": "the data support the claim",
                "kind": "grammar",
            },
            {"quote": "", "suggestion": "no quote", "kind": "typo"},
            {
                "quote": "framework is applied",
                "suggestion": "framework is utilized",
                "kind": "synonym",
            },  # invalid kind -> dropped
        ]
    }
    items = await ReviewPipeline._copyedit(StubPipeline(), StubPool(payload), DOC)
    assert len(items) == 2
    assert items[0]["quote"].startswith("Widget networks are everywere")
    assert items[0]["kind"] == "typo"
    assert items[1]["suggestion"] == "the data support the claim"
    assert items[1]["section"]  # section attached from the anchor block


@pytest.mark.asyncio
async def test_copyedit_knob_off_skips_call():
    class Off(StubPipeline):
        def plan_knobs(self):
            from open_referee.pipeline.budget import PlanKnobs

            return PlanKnobs(copyedit=False)

    pool = StubPool({"items": []})
    items = await ReviewPipeline._copyedit(Off(), pool, DOC)
    assert items == []
    assert pool.calls == []


def test_budget_counts_and_unit():
    import sys

    sys.path.insert(0, "tests")
    from open_referee.pipeline.budget import DocumentStats, PlanKnobs
    from test_budget import make_harness

    stats = DocumentStats(chars=60_000, n_sections=6, n_claims=20, n_references=10)
    h, _ = make_harness(budget=50.0, stats=stats)
    assert h._call_counts(PlanKnobs())["copyedit"] == 1
    assert h._call_counts(PlanKnobs(copyedit=False)).get("copyedit") is None
    role, _, out = h._base_unit("copyedit")
    assert role == "small" and out == 5000
    # starved floor skips the annex
    assert h.min_viable_knobs().copyedit is False


def test_markdown_annex_renders_between_notes_and_budget():
    from tests.test_render_meta import _review_report

    from open_referee.report.render import render_markdown

    report = _review_report()
    report.validator_notes = "validated."
    report.copyedit_items = [
        CopyeditItem(
            quote="everywere in sciance",
            suggestion="everywhere in science",
            kind="typo",
            section="1 Introduction",
        ),
    ]
    md = render_markdown(report)
    assert "## Suggested language fixes (1)" in md
    assert "| 1 | typo | 1 Introduction | everywere in sciance | everywhere in science |" in md
    assert md.index("## Validator notes") < md.index("## Suggested language fixes")
    assert md.index("## Suggested language fixes") < md.index("## Build metadata")
    # empty annex renders nothing
    report.copyedit_items = []
    assert "Suggested language fixes" not in render_markdown(report)
