"""Tests for the referee prompt system: lens selection, citation extraction,
prompt content guarantees."""

from __future__ import annotations

import json

from conftest import triage_json
from open_referee.config import ModelRole
from open_referee.ingestion.readers import document_from_markdown, ingest_document
from open_referee.pipeline import prompts
from open_referee.pipeline.orchestrator import ReviewPipeline, _extract_intext_citations

# ------------------------------------------------------------ lens selection --


def _pipeline():
    return ReviewPipeline.__new__(ReviewPipeline)  # no cfg needed for _lenses_for


def test_lenses_theory_paper():
    lenses = _pipeline()._lenses_for(
        {
            "mathematical_density": "heavy",
            "statistical_content": "none",
            "methodological_style": "theory",
        }
    )
    names = [n for n, _ in lenses]
    assert "math" in names
    assert "empirical" not in names
    assert "prose" in names and "lit" in names


def test_lenses_empirical_paper():
    lenses = _pipeline()._lenses_for(
        {
            "mathematical_density": "light",
            "statistical_content": "causal_inference",
            "methodological_style": "empirical",
        }
    )
    names = [n for n, _ in lenses]
    assert "math" in names  # light density still includes math
    assert "empirical" in names
    assert "prose" in names and "lit" in names


def test_lenses_pure_prose_paper():
    lenses = _pipeline()._lenses_for(
        {
            "mathematical_density": "none",
            "statistical_content": "none",
            "methodological_style": "other",
        }
    )
    names = [n for n, _ in lenses]
    assert names == ["prose", "lit"]


def test_lenses_missing_triage_fields_safe():
    lenses = _pipeline()._lenses_for({})
    names = [n for n, _ in lenses]
    assert "prose" in names and "lit" in names


def test_all_lens_prompts_carry_persona_and_schema():
    for name, system in [
        ("math", prompts.MATH_VERIFIER_SYSTEM),
        ("empirical", prompts.EMPIRICAL_VERIFIER_SYSTEM),
        ("prose", prompts.PROSE_VERIFIER_SYSTEM),
        ("lit", prompts.LIT_VERIFIER_SYSTEM),
    ]:
        assert "top academic journal" in system, name
        assert "STRICT JSON" in system, name
        assert "paragraph_anchor" in system, name
        assert "severity" in system.lower(), name


# -------------------------------------------------------- citation extraction --

DOC = """# Paper

Smith (2001) proved that classical widgets converge. This extends the
framework of (Doe and Roe, 1999; Alpha, 2005) to noisy settings.

Numerical evidence supports the claim [1], and robustness checks appear in
[2, 3]. The estimator follows Jones2020 closely.
"""


def test_intext_extraction_author_year():
    doc = document_from_markdown(DOC, source_format="md")
    out = _extract_intext_citations(doc)
    assert "Smith (2001)" in out
    assert "(Doe and Roe, 1999; Alpha, 2005)" in out
    # claim context is attached
    assert "classical widgets converge" in out


def test_intext_extraction_numeric_and_natbib():
    doc = document_from_markdown(DOC, source_format="md")
    out = _extract_intext_citations(doc)
    assert "[1]" in out
    assert "[2, 3]" in out
    assert "Jones2020" in out


def test_intext_extraction_no_citations():
    doc = document_from_markdown("# T\n\nNo citations at all here.", source_format="md")
    out = _extract_intext_citations(doc)
    assert "no in-text citations" in out


def test_intext_extraction_caps_at_limit():
    text = "# T\n\n" + "\n\n".join(f"Claim number {i} (Author{i}, 2001) holds." for i in range(200))
    doc = document_from_markdown(text, source_format="md")
    out = _extract_intext_citations(doc)
    n = out.count("::")
    assert n <= 60


# --------------------------------------------------------------- prompt design --


def test_persona_present_in_all_stage_prompts():
    for p in [
        prompts.TRIAGE_SYSTEM,
        prompts.CHALLENGER_SYSTEM,
        prompts.BIBLIOGRAPHY_SYSTEM,
        prompts.META_SYSTEM,
        prompts.VALIDATOR_SYSTEM,
        prompts.FIGURE_SYSTEM,
    ]:
        assert "referee for a top academic journal" in p or "top academic journal" in p


def test_severity_rubric_in_comment_producers():
    for p in [
        prompts.MATH_VERIFIER_SYSTEM,
        prompts.EMPIRICAL_VERIFIER_SYSTEM,
        prompts.PROSE_VERIFIER_SYSTEM,
        prompts.LIT_VERIFIER_SYSTEM,
        prompts.FIGURE_SYSTEM,
        prompts.CHALLENGER_SYSTEM,
    ]:
        assert "0.80-1.00" in p


def test_bibliography_prompt_covers_existence_and_quotation():
    p = prompts.BIBLIOGRAPHY_SYSTEM
    assert "EXISTENCE" in p
    assert "QUOTATION CONSISTENCY" in p
    assert "in_text_issues" in p


def test_math_lens_covers_statements_proofs_and_cases():
    p = prompts.MATH_VERIFIER_SYSTEM
    for needle in [
        "statement",
        "proof",
        "without loss of generality",
        "boundary",
        "notation",
        "circular",
    ]:
        assert needle.lower() in p.lower(), needle


def test_empirical_lens_covers_econometric_practice():
    p = prompts.EMPIRICAL_VERIFIER_SYSTEM
    for needle in [
        "identification",
        "parallel trends",
        "instrument",
        "clustering",
        "multiple hypothesis",
        "robustness",
        "external validity",
    ]:
        assert needle.lower() in p.lower(), needle


def test_meta_prompt_requires_recommendation_and_strengths():
    p = prompts.META_SYSTEM
    assert "strengths" in p.lower()
    assert "recommendation" in p.lower()
    assert "major revision" in p


def test_validator_prompt_drops_unsubstantiated():
    p = prompts.VALIDATOR_SYSTEM
    assert "DROP" in p
    assert "fabricated" in p


# ------------------------------------------------------- whole-paper passes --


def test_whole_paper_verifier_prompt_scope():
    p = prompts.WHOLE_PAPER_VERIFIER_SYSTEM
    assert "ENTIRE manuscript" in p
    # covers the cross-section dimensions the user asked about
    for needle in [
        "abstract",
        "contribution",
        "notation",
        "results",
        "conclusion",
        "cross-section",
    ]:
        assert needle.lower() in p.lower(), needle
    # persona + schema
    assert "top academic journal" in p
    assert "paragraph_anchor" in p


def test_whole_paper_challenger_prompt_validates_and_hunts():
    p = prompts.WHOLE_PAPER_CHALLENGER_SYSTEM
    assert "validate" in p.lower()
    assert "GLOBAL" in p
    assert "overclaiming" in p
    assert "verdict" in p


async def test_whole_paper_passes_run_in_pipeline(
    test_config, sample_paper_file, monkeypatch, tmp_path
):
    """Whole-paper passes fire and their comments reach the final report."""
    import asyncio

    import open_referee.config as cfg_mod

    home = tmp_path / "home"
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_DIR", home)
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", home / "config.yaml")

    async def _fake_lit(self, pool, doc, triage, queries, key_citations, lit_docs):
        from open_referee.literature import ContextPack

        return ContextPack(paper_title=doc.title, field_hint="t")

    async def _fake_scout(self, pool, title, key_citations, pack):
        return None

    async def _fake_bib(self, pool, doc, pack):
        return {"entries": [], "in_text_issues": [], "missing_key_references": []}

    monkeypatch.setattr(ReviewPipeline, "_literature", _fake_lit)
    monkeypatch.setattr(ReviewPipeline, "_scout", _fake_scout)
    monkeypatch.setattr(ReviewPipeline, "_bibliography", _fake_bib)
    test_config.llm.max_retries = 0

    pipeline = ReviewPipeline(test_config, run_id="wp01")
    pool = await pipeline._ensure_pool()
    strong = pool.providers[ModelRole.STRONG]
    small = pool.providers[ModelRole.SMALL]

    doc = await asyncio.to_thread(ingest_document, sample_paper_file)
    sections = pipeline._reviewable_sections(doc)
    triage = json.loads(triage_json())
    n_lens = len(pipeline._lenses_for(triage))

    wp = {
        "title": "WP: abstract overclaims",
        "paragraph_anchor": (
            "The abstract promises a welfare theorem that the body never establishes."
        ),
        "quote": "The abstract promises a welfare theorem",
        "message": "Abstract claims a proof the body never delivers.",
        "score": 0.7,
        "category": "consistency",
    }

    strong.queue(triage_json())
    strong.queue(json.dumps({"claims": []}))
    small.queue(
        json.dumps({"selected": [], "state_of_the_art_notes": "", "missing_references": []})
    )
    for _ in range(len(sections) * n_lens):
        small.queue(json.dumps({"comments": []}))
    strong.queue(json.dumps({"comments": [wp]}))  # whole-paper verifier
    for _ in sections:
        strong.queue(json.dumps({"validated": [], "new_comments": []}))
    strong.queue(json.dumps({"validated": [dict(wp, verdict="kept")], "new_comments": []}))
    small.queue(json.dumps({"defense": "d", "defense_strength": "weak"}))
    strong.queue(json.dumps({"verdict": "upheld", "comment": wp}))
    strong.queue(json.dumps({"paper_summary": "s", "overall_feedback": "## F", "comments": [wp]}))
    strong.queue(
        json.dumps(
            {
                "comments": [wp],
                "overall_feedback": "## F",
                "paper_summary": "s",
                "validator_notes": "n",
            }
        )
    )

    report = await pipeline.run(sample_paper_file)
    titles = [c.title for c in report.comments]
    assert any(t.startswith("WP:") for t in titles), f"whole-paper comment missing: {titles}"
