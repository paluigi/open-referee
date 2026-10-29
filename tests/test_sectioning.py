"""Sectioning: HTML-anchored markdown headings and heading-heuristic fallback."""

from __future__ import annotations

from open_referee.ingestion.readers import document_from_markdown
from open_referee.pipeline.orchestrator import _split_by_headings

ANCHORED = (
    """# Title

Author One, Author Two

Abstract

Filler abstract text. """
    + "word " * 120
    + """

<a id="introduction"></a>## 1 Introduction

First section body. """
    + "sentence " * 200
    + """

<a id="conclusions"></a>## 5 Conclusions

Closing body. """
    + "sentence " * 200
)
PLAIN = (
    """Paper title line

Some front matter. """
    + "text " * 200
    + """

1. Introduction

Intro body. """
    + "sentence " * 200
    + """

2.3 Methods and data

Methods body. """
    + "sentence " * 200
    + """

CONCLUSIONS

Conclusions body. """
    + "sentence " * 200
)


def test_anchored_markdown_headings_produce_sections():
    doc = document_from_markdown(ANCHORED, source_format="md", title_hint="t")
    titles = [s.title for s in doc.sections]
    assert "1 Introduction" in titles
    assert "5 Conclusions" in titles
    # anchors must not leak into block text
    assert all("<a id=" not in b.text for b in doc.blocks)


def test_split_by_headings_numbered_and_caps():
    out = _split_by_headings(PLAIN)
    titles = [t for t, _ in out]
    assert "1. Introduction" in titles
    assert "2.3 Methods and data" in titles
    assert "Conclusions" in titles
    assert all(len(text) > 200 for _, text in out)


def test_split_by_headings_merges_small_chunks_and_caps_authors_ignored():
    text = (
        "ANDREA GALEOTTI, BENJAMIN GOLUB, SANJEEV GOYAL\n\n"
        "Abstract\n\n" + "abstract text. " * 80 + "\n\n"
        "## 1 Introduction\n\n" + "body. " * 200 + "\n\n"
        "## 2 Short\n\ntiny\n\n"
        "## 3 More\n\n" + "body. " * 200
    )
    out = _split_by_headings(text)
    titles = [t for t, _ in out]
    # author-list caps line must NOT become a section boundary
    assert not any("GALEOTTI" in t for t in titles)
    # the tiny "## 2 Short" chunk merges into a neighbour
    assert "2 Short" not in titles
    assert any(t == "1 Introduction" for t in titles)


def test_split_by_headings_caps_sections_without_commas():
    text = "INTRODUCTION\n\n" + "body " * 200 + "\n\nRESULTS\n\n" + "body " * 200
    out = _split_by_headings(text)
    assert [t for t, _ in out] == ["INTRODUCTION", "RESULTS"]


def test_split_by_headings_empty_when_no_headings():
    assert _split_by_headings("just one long paragraph " * 500) == []


def test_reviewable_sections_uses_fallback(monkeypatch):
    from open_referee.pipeline.orchestrator import ReviewPipeline

    doc = document_from_markdown(PLAIN, source_format="md", title_hint="plain")
    # no markdown headings -> parser yields a single degenerate title section
    assert len(doc.sections) <= 1
    pipeline = ReviewPipeline.__new__(ReviewPipeline)
    out = pipeline._reviewable_sections(doc)
    assert len(out) > 1
    assert all(len(text) > 200 for _, text in out)
