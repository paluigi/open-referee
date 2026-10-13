"""Shared fixtures: FakeProvider wiring and sample documents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from open_referee.config import Config, LLMConfig, ProviderConfig, RoleConfig
from open_referee.ingestion.readers import document_from_markdown

SAMPLE_PAPER = """# On the Stability of Widget Networks

## 1. Introduction

Widget networks are everywhere. We prove that all widgets are stable
under mild assumptions, extending the classical results of Smith (2001).

## 2. Model

Let G = (V, E) be a widget graph. Each widget has mass m_i > 0.

## 3. Main result

Theorem 1. Every widget network is stable. Table 1 reports the outcomes.

## References

[1] Smith, J. (2001). Classical widgets. Journal of Widgets, 1(2), 3-14.
[2] Doe, A. (2019). Modern widgets. Widgets Today, 5(1), 1-20.
"""


@pytest.fixture
def sample_doc():
    return document_from_markdown(SAMPLE_PAPER, source_format="md", title_hint="sample")


@pytest.fixture
def sample_paper_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample_paper.md"
    p.write_text(SAMPLE_PAPER)
    return p


def triage_json() -> str:
    return json.dumps(
        {
            "domain": "network science",
            "language": "en",
            "paper_type": "theory",
            "main_claims": ["All widgets are stable"],
            "contribution": "A stability theorem for widget networks.",
            "methodological_style": "theory",
            "key_sections": ["3. Main result"],
            "mathematical_density": "moderate",
            "statistical_content": "none",
            "search_queries": ["widget network stability"],
            "key_citations": ["Smith, 2001: Classical widgets"],
            "novelty_claims": [],
        }
    )


def comments_json(n: int = 2, anchored_quote: str | None = None) -> str:
    comments = [
        {
            "title": "Theorem 1 lacks a proof",
            "paragraph_anchor": "Theorem 1. Every widget network is stable.",
            "quote": anchored_quote or "Every widget network is stable",
            "message": "The theorem is stated but Section 3 provides no proof or citation.",
            "score": 0.8,
            "category": "math",
        }
    ]
    if n > 1:
        comments.append(
            {
                "title": "Claim in introduction overreaches",
                "paragraph_anchor": "Widget networks are everywhere.",
                "quote": "Widget networks are everywhere",
                "message": "This sweeping claim needs a citation or hedging.",
                "score": 0.4,
                "category": "evidence",
            }
        )
    return json.dumps({"comments": comments})


def overall_json() -> str:
    return json.dumps(
        {
            "paper_summary": "The paper claims all widget networks are stable.",
            "overall_feedback": (
                "## Central claim\n\nTheorem 1 is unproven; Table 1 is absent."
                "\n\n## Recommendation\n\nMajor revision."
            ),
            "comments": comments_payload(),
        }
    )


def comments_payload() -> list[dict]:
    return [
        {
            "title": "Theorem 1 lacks a proof",
            "paragraph_anchor": "Theorem 1. Every widget network is stable.",
            "quote": "Every widget network is stable",
            "message": "Provide a proof or cite one.",
            "score": 0.75,
            "category": "math",
        }
    ]


def validator_json() -> str:
    return json.dumps(
        {
            "comments": comments_payload(),
            "overall_feedback": (
                "## Central claim\n\nTheorem 1 is unproven (validated)."
                "\n\n## Recommendation\n\nMajor revision."
            ),
            "paper_summary": "Validated summary: the paper claims widget-network stability.",
            "validator_notes": "Dropped one unverifiable comment; recalibrated severity.",
        }
    )


@pytest.fixture
def test_config() -> Config:
    cfg = Config()
    cfg.llm = LLMConfig(
        providers={"fake": ProviderConfig(type="fake")},  # type: ignore[arg-type]
        roles={
            "strong": RoleConfig(provider="fake", model="strong-fake"),
            "small": RoleConfig(provider="fake", model="small-fake"),
        },  # type: ignore[arg-type]
    )
    cfg.review.max_cost_usd = 100.0
    return cfg


@pytest.fixture
def client_with_tmp_config(tmp_path, monkeypatch):
    """TestClient against a tmp config dir, incl. a hand-edited custom provider."""
    import yaml

    import open_referee.server.app as app_mod

    monkeypatch.setattr(app_mod, "DEFAULT_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(app_mod, "DEFAULT_CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(app_mod, "UPLOAD_DIR", tmp_path / "uploads")
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {"llm": {"providers": {"mycustom": {"type": "ollama", "base_url": "http://x:11434"}}}}
        )
    )
    app = app_mod.create_app()
    app.state.config_path = tmp_path / "config.yaml"
    return TestClient(app)
