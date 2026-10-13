"""Server tests via TestClient."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from conftest import overall_json
from open_referee.server.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    import open_referee.server.app as app_mod

    monkeypatch.setattr(app_mod, "DEFAULT_CONFIG_DIR", tmp_path)
    app = create_app()
    app.state.config_path = tmp_path / "config.yaml"
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_index_and_settings_pages(client):
    r = client.get("/")
    assert r.status_code == 200 and "New review" in r.text
    r = client.get("/settings")
    assert r.status_code == 200 and "LLM providers" in r.text


def test_manifest(client):
    m = client.get("/manifest.webmanifest").json()
    assert m["name"] == "Open Referee"


def test_settings_save_round_trip(client, tmp_path):
    data = {
        "provider_openrouter_type": "openai_compatible",
        "provider_openrouter_base_url": "https://openrouter.ai/api/v1",
        "provider_openrouter_api_key_env": "OPENROUTER_API_KEY",
        "role_strong_provider": "openrouter",
        "role_strong_model": "anthropic/claude-sonnet-4.5",
        "review_max_cost_usd": "7.5",
        "openalex_api_key_env": "OPENALEX_API_KEY",
        "crossref_email": "me@uni.edu",
        "search_order": "tavily,brave,tinyfish",
        "search_tavily_enabled": "on",
        "search_tavily_api_key_env": "TAVILY_API_KEY",
        "prs_pubpeer": "on",
    }
    r = client.post("/api/settings", data=data)
    assert r.status_code == 200
    cfg_file = tmp_path / "config.yaml"
    assert cfg_file.exists()
    text = cfg_file.read_text()
    assert "OPENROUTER_API_KEY" in text  # env var NAME stored
    assert "sk-" not in text  # never a real key value


def test_report_endpoints_404_when_absent(client):
    assert client.get("/api/reviews/deadbeef/report").status_code == 404
    assert client.get("/api/reviews/deadbeef/report.pdf").status_code == 404


def test_report_endpoints_serve_when_present(client, tmp_path):
    run_dir = tmp_path / "runs" / "abc123"
    run_dir.mkdir(parents=True)
    report = json.loads(overall_json())
    report.update(
        {
            "run_id": "abc123",
            "paper_title": "T",
            "comments": [
                {
                    "id": "c001",
                    "title": "t",
                    "message": "m",
                    "score": 0.5,
                    "category": "math",
                }
            ],
        }
    )
    (run_dir / "report.json").write_text(json.dumps(report))
    (run_dir / "report.md").write_text("# ok")
    r = client.get("/api/reviews/abc123/report")
    assert r.status_code == 200 and r.json()["paper_title"] == "T"
    r = client.get("/api/reviews/abc123/report.md")
    assert r.status_code == 200
