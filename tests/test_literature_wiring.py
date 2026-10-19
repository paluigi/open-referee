"""Polite-pool / mailto wiring for OpenAlex and Crossref clients."""

from __future__ import annotations

import httpx
import pytest

from open_referee.config import CrossrefConfig, OpenAlexConfig
from open_referee.literature.crossref import CrossrefClient
from open_referee.literature.openalex_client import OpenAlexClient, _params


class _Recorder(httpx.MockTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        super().__init__(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={})


@pytest.mark.asyncio
async def test_openalex_mailto_and_api_key_in_every_request(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "oa-key")
    monkeypatch.setenv("EMAIL", "researcher@uni.edu")
    client = OpenAlexClient(OpenAlexConfig())
    transport = _Recorder()
    client._http = httpx.AsyncClient(
        base_url="https://api.openalex.org", transport=transport
    )
    async with client:
        await client._search_rest("widgets", 5)
        await client.get_work_by_doi("10.1000/fake")

    assert transport.requests, "no requests were issued"
    for req in transport.requests:
        assert req.url.params.get("api_key") == "oa-key"
        assert req.url.params.get("mailto") == "researcher@uni.edu"


@pytest.mark.asyncio
async def test_openalex_explicit_email_wins_over_env(monkeypatch):
    monkeypatch.setenv("EMAIL", "fallback@uni.edu")
    cfg = OpenAlexConfig(email="explicit@uni.edu")
    assert cfg.resolved_email() == "explicit@uni.edu"
    assert _params(None, cfg.resolved_email()) == {"mailto": "explicit@uni.edu"}


@pytest.mark.asyncio
async def test_openalex_without_credentials_sends_neither(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("OPENALEX_EMAIL", raising=False)
    monkeypatch.delenv("EMAIL", raising=False)
    assert OpenAlexConfig().resolved_email() is None
    assert _params(None, None) == {}


@pytest.mark.asyncio
async def test_crossref_polite_pool_user_agent(monkeypatch):
    monkeypatch.delenv("CROSSREF_EMAIL", raising=False)
    monkeypatch.setenv("EMAIL", "me@uni.edu")
    client = CrossrefClient(CrossrefConfig())
    transport = _Recorder()
    client._http = httpx.AsyncClient(
        base_url="https://api.crossref.org/works",
        headers=client._http.headers,
        transport=transport,
    )
    async with client:
        await client.lookup("Classical widgets")

    assert len(transport.requests) == 1
    ua = transport.requests[0].headers.get("user-agent", "")
    assert "mailto:me@uni.edu" in ua


@pytest.mark.asyncio
async def test_crossref_without_email_has_plain_user_agent(monkeypatch):
    monkeypatch.delenv("CROSSREF_EMAIL", raising=False)
    monkeypatch.delenv("EMAIL", raising=False)
    client = CrossrefClient(CrossrefConfig())
    transport = _Recorder()
    client._http = httpx.AsyncClient(
        base_url="https://api.crossref.org/works",
        headers=client._http.headers,
        transport=transport,
    )
    async with client:
        await client.lookup("Classical widgets")
    assert "mailto" not in transport.requests[0].headers.get("user-agent", "")
