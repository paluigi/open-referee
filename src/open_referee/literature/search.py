"""Web search router: Tavily / TinyFish / Brave in user-defined priority order.

Failover: try engines down the list until one returns results. All engines
return a normalized SearchHit list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from open_referee.config import SearchConfig

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    engine: str = ""
    extra: dict = field(default_factory=dict)


class SearchRouter:
    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg
        self._http = httpx.AsyncClient(timeout=30.0)

    async def __aenter__(self) -> SearchRouter:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def search(self, query: str, limit: int = 8) -> list[SearchHit]:
        """Run through the engine priority list with failover."""
        for engine in self.cfg.enabled_engines():
            try:
                hits = await self._search_engine(engine, query, limit)
                if hits:
                    return hits
            except Exception as e:
                logger.warning("search engine %s failed: %s", engine, e)
        return []

    async def search_all(self, query: str, limit: int = 5) -> list[SearchHit]:
        """Query every enabled engine (used for review-community scans)."""
        out: list[SearchHit] = []
        for engine in self.cfg.enabled_engines():
            try:
                out.extend(await self._search_engine(engine, query, limit))
            except Exception as e:
                logger.warning("search engine %s failed: %s", engine, e)
        return out

    async def _search_engine(self, engine: str, query: str, limit: int) -> list[SearchHit]:
        cfg = getattr(self.cfg, engine)
        api_key = cfg.resolved_api_key()
        if engine == "tavily":
            return await self._tavily(query, limit, api_key)
        if engine == "tinyfish":
            return await self._tinyfish(query, limit, api_key)
        if engine == "brave":
            return await self._brave(query, limit, api_key)
        return []

    async def _tavily(self, query: str, limit: int, key: str | None) -> list[SearchHit]:
        if not key:
            return []
        r = await self._http.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": query, "max_results": limit},
        )
        r.raise_for_status()
        return [
            SearchHit(
                title=h.get("title", ""),
                url=h.get("url", ""),
                snippet=h.get("content", ""),
                engine="tavily",
            )
            for h in r.json().get("results", [])
        ]

    async def _tinyfish(self, query: str, limit: int, key: str | None) -> list[SearchHit]:
        if not key:
            return []
        r = await self._http.get(
            "https://api.search.tinyfish.ai/search",
            params={"query": query, "num_results": str(limit)},
            headers={"X-API-Key": key},
        )
        r.raise_for_status()
        return [
            SearchHit(
                title=h.get("title", ""),
                url=h.get("url", ""),
                snippet=h.get("snippet", ""),
                engine="tinyfish",
            )
            for h in r.json().get("results", [])
        ]

    async def _brave(self, query: str, limit: int, key: str | None) -> list[SearchHit]:
        if not key:
            return []
        r = await self._http.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": str(limit)},
            headers={"X-Subscription-Token": key},
        )
        r.raise_for_status()
        web = (r.json().get("web") or {}).get("results", [])
        return [
            SearchHit(
                title=h.get("title", ""),
                url=h.get("url", ""),
                snippet=h.get("description", ""),
                engine="brave",
            )
            for h in web
        ]
