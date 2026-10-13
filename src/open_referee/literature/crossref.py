"""Crossref client (polite pool via mailto). Used as DOI/title fallback."""

from __future__ import annotations

import httpx

from open_referee.config import CrossrefConfig
from open_referee.literature.context import LiteratureItem

URL = "https://api.crossref.org/works"


class CrossrefClient:
    def __init__(self, cfg: CrossrefConfig):
        headers = {}
        if cfg.email:
            headers["User-Agent"] = f"open-referee/0.1 (mailto:{cfg.email})"
        self._http = httpx.AsyncClient(base_url=URL, timeout=30.0, headers=headers)

    async def __aenter__(self) -> CrossrefClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def lookup(self, title: str) -> LiteratureItem | None:
        try:
            r = await self._http.get("", params={"query.bibliographic": title, "rows": "1"})
            r.raise_for_status()
            item = (r.json().get("message", {}).get("items") or [None])[0]
            if not item:
                return None
            return LiteratureItem(
                source="crossref",
                title=(item.get("title") or [""])[0],
                authors=[
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in (item.get("author") or [])[:6]
                ],
                year=(item.get("issued", {}).get("date-parts") or [[None]])[0][0],
                doi=item.get("DOI"),
                venue=(item.get("container-title") or [None])[0],
                url=item.get("URL"),
            )
        except httpx.HTTPError:
            return None
