"""OpenAlex client backed by openalex-py (import ``openalexpy``).

Isolated behind this thin wrapper so the library can be swapped without
touching the pipeline. Always keeps a raw-REST httpx client as a fallback for
when openalexpy is unavailable or its call fails.
"""

from __future__ import annotations

import logging

import httpx

from open_referee.config import OpenAlexConfig
from open_referee.literature.context import LiteratureItem

logger = logging.getLogger(__name__)


class OpenAlexClient:
    def __init__(self, cfg: OpenAlexConfig):
        self.cfg = cfg
        self._api_key = cfg.resolved_api_key()
        self._client = None
        self._http = httpx.AsyncClient(base_url="https://api.openalex.org", timeout=30.0)

    async def __aenter__(self) -> OpenAlexClient:
        try:
            import openalexpy

            openalexpy.config.api_key = self._api_key
            self._client = openalexpy
            self._mode = "openalexpy"
        except ImportError:
            logger.info("openalex-py not installed; using raw REST client")
            self._mode = "rest"
        return self

    async def __aexit__(self, *exc) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------ api --

    async def search_works(self, query: str, limit: int = 10) -> list[LiteratureItem]:
        if self._mode == "openalexpy" and self._client is not None:
            try:
                results = await self._client.Works().search(query).get(per_page=limit)
                return [self._work_to_item(w) for w in results]
            except Exception as e:
                logger.warning("openalexpy search failed (%s); falling back to REST", e)
        return await self._search_rest(query, limit)

    async def get_work_by_doi(self, doi: str) -> LiteratureItem | None:
        doi = doi.lower().replace("https://doi.org/", "")
        if self._mode == "openalexpy" and self._client is not None:
            try:
                work = await self._client.Works().get_by_id(f"doi:{doi}")
                return self._work_to_item(work)
            except Exception:
                logger.info("openalexpy DOI lookup failed for %s; trying REST", doi)
        try:
            r = await self._http.get(f"/works/doi:{doi}", params=_params(self._api_key))
            r.raise_for_status()
            return _rest_work_to_item(r.json())
        except Exception:
            return None

    async def _search_rest(self, query: str, limit: int) -> list[LiteratureItem]:
        try:
            r = await self._http.get(
                "/works",
                params=_params(self._api_key, {"search": query, "per-page": str(limit)}),
            )
            r.raise_for_status()
            return [_rest_work_to_item(w) for w in r.json().get("results", [])]
        except Exception as e:
            logger.warning("OpenAlex REST search failed: %s", e)
            return []

    # -------------------------------------------------------------- mapping --

    @staticmethod
    def _work_to_item(w) -> LiteratureItem:
        """Map an openalexpy Work to a LiteratureItem using its real attributes.

        The pydantic Work exposes: title/name, doi, primary_location,
        publication_year, abstract, id, authorships (verified against 0.1.0).
        `authors`/`citations` do NOT exist as attributes — use authorships and
        cited_by_count.
        """
        authors: list[str] = []
        authorships = getattr(w, "authorships", None) or []
        for a in authorships:
            name = None
            inner = getattr(a, "author", None)
            if inner is not None:
                name = getattr(inner, "display_name", None) or (
                    inner.get("display_name") if isinstance(inner, dict) else None
                )
            elif isinstance(a, dict):
                name = (a.get("author") or {}).get("display_name")
            if name:
                authors.append(name)

        venue = None
        loc = getattr(w, "primary_location", None)
        if loc is not None:
            src = getattr(loc, "source", None)
            if src is not None:
                venue = getattr(src, "display_name", None) or (
                    src.get("display_name") if isinstance(src, dict) else None
                )

        doi = getattr(w, "doi", None)
        if isinstance(doi, str):
            doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "") or None
        return LiteratureItem(
            source="openalex",
            title=getattr(w, "title", None) or getattr(w, "name", None) or "",
            authors=authors[:6],
            year=getattr(w, "publication_year", None),
            doi=doi,
            venue=venue,
            cited_by=getattr(w, "cited_by_count", None),
            abstract=getattr(w, "abstract", None),
            url=getattr(w, "id", None),
        )


def _params(api_key: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    p = dict(extra or {})
    if api_key:
        p["api_key"] = api_key
    return p


def _rest_work_to_item(w: dict) -> LiteratureItem:
    authorships = w.get("authorships") or []
    return LiteratureItem(
        source="openalex",
        title=w.get("display_name") or "",
        authors=[a.get("author", {}).get("display_name", "") for a in authorships[:6]],
        year=w.get("publication_year"),
        doi=(w.get("doi") or "").replace("https://doi.org/", "").replace("http://doi.org/", "")
        or None,
        venue=((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
        cited_by=w.get("cited_by_count"),
        abstract=_rest_abstract(w),
        url=w.get("id"),
    )


def _rest_abstract(w: dict) -> str | None:
    inv = w.get("abstract_inverted_index")
    if not inv:
        return None
    pos: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            pos.append((i, word))
    return " ".join(word for _, word in sorted(pos))
