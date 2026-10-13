"""Peer-review-community scout: PubPeer, OpenReview, PREreview, etc.

Searches specialized review sites for prior discussion of the manuscript and
of its key citations; the findings inform verifier and challenger prompts.
Engines are queried with ``site:``-scoped queries when possible.
"""

from __future__ import annotations

import logging

from open_referee.config import PeerReviewSourcesConfig
from open_referee.literature.context import LiteratureItem
from open_referee.literature.search import SearchRouter

logger = logging.getLogger(__name__)

DEFAULT_SITES = {
    "pubpeer": "pubpeer.com",
    "openreview": "openreview.net",
    "prereview": "prereview.org",
}


class PeerReviewScout:
    def __init__(self, cfg: PeerReviewSourcesConfig, router: SearchRouter):
        self.cfg = cfg
        self.router = router

    def _sites(self) -> list[str]:
        sites = [domain for flag, domain in DEFAULT_SITES.items() if getattr(self.cfg, flag)]
        sites.extend(self.cfg.extra_sites)
        return sites

    async def scan_title(self, title: str) -> list[LiteratureItem]:
        """Look for prior community discussion of this exact paper."""
        out: list[LiteratureItem] = []
        for site in self._sites():
            hits = await self.router.search(f'site:{site} "{_clip(title)}"', limit=5)
            for h in hits:
                if site.split("/")[0] not in h.url:
                    continue
                out.append(
                    LiteratureItem(
                        source="review_community",
                        title=h.title or f"Discussion on {site}",
                        url=h.url,
                        abstract=h.snippet or None,
                        why_relevant=f"Peer-review community discussion found on {site}",
                    )
                )
        return out

    async def scan_citation(self, title: str) -> list[LiteratureItem]:
        """Lighter scan for a cited work (used for the top-n key references)."""
        out: list[LiteratureItem] = []
        for site in self._sites():
            hits = await self.router.search(f'site:{site} "{_clip(title)}"', limit=3)
            for h in hits:
                if site.split("/")[0] not in h.url:
                    continue
                out.append(
                    LiteratureItem(
                        source="review_community",
                        title=h.title or f"Discussion on {site}",
                        url=h.url,
                        abstract=h.snippet or None,
                        why_relevant=f"Community discussion of a cited work on {site}",
                    )
                )
        return out


def _clip(title: str, max_chars: int = 90) -> str:
    t = " ".join(title.split())
    return t[:max_chars]
