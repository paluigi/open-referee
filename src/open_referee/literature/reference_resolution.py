"""Deterministic per-reference resolution against Crossref + OpenAlex.

For every bibliography entry: build a query from the entry text, look it up
in Crossref (polite pool) and OpenAlex, and score the candidates against the
entry's probable title. Entries with no acceptable match become
"unmatched" — phantom-candidate findings the bibliography auditor and the
validator can use as ground truth. Free of LLM cost; time-boxed.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re

from open_referee.config import Config
from open_referee.literature.context import LiteratureItem
from open_referee.literature.crossref import CrossrefClient
from open_referee.literature.openalex_client import OpenAlexClient

logger = logging.getLogger(__name__)

MAX_REFS = 60
CONCURRENCY = 4
OVERALL_TIMEOUT_S = 60
MATCH_THRESHOLD = 0.72

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def split_references(references_text: str) -> list[str]:
    """Split a raw bibliography blob into entries.

    Handles numbered ([1] / 1.), bulleted (-) and blank-line separated lists."""
    entries: list[str] = []
    current: list[str] = []
    start_re = re.compile(r"^(?:\[\d+\]|\d{1,3}[.)]\s|[-•]\s|\*\s?)")
    for line in references_text.splitlines():
        s = line.strip()
        if not s:
            if current:
                entries.append(" ".join(current))
                current = []
            continue
        if start_re.match(s) and current:
            entries.append(" ".join(current))
            current = [s]
        elif current or start_re.match(s):
            current.append(s)
        elif len(s) > 40:
            current = [s]
    if current:
        entries.append(" ".join(current))
    return [e for e in (re.sub(r"\s+", " ", e).strip() for e in entries) if len(e) > 15]


def _title_guess(entry: str) -> str:
    """Best-effort title substring of a raw reference entry."""
    # drop the leading marker, then take the longest dot-separated chunk
    # that is not a person-name list (no year, >= 5 words, has lowercase).
    body = re.sub(r"^\s*(?:\[\d+\]|\d{1,3}[.)])\s*", "", entry)
    year_m = _YEAR_RE.search(body)
    if year_m:
        body = body[: year_m.start()] + " " + body[year_m.end() :]
    chunks = re.split(r"\.\s+|,\s+", body)
    best = ""
    for c in chunks:
        words = c.split()
        has_lower = any(w.islower() for w in words)
        if len(words) >= 4 and has_lower and len(c) > len(best):
            best = c
    return best or body


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def _score(entry: str, title_guess: str, item: LiteratureItem) -> float:
    if not item.title:
        return 0.0
    a = _norm(title_guess)
    b = _norm(item.title)
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    # whole-entry overlap helps when the title guess failed (authory chunks)
    whole = difflib.SequenceMatcher(None, _norm(entry)[:220], b).ratio()
    return max(ratio, whole * 0.9)


def _query_for(entry: str) -> str:
    body = re.sub(r"^\s*(?:\[\d+\]|\d{1,3}[.)])\s*", "", entry)
    return body[:220]


async def resolve_references(
    references_text: str, cfg: Config, *, max_refs: int = MAX_REFS
) -> list[dict]:
    entries = split_references(references_text)[:max_refs]
    if not entries:
        return []
    out: list[dict] = []
    sem = asyncio.Semaphore(CONCURRENCY)

    async with CrossrefClient(cfg.crossref) as cr, OpenAlexClient(cfg.openalex) as oa:
        async def one(entry: str) -> dict:
            async with sem:
                query = _query_for(entry)
                candidates: list[LiteratureItem] = []
                try:
                    hit = await cr.lookup(query)
                    if hit is not None:
                        candidates.append(hit)
                except Exception:  # noqa: BLE001 - resolution is best-effort
                    pass
                try:
                    candidates.extend(await oa.search_works(query, limit=3))
                except Exception:  # noqa: BLE001
                    pass
                title_guess = _title_guess(entry)
                best, best_score = None, 0.0
                for item in candidates:
                    s = _score(entry, title_guess, item)
                    if s > best_score:
                        best, best_score = item, s
                rec = {"ref": entry[:240], "status": "unmatched", "score": round(best_score, 2)}
                if best is not None and best_score >= MATCH_THRESHOLD:
                    rec.update(
                        status="matched",
                        doi=best.doi,
                        title=best.title,
                        venue=best.venue,
                        year=best.year,
                        source=best.source,
                    )
                return rec

        async def guarded(entry: str) -> dict:
            try:
                return await one(entry)
            except Exception as e:  # noqa: BLE001
                logger.warning("reference resolution failed: %s", e)
                return {
                    "ref": entry[:240], "status": "unmatched",
                    "score": 0.0, "error": str(e)[:120],
                }

        try:
            out = await asyncio.wait_for(
                asyncio.gather(*(guarded(e) for e in entries)), timeout=OVERALL_TIMEOUT_S
            )
        except TimeoutError:
            logger.warning("reference resolution timed out after %ss", OVERALL_TIMEOUT_S)
            out = [{"ref": e[:240], "status": "unmatched", "score": 0.0, "error": "timeout"}
                   for e in entries]
    return list(out)
