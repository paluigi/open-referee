"""Claim-based verification engine: inventory -> per-claim workers ->
artifact verification -> defense-adjudication.

This module implements the decomposition approach: one dedicated LLM call per
atomic claim (math re-derivation, numbers, table/figure vs text, citations,
cross-references) instead of coarse section reading, plus a per-comment
defense/adjudication quality gate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from open_referee.config import Config, ModelRole
from open_referee.ingestion import Document, build_definitions_pack
from open_referee.pipeline import prompts_claims
from open_referee.pipeline.orchestrator import RolePool, _norm
from open_referee.pipeline.state import Stage, StageStatus
from open_referee.providers.base import ChatMessage

logger = logging.getLogger(__name__)

MAX_MANUSCRIPT_CHARS = 120_000

CLAIM_TYPE_ROLES = {
    # strong model re-derives math; small handles mechanical checks
    "theorem_proof": ModelRole.STRONG,
    "equation": ModelRole.STRONG,
    "logical": ModelRole.STRONG,
    "assumption_use": ModelRole.STRONG,
    "numerical": ModelRole.SMALL,
    "table_text": ModelRole.SMALL,
    "figure_text": ModelRole.SMALL,
    "citation_claim": ModelRole.SMALL,
    "cross_reference": ModelRole.SMALL,
}


async def run_claim_inventory(
    pool: RolePool, doc: Document, triage: dict, cfg: Config
) -> list[dict]:
    """CLAIMS stage: exhaustive atomic claim catalogue for the manuscript."""
    user = prompts_claims.CLAIM_INVENTORY_USER.format(
        title=doc.title,
        triage_json=json.dumps(triage)[:3000],
        manuscript=doc.full_text()[:MAX_MANUSCRIPT_CHARS],
    )
    try:
        res = await pool.complete_json(
            ModelRole.STRONG, prompts_claims.CLAIM_INVENTORY_SYSTEM, user
        )
    except Exception as e:
        logger.warning("claim inventory failed: %s", e)
        return []
    claims = res.get("claims", [])
    if not isinstance(claims, list):
        return []
    # basic sanitation
    out = []
    for i, c in enumerate(claims):
        if not isinstance(c, dict) or not c.get("components"):
            continue
        c.setdefault("id", f"cl{i + 1:03d}")
        c.setdefault("type", "logical")
        c.setdefault("importance", "supporting")
        out.append(c)
    return out


async def verify_claims(
    pipeline, pool: RolePool, doc: Document, claims: list[dict], cfg: Config
) -> list[dict]:
    """Per-claim verification workers, parallel with the configured cap."""
    preset = cfg.review.depth_preset()
    # central claims first, then supporting; peripheral only if budget remains
    order = {"central": 0, "supporting": 1, "peripheral": 2}
    ordered = sorted(claims, key=lambda c: order.get(str(c.get("importance")).lower(), 1))
    selected = ordered[: preset.max_claims]

    sem = asyncio.Semaphore(cfg.review.max_parallel_calls)
    results: list[dict] = []

    # definitions packs: built lazily per theorem environment, keyed by anchor
    async def verify_one(claim: dict):
        async with sem:
            ctype = str(claim.get("type")).lower()
            role = CLAIM_TYPE_ROLES.get(ctype, ModelRole.SMALL)
            system = prompts_claims.claim_verify_system(ctype)
            components = json.dumps(claim.get("components", {}), indent=1)[:12_000]
            defs = _definitions_for(doc, claim)
            user = prompts_claims.CLAIM_VERIFY_USER.format(
                claim_id=claim.get("id"),
                claim_type=ctype,
                importance=claim.get("importance"),
                section=claim.get("section", ""),
                components=components,
                definitions=defs[:6000],
            )
            try:
                res = await pool.complete_json(role, system, user)
            except Exception as e:
                logger.warning("claim %s (%s) failed: %s", claim.get("id"), ctype, e)
                return []
            comments = res.get("comments", [])
            for c in comments:
                c.setdefault("paragraph_anchor", claim.get("anchor", ""))
                c["claim_id"] = claim.get("id")
            await pipeline._emit(
                Stage.VERIFY, StageStatus.RUNNING, f"claim {claim.get('id')} [{ctype}] verified"
            )
            return comments

    tasks = [verify_one(c) for c in selected]
    for r in await asyncio.gather(*tasks):
        results.extend(r)
    return results


def _definitions_for(doc: Document, claim: dict) -> str:
    """Definitions pack: match the claim's anchor to a theorem environment
    when possible; otherwise a light global pack."""
    anchor = (claim.get("anchor") or "").lower()
    for env in doc.theorems:
        if (
            env.statement_block_id
            and anchor
            and (
                _norm(env.statement)[:80] in _norm(anchor)
                or _norm(anchor)[:80] in _norm(env.statement)
            )
        ):
            return build_definitions_pack(doc, env)
    return _light_global_pack(doc)


def _light_global_pack(doc: Document) -> str:
    parts: list[str] = []
    for env in doc.theorems:
        if env.kind in ("definition", "assumption") and len(parts) < 12:
            parts.append(f"[{env.label or env.kind}] {env.statement[:300]}")
    if not parts:
        return "(no formal definitions extracted)"
    return "\n\n".join(parts)


# ------------------------------------------------------ artifact verification --

_ARTIFACT_RE = {
    "figure": re.compile(r"\b(figure|fig\.)\s*([0-9]+)", re.I),
    "table": re.compile(r"\b(table|tab\.)\s*([0-9]+)", re.I),
}


def find_mentions(doc: Document, kind: str, artifact_id: str) -> list[str]:
    """Blocks whose text mentions this figure/table. For PDF artifacts with a
    page number, also include blocks from the same page."""
    pat = _ARTIFACT_RE.get(kind)
    num = re.search(r"([0-9]+)", artifact_id)
    out: list[str] = []
    for b in doc.blocks:
        text = b.text
        if pat and num and pat.search(text):
            # matches any numbered ref; narrow by number when present in text
            out.append(text)
        elif artifact_id.startswith("tbl_md_") and b.type.value == "table":
            pass
    # keep it bounded
    return out[:12]


async def verify_artifacts(pipeline, pool: RolePool, doc: Document, cfg: Config) -> list[dict]:
    """Vision verification of ALL figures and tables against the prose."""
    out: list[dict] = []
    preset = cfg.review.depth_preset()
    if ModelRole.VISION not in pool.providers:
        return out

    vision = pool.providers[ModelRole.VISION]
    spec = pool.specs[ModelRole.VISION]
    sem = asyncio.Semaphore(cfg.review.max_parallel_calls)

    async def check(
        kind: str, artifact_id: str, image: str | None, text_block: str | None, page: int | None
    ):
        async with sem:
            if not image and not text_block:
                return []
            mentions = find_mentions(doc, kind, artifact_id)
            if not mentions:
                # no prose references: fall back to nearby blocks on the page
                mentions = _near_page_blocks(doc, page)
            content_parts = []
            if text_block:
                content_parts.append(f"Parsed contents:\n{text_block[:6000]}")
            user = prompts_claims.ARTIFACT_VERIFY_USER.format(
                artifact_kind=kind,
                artifact_id=artifact_id,
                page=page or "?",
                artifact_content_block="\n\n".join(content_parts) or "(see image)",
                mentions="\n---\n".join(m for m in mentions)[:8000],
            )
            messages = [
                ChatMessage(
                    role="system",
                    content=(
                        prompts_claims.FIGURE_VERIFY_SYSTEM
                        if kind == "figure"
                        else prompts_claims.TABLE_VERIFY_SYSTEM
                    ),
                ),
            ]
            if image:
                messages.append(ChatMessage(role="user", content=user, images=[image]))
            else:
                messages.append(ChatMessage(role="user", content=user))
            try:
                await pool.ledger.check_budget(spec, 4000, 2000)
                resp = await vision.complete(messages, json_mode=True)
                await pool.ledger.record(spec, "vision", resp)
                from open_referee.providers.adapters import extract_json

                comments = extract_json(resp.text).get("comments", [])
                for c in comments:
                    c.setdefault("paragraph_anchor", mentions[0] if mentions else "")
                    c["artifact_id"] = artifact_id
                await pipeline._emit(
                    Stage.VERIFY, StageStatus.RUNNING, f"{kind} {artifact_id} verified"
                )
                return comments
            except Exception as e:
                logger.warning("%s %s verification failed: %s", kind, artifact_id, e)
                return []

    tasks = []
    for fig in doc.figures[: preset.max_artifacts]:
        if fig.image_data_url:
            tasks.append(check("figure", fig.id, fig.image_data_url, None, fig.page))
    for tbl in doc.tables[: preset.max_artifacts]:
        if tbl.image_data_url or tbl.markdown:
            tasks.append(check("table", tbl.id, tbl.image_data_url, tbl.markdown, tbl.page))
    for r in await asyncio.gather(*tasks):
        out.extend(r)
    return out


def _near_page_blocks(doc: Document, page: int | None, window: int = 6) -> list[str]:
    if page is None:
        return []
    same = [b.text for b in doc.blocks if b.page == page]
    if same:
        return same[:window]
    return []


# ------------------------------------------------------- defense-adjudication --


async def defense_round(
    pipeline, pool: RolePool, doc: Document, candidates: list[dict], cfg: Config
) -> list[dict]:
    """Per-comment quality gate: authors' defense -> adjudication.

    Only for comments whose anchor resolves to document text (unanchored ones
    go straight to the meta-reviewer, which re-checks them globally).
    """
    preset = cfg.review.depth_preset()
    if not preset.defense_round or not candidates:
        return list(candidates)

    sem = asyncio.Semaphore(cfg.review.max_parallel_calls)

    async def gate(c: dict) -> dict | None:
        async with sem:
            anchor = c.get("paragraph_anchor") or c.get("quote") or ""
            passage = _passage_for(doc, anchor) or "(anchor not found in document)"
            try:
                defense = await pool.complete_json(
                    ModelRole.SMALL,
                    prompts_claims.DEFENSE_SYSTEM,
                    prompts_claims.DEFENSE_USER.format(
                        title=c.get("title", ""),
                        quote=(c.get("quote") or "")[:600],
                        message=(c.get("message") or "")[:1500],
                        passage=passage[:2500],
                    ),
                )
                adjudication = await pool.complete_json(
                    ModelRole.STRONG,
                    prompts_claims.ADJUDICATION_SYSTEM,
                    prompts_claims.ADJUDICATION_USER.format(
                        title=c.get("title", ""),
                        quote=(c.get("quote") or "")[:600],
                        message=(c.get("message") or "")[:1500],
                        score=c.get("score"),
                        defense_strength=defense.get("defense_strength", "unknown"),
                        defense=(defense.get("defense") or "")[:2000],
                        passage=passage[:2500],
                    ),
                )
            except Exception as e:
                logger.warning("defense gate failed for %s: %s", c.get("title"), e)
                return c  # gate failure never drops a comment
            verdict = adjudication.get("verdict")
            if verdict == "dismissed":
                return None
            final = adjudication.get("comment")
            if isinstance(final, dict) and final.get("message"):
                # preserve provenance fields from the original candidate
                merged = {k: v for k, v in c.items() if k not in final}
                merged.update(final)
                return merged
            return c

    results = await asyncio.gather(*(gate(c) for c in candidates))
    kept = [r for r in results if r is not None]
    await pipeline._emit(
        Stage.CHALLENGE,
        StageStatus.RUNNING,
        f"defense gate: {len(kept)}/{len(candidates)} comments upheld",
    )
    return kept


def _passage_for(doc: Document, anchor: str) -> str | None:
    if not anchor:
        return None
    hits = doc.find_blocks(anchor)
    if not hits:
        return None
    b = hits[0]
    lo = max(0, b.order - 1)
    hi = min(len(doc.blocks), b.order + 2)
    return "\n\n".join(x.text for x in doc.blocks[lo:hi])
