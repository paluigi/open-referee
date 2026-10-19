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
    pool: RolePool, doc: Document, triage: dict, cfg: Config, output_budget: str = "default"
) -> list[dict]:
    """CLAIMS stage: exhaustive atomic claim catalogue for the manuscript."""
    from open_referee.pipeline.budget import call_max_tokens

    user = prompts_claims.CLAIM_INVENTORY_USER.format(
        title=doc.title,
        triage_json=json.dumps(triage)[:3000],
        manuscript=doc.full_text()[:MAX_MANUSCRIPT_CHARS],
    )
    try:
        res = await pool.complete_json(
            ModelRole.STRONG,
            prompts_claims.CLAIM_INVENTORY_SYSTEM,
            user,
            call_type="claim_inventory",
            max_tokens=call_max_tokens("claim_inventory", output_budget),
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


def claim_role(ctype: str, routing: str) -> ModelRole:
    """Role for a claim type under the plan's routing level.

    base: today's split (strong re-derives math, small does mechanical checks).
    strong: numerical/table claims also go to strong (recomputation covers the
    mechanical part). all_strong: every claim on the strong model."""
    base = CLAIM_TYPE_ROLES.get(ctype, ModelRole.SMALL)
    if routing == "all_strong":
        return ModelRole.STRONG
    if routing == "strong" and ctype in ("numerical", "table_text"):
        return ModelRole.STRONG
    return base


def _similar_comments(a: dict, b: dict) -> bool:
    """Two comments flag the same issue (title+message overlap)."""
    import difflib

    ta = _norm(str(a.get("title", "")) + " " + str(a.get("message", "")))
    tb = _norm(str(b.get("title", "")) + " " + str(b.get("message", "")))
    if not ta or not tb:
        return False
    return difflib.SequenceMatcher(None, ta, tb).ratio() >= 0.6


async def _arbitrate(pipeline, pool, claim: dict, candidate: dict) -> dict | None:
    """Strong-model gate for a finding that only one verification pass found."""
    if not pipeline.budget_allows("claim_arbitrate"):
        return None
    try:
        verdict = await pool.complete_json(
            ModelRole.STRONG,
            prompts_claims.ARBITRATE_SYSTEM,
            prompts_claims.ARBITRATE_USER.format(
                claim_id=claim.get("id"),
                claim_type=claim.get("type"),
                components=json.dumps(claim.get("components", {}))[:6000],
                title=str(candidate.get("title", ""))[:200],
                message=str(candidate.get("message", ""))[:1200],
                quote=str(candidate.get("quote", ""))[:400],
            ),
            call_type="claim_arbitrate",
        )
    except Exception as e:
        logger.warning("arbitration for claim %s failed: %s", claim.get("id"), e)
        return None
    if not verdict.get("confirmed"):
        return None
    out = dict(candidate)
    if verdict.get("message"):
        out["message"] = str(verdict["message"])[:2000]
    if verdict.get("severity") is not None:
        try:
            out["score"] = float(verdict["severity"])
        except (TypeError, ValueError):
            pass
    return out


async def verify_claims(
    pipeline, pool: RolePool, doc: Document, claims: list[dict], cfg: Config
) -> list[dict]:
    """Per-claim verification workers, parallel with the configured cap.

    When the committee already importance-ranked the claims (auto mode), the
    given order is authoritative; otherwise the inventory's coarse importance
    sorts the queue (central first).

    Depth engine: with verify_passes = k each claim is verified k times
    independently (passes >= 2 route to the auditor model when configured).
    A finding must reproduce in at least two passes to be confirmed outright;
    a single-pass finding goes to a strong-model arbitration gate."""
    knobs = pipeline.plan_knobs()
    passes = max(1, int(getattr(knobs, "verify_passes", 1)))
    routing = str(getattr(knobs, "claim_routing", "base")).lower()
    out_budget = str(getattr(knobs, "output_budget", "default"))
    from open_referee.pipeline.budget import call_max_tokens

    if getattr(pipeline, "_claims_ranked", False):
        ordered = list(claims)
    else:
        order = {"central": 0, "supporting": 1, "peripheral": 2}
        ordered = sorted(claims, key=lambda c: order.get(str(c.get("importance")).lower(), 1))
    selected = ordered[: knobs.max_claims]

    auditor_available = ModelRole.AUDITOR in getattr(pool, "providers", {})
    sem = asyncio.Semaphore(cfg.review.max_parallel_calls)
    results: list[dict] = []
    verified = 0

    async def run_pass(claim: dict, pass_idx: int) -> list[dict]:
        ctype = str(claim.get("type")).lower()
        if pass_idx >= 1 and auditor_available:
            role = ModelRole.AUDITOR
        else:
            role = claim_role(ctype, routing)
        system = prompts_claims.claim_verify_system(ctype)
        components = json.dumps(claim.get("components", {}), indent=1)[:12_000]
        defs = _definitions_for(doc, claim)
        pass_tag = (
            f"\n\n[Independent verification pass {pass_idx + 1} of {passes}. "
            "Re-derive everything from the definitions; do not assume earlier "
            "passes were correct.]"
            if passes > 1
            else ""
        )
        user = (
            prompts_claims.CLAIM_VERIFY_USER.format(
                claim_id=claim.get("id"),
                claim_type=ctype,
                importance=claim.get("importance"),
                section=claim.get("section", ""),
                components=components,
                definitions=defs[:6000],
            )
            + pass_tag
        )
        res = await pool.complete_json(
            role,
            system,
            user,
            call_type="claim_verify",
            max_tokens=call_max_tokens("claim_verify", out_budget),
        )
        comments = [c for c in res.get("comments", []) if isinstance(c, dict)]
        for c in comments:
            c.setdefault("paragraph_anchor", claim.get("anchor", ""))
            c["claim_id"] = claim.get("id")
        return comments

    async def verify_one(claim: dict):
        nonlocal verified
        async with sem:
            if not pipeline.budget_allows("claim_verify"):
                return []
            verified += 1
            pass_results: list[list[dict]] = []
            for pass_idx in range(passes):
                try:
                    pass_results.append(await run_pass(claim, pass_idx))
                except Exception as e:
                    logger.warning(
                        "claim %s pass %s failed: %s", claim.get("id"), pass_idx, e
                    )
                    pass_results.append([])
            if passes == 1:
                comments = pass_results[0]
            else:
                comments = await _aggregate_passes(
                    pipeline, pool, claim, pass_results, passes
                )
            # observed yield feeds the replanning committee
            h = getattr(pool, "harness", None)
            if h is not None and comments:
                try:
                    h.note_yield(
                        "claim_verify",
                        len(comments),
                        max((float(c.get("score", 0)) for c in comments), default=0.0),
                    )
                except (TypeError, ValueError):
                    pass
            await pipeline._emit(
                Stage.VERIFY,
                StageStatus.RUNNING,
                f"claim {claim.get('id')} verified ({passes} pass)",
            )
            return comments

    tasks = [verify_one(c) for c in selected]
    for r in await asyncio.gather(*tasks):
        results.extend(r)
    if verified < len(selected):
        pipeline._note_gate_stop("claim_verify")
        await pipeline._emit(
            Stage.VERIFY,
            StageStatus.RUNNING,
            f"claim verification stopped at {verified}/{len(selected)} (budget projection)",
        )
    return results


async def _aggregate_passes(
    pipeline, pool, claim: dict, pass_results: list[list[dict]], passes: int
) -> list[dict]:
    """Reproduction voting across independent passes.

    Findings present in >= 2 passes are confirmed with a k-of-n provenance;
    findings from a single pass go to the strong-model arbitration gate."""
    clusters: list[dict] = []  # {comments: [...], passes: set[int]}
    for pi, comments in enumerate(pass_results):
        for c in comments:
            placed = False
            for cl in clusters:
                if any(_similar_comments(c, oc) for oc in cl["comments"]):
                    cl["comments"].append(c)
                    cl["passes"].add(pi)
                    placed = True
                    break
            if not placed:
                clusters.append({"comments": [c], "passes": {pi}})
    confirmed: list[dict] = []
    for cl in clusters:
        best = max(cl["comments"], key=lambda c: float(c.get("score", 0) or 0))
        if len(cl["passes"]) >= 2:
            out = dict(best)
            out["provenance"] = f"claim_verify {len(cl['passes'])}-of-{passes}"
            confirmed.append(out)
        else:
            kept = await _arbitrate(pipeline, pool, claim, best)
            if kept is not None:
                kept["provenance"] = f"claim_verify 1-of-{passes} + arbitrated"
                confirmed.append(kept)
    return confirmed


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
    knobs = pipeline.plan_knobs()
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
            if not pipeline.budget_allows("artifact_verify"):
                pipeline._note_gate_stop("artifact_verify")
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
                rec = await pool.ledger.record(spec, "vision", resp, call_type="artifact_verify")
                if pool.harness is not None:
                    pool.harness.note_record(rec)
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
    for fig in doc.figures[: knobs.max_artifacts]:
        if fig.image_data_url:
            tasks.append(check("figure", fig.id, fig.image_data_url, None, fig.page))
    for tbl in doc.tables[: knobs.max_artifacts]:
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
    """Per-comment quality gate: authors' defense (small LLM) -> verdict.

    With a decision model configured, verdicts + recalibrated severity are
    BATCHED decision questions (uphold/adjust/dismiss + severity rubric) in
    one call over all comments; an LLM rewrites only 'adjust' comments.
    Without one, the strong-model adjudicator adjudicates per comment.
    Gate failures never drop a comment.
    """
    knobs = pipeline.plan_knobs()
    if not knobs.defense_round or not candidates:
        return list(candidates)

    if getattr(pool, "decider", None) is not None:
        return await _defense_round_dm(pipeline, pool, doc, candidates, cfg)
    return await _defense_round_llm(pipeline, pool, doc, candidates, cfg)


async def _defense_defense_texts(
    pipeline, pool: RolePool, doc: Document, candidates: list[dict], cfg: Config
) -> list[str]:
    """Authors' best honest defense per comment (small model), in parallel."""
    sem = asyncio.Semaphore(cfg.review.max_parallel_calls)

    async def one(c: dict) -> str:
        async with sem:
            if not pipeline.budget_allows("defense"):
                pipeline._note_gate_stop("defense")
                return "(no defense drafted — budget gate)"
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
                    call_type="defense",
                )
                return (
                    f"[{defense.get('defense_strength', 'unknown')}] "
                    + (defense.get("defense") or "")[:1200]
                )
            except Exception as e:
                logger.warning("defense drafting failed for %s: %s", c.get("title"), e)
                return "(defense drafting failed)"

    return list(await asyncio.gather(*(one(c) for c in candidates)))


async def _defense_round_dm(
    pipeline, pool: RolePool, doc: Document, candidates: list[dict], cfg: Config
) -> list[dict]:
    """Decision-model adjudication: batched verdicts + severity, one call."""
    from open_referee.providers.decisions import ChoiceQuestion, ScoreQuestion

    defenses = await _defense_defense_texts(pipeline, pool, doc, candidates, cfg)
    listing = "\n".join(
        f"{i}. [{c.get('title', '')[:70]}] severity={c.get('score')} "
        f"message={(c.get('message') or '')[:400]}\n   defense: {defenses[i - 1][:400]}"
        for i, c in enumerate(candidates, 1)
    )
    n = len(candidates)
    questions = []
    for i in range(1, n + 1):
        questions.append(
            ChoiceQuestion(
                name=f"verdict_{i}",
                instructions=(
                    f"Adjudicate comment {i} after reading its authors' defense. "
                    "dismiss = the defense refutes it; adjust = partially right, "
                    "severity or wording needs correction; uphold = it stands."
                ),
                choices=[{"value": v} for v in ("uphold", "adjust", "dismiss")],
            )
        )
        questions.append(
            ScoreQuestion(
                name=f"severity_{i}",
                instructions=f"Recalibrated severity for comment {i} (if it stands).",
                levels=[
                    {"label": "polish", "description": "0.1-0.34"},
                    {"label": "minor", "description": "0.35-0.59"},
                    {"label": "major", "description": "0.6-0.79"},
                    {"label": "invalidating", "description": "0.8-1.0"},
                ],
            )
        )
    try:
        dm = await pool.decide(
            f"Manuscript: {doc.title}\n\nComments with defenses:\n\n{listing}"[:60_000],
            questions,
            call_type="dm_defense",
        )
    except Exception as e:
        logger.warning("DM adjudication failed (%s); all comments upheld", e)
        return list(candidates)

    sev_map = {"polish": 0.25, "minor": 0.45, "major": 0.7, "invalidating": 0.9}
    kept: list[dict] = []
    to_adjust: list[tuple[int, dict]] = []
    for i, c in enumerate(candidates, 1):
        v = dm.answer(f"verdict_{i}")
        verdict = v.choice if v and v.choice else "uphold"
        if verdict == "dismiss":
            continue
        cc = dict(c)
        sv = dm.answer(f"severity_{i}")
        if sv is not None and sv.probabilities:
            best = max(sv.probabilities, key=lambda d: d.get("probability", 0))
            label = best.get("label")
            if label in sev_map:
                cc["score"] = sev_map[label]
        if verdict == "adjust":
            to_adjust.append((i, cc))
        else:
            kept.append(cc)

    # LLM rewrites only the adjusted comments (usually few)
    for i, cc in to_adjust:
        try:
            rewrite = await pool.complete_json(
                ModelRole.SMALL,
                prompts_claims.ADJUDICATION_SYSTEM,
                prompts_claims.ADJUDICATION_USER.format(
                    title=cc.get("title", ""),
                    quote=(cc.get("quote") or "")[:600],
                    message=(cc.get("message") or "")[:1500],
                    score=cc.get("score"),
                    defense_strength="adjusted by decision model",
                    defense=defenses[i - 1][:2000],
                    passage=(_passage_for(doc, cc.get("paragraph_anchor") or "") or "")[:2500],
                ),
                call_type="adjudication",
            )
            final = rewrite.get("comment")
            if isinstance(final, dict) and final.get("message"):
                merged = {k: v for k, v in cc.items() if k not in final}
                merged.update(final)
                kept.append(merged)
            else:
                kept.append(cc)
        except Exception as e:
            logger.warning("adjust rewrite failed for %s: %s", cc.get("title"), e)
            kept.append(cc)

    await pipeline._emit(
        Stage.CHALLENGE,
        StageStatus.RUNNING,
        f"defense gate (decision model): {len(kept)}/{len(candidates)} comments upheld "
        f"({len(to_adjust)} adjusted)",
    )
    return kept


async def _defense_round_llm(
    pipeline, pool: RolePool, doc: Document, candidates: list[dict], cfg: Config
) -> list[dict]:
    """Strong-model per-comment adjudication (no decision model configured)."""
    sem = asyncio.Semaphore(cfg.review.max_parallel_calls)

    async def gate(c: dict) -> dict | None:
        async with sem:
            if not pipeline.budget_allows("defense"):
                pipeline._note_gate_stop("defense")
                return c  # gate failure never drops a comment
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
                    call_type="defense",
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
                    call_type="adjudication",
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
