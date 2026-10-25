"""Hybrid formula recomputation: LLM extracts (formula, inputs) pairs, a
sandboxed sympy evaluator recomputes them, and a strong-model gate confirms
only real mismatches before they become findings.

Cost model: one ``recompute_extract`` (small) call per run plus one
``recompute_verify`` (strong) call per mismatch — the deterministic middle
step is free.
"""

from __future__ import annotations

import json
import logging
import re

from open_referee.config import Config, ModelRole
from open_referee.ingestion.document import Document
from open_referee.pipeline import prompts_checkers
from open_referee.pipeline.checkers import CheckFinding
from open_referee.pipeline.orchestrator import RolePool
from open_referee.pipeline.sandbox import evaluate

logger = logging.getLogger(__name__)

MAX_EXCERPT_CHARS = 12_000
REL_TOLERANCE = 0.02
ABS_TOLERANCE = 1e-9
_DECIMAL_RE = re.compile(r"\d+\.\d+")


async def run_recompute(pipeline, pool: RolePool, doc: Document, cfg: Config) -> list[CheckFinding]:
    """Extract -> evaluate -> verify; returns confirmed findings only."""
    excerpt = _excerpt(doc)
    if not excerpt:
        return []
    if not pipeline.budget_allows("recompute_extract"):
        return []
    from open_referee.pipeline.budget import call_max_tokens

    out_level = str(getattr(pipeline.plan_knobs(), "output_budget", "default"))
    try:
        res = await pool.complete_json(
            ModelRole.SMALL,
            prompts_checkers.RECOMPUTE_EXTRACT_SYSTEM,
            prompts_checkers.RECOMPUTE_EXTRACT_USER.format(title=doc.title, excerpt=excerpt),
            call_type="recompute_extract",
            max_tokens=call_max_tokens("recompute_extract", out_level),
        )
    except Exception as e:
        logger.warning("recompute extraction failed: %s", e)
        return []
    quantities = [q for q in res.get("quantities", []) if isinstance(q, dict)]
    if not quantities:
        return []

    candidates = [
        {
            "expression": str(q.get("formula", "")),
            "subs": {
                k: v
                for k, v in (q.get("inputs") or {}).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            },
        }
        for q in quantities
    ]
    results = evaluate(candidates)

    findings: list[CheckFinding] = []
    for q, r in zip(quantities, results, strict=False):
        if not r.get("ok"):
            continue
        try:
            reported = float(q.get("reported_value"))
        except (TypeError, ValueError):
            continue
        computed = float(r["value"])
        if _matches(computed, reported):
            continue
        finding = await _verify_mismatch(pipeline, pool, q, computed, reported)
        if finding is not None:
            findings.append(finding)
    if findings:
        h = getattr(pool, "harness", None)
        if h is not None:
            try:
                h.note_yield("recompute_verify", len(findings), 0.75)
            except (TypeError, ValueError):
                pass
    return findings


def _matches(computed: float, reported: float) -> bool:
    if abs(computed - reported) <= ABS_TOLERANCE:
        return True
    denom = max(abs(computed), abs(reported), 1e-12)
    return abs(computed - reported) / denom <= REL_TOLERANCE


async def _verify_mismatch(
    pipeline, pool: RolePool, q: dict, computed: float, reported: float
) -> CheckFinding | None:
    if not pipeline.budget_allows("recompute_verify"):
        return None
    from open_referee.pipeline.budget import call_max_tokens

    out_level = str(getattr(pipeline.plan_knobs(), "output_budget", "default"))
    try:
        verdict = await pool.complete_json(
            ModelRole.STRONG,
            prompts_checkers.RECOMPUTE_VERIFY_SYSTEM,
            prompts_checkers.RECOMPUTE_VERIFY_USER.format(
                symbol=str(q.get("symbol", "quantity"))[:80],
                formula=str(q.get("formula", ""))[:300],
                inputs=json.dumps(q.get("inputs") or {}),
                computed=f"{computed:.6g}",
                reported=f"{reported:.6g}",
                quote=str(q.get("quote", ""))[:400],
                section=str(q.get("section", ""))[:120],
            ),
            call_type="recompute_verify",
            max_tokens=call_max_tokens("recompute_verify", out_level),
        )
    except Exception as e:
        logger.warning("recompute verification failed: %s", e)
        return None
    if not verdict.get("confirmed"):
        return None
    try:
        severity = min(0.9, max(0.3, float(verdict.get("severity", 0.75))))
    except (TypeError, ValueError):
        severity = 0.75
    return CheckFinding(
        check="recompute",
        title=str(verdict.get("title", "Reported value disagrees with its formula"))[:120],
        message=str(
            verdict.get(
                "message",
                f"The paper reports {reported:g} but its own formula evaluates "
                f"to {computed:g}.",
            )
        ),
        quote=str(q.get("quote", ""))[:240],
        section=str(q.get("section", "")) or None,
        category="consistency",
        severity=severity,
        meta={
            "formula": str(q.get("formula", ""))[:300],
            "inputs": q.get("inputs") or {},
            "computed": computed,
            "reported": reported,
        },
    )


def _excerpt(doc: Document) -> str:
    """Equations, numeric prose paragraphs, tables, and decimal captions,
    bounded.

    Prose-stated formulas ("u = sqrt(v/q) with v = 2.0 ... so u = 0.9") are
    included whenever tables/captions exist — previously they were a
    fallback only, so any paper with a table silently lost its
    prose-embedded formulas from recomputation.
    """
    parts: list[str] = []
    for b in doc.blocks:
        if b.type.value == "equation":
            parts.append(b.text)
    parts += [
        b.text
        for b in doc.blocks
        if b.type.value in ("paragraph", "abstract")
        and len(_DECIMAL_RE.findall(b.text)) >= 3
    ][:20]
    for b in doc.blocks:
        if b.type.value == "table" and len(b.text) < 3000:
            parts.append(b.text)
        elif b.type.value == "caption" and _DECIMAL_RE.search(b.text):
            parts.append(b.text)
    for t in doc.tables:
        if t.markdown:
            parts.append(f"[table {t.id}] {t.markdown[:2000]}")
    excerpt = "\n\n".join(p for p in parts if p)[:MAX_EXCERPT_CHARS]
    return excerpt
