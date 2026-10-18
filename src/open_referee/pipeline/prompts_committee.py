"""Prompts for the planning committee (strategy reflection, replan
consultation, incomplete-review wrap-up)."""

from __future__ import annotations

import json

STRATEGY_REFLECT_SYSTEM = """\
You are the strategy chair of an academic-paper review pipeline with a hard \
USD budget. You reason about WHAT KIND of review this paper deserves; you do \
NOT choose the final plan and you never sum costs — the harness has already \
costed every option.

You receive: document statistics, the triage summary, the importance-ranked \
top claims, and a frontier of feasible review plans (each with its cost and \
what it covers: claims verified individually = depth; whole-paper/lens/\
artifact passes = breadth).

Reply with ONLY a JSON object:
{"draft": "your first-pass reasoning about depth vs breadth for this paper",
 "critique": "challenge your draft: what would a tough referee say about \
this plan for THIS paper? what is the risk of each emphasis?",
 "evaluation_summary": "2-4 sentences: the balanced recommendation",
 "preferred_path": "frontier name you lean toward"}"""

STRATEGY_REFLECT_USER_TEMPLATE = """\
## Document stats

{stats_json}

## Triage summary

{triage}

## Top importance-ranked claims

{top_claims}

## Feasible frontier (costed by the harness; budget ${budget:.2f})

{frontier_menu}
"""


def strategy_reflect_user(
    stats: dict, triage: str, top_claims: str, frontier_menu: str, budget: float
) -> str:
    return STRATEGY_REFLECT_USER_TEMPLATE.format(
        stats_json=json.dumps(stats, indent=1),
        triage=triage or "(unavailable)",
        top_claims=top_claims,
        frontier_menu=frontier_menu,
        budget=budget,
    )


REPLAN_CONSULT_SYSTEM = """\
You are the budget replanner of an academic-paper review. The run is over \
budget and several pre-costed continuations of the REMAINING work are \
available. You receive a mechanical snapshot: spend so far, residual budget, \
which importance-ranked claims are already verified, OBSERVED YIELD (findings \
per pass so far), and the continuation options with their projected final cost.

Rule of thumb: keep spending where findings are emerging; cut where they are \
not. Depth = more claims verified; breadth = more passes.

Reply with ONLY a JSON object:
{"continuation": "option name", "rationale": "1-3 sentences"}"""

REPLAN_CONSULT_USER_TEMPLATE = """\
## Mechanical snapshot

{ctx_json}

## Continuation options (projected final cost, what they include)

{paths}
"""


def replan_consult_user(ctx: dict, paths) -> str:
    lines = [
        f"- {p.name}: final ~${p.projected_final_usd:.4f}; +{p.claims_remaining} claims; "
        f"{', '.join(p.passes_remaining) or 'no new passes'} — {p.description}"
        for p in paths
    ]
    return REPLAN_CONSULT_USER_TEMPLATE.format(
        ctx_json=json.dumps(ctx, indent=1)[:20_000], paths="\n".join(lines)
    )


WRAPUP_SYSTEM = """\
You are wrapping up an academic-paper review that ran out of budget before \
completing all planned stages. Synthesize the best possible review report \
from what was actually verified — never present unverified areas as checked, \
and say explicitly which areas were NOT examined.

Reply with ONLY a JSON object:
{"paper_summary": "1-3 sentences",
 "overall_feedback": "markdown: what was verified and found; then a clearly \
marked 'Not verified' section listing the skipped areas",
 "comments": [{"title", "paragraph_anchor", "quote", "message", "score", \
"category"}],
 "validator_notes": "what remains unverified and what a continuation should do"}"""

WRAPUP_USER_TEMPLATE = """\
## Manuscript: {title}

{manuscript}

## Review progress when the budget ran out

{progress}

## Findings so far (candidate comments)

{comments}
"""


def wrapup_user(pipeline, harness) -> str:
    doc = getattr(pipeline, "_ingested_doc", None)
    state = pipeline.state
    candidates = (state.artifacts.get("candidates") if state else None) or []
    validated = (state.artifacts.get("validated") if state else None) or candidates
    progress = harness.status() if harness else {}
    coverage = {}
    if harness is not None and harness.plan is not None:
        coverage = {
            "claims_planned": min(
                harness.plan.knobs.max_claims, harness.stats.known_or_estimated_claims
            ),
            "claims_verified": harness.done.get("claim_verify", 0),
            "gate_stops": harness.gate_stopped,
            "stages_done": {
                k: v for k, v in (state.stages.items() if state else {}) if v.value == "done"
            },
        }
    return WRAPUP_USER_TEMPLATE.format(
        title=doc.title if doc else "?",
        manuscript=(doc.full_text()[:60_000] if doc else "(unavailable)"),
        progress=json.dumps({**progress, "coverage": coverage}, indent=1)[:4000],
        comments=json.dumps(validated, indent=1)[:40_000],
    )
