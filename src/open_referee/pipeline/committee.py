"""Planning committee: strategy and replanning protocols on the harness
frontier, with decision models in the decisive seats.

Principles (agreed with the project owner):
- the LLM (strong) REASONS — draft, self-challenge, evaluation — but has no
  commit authority;
- the decision model CHOOSES among feasible-by-construction paths and
  CONFIRMS; on a rejected confirmation exactly one more round runs and its
  choice commits unconditionally;
- an infeasible plan can never be committed: every choice is a frontier path
  costed by the harness;
- replans are unbounded in count (anti-thrash via thresholds + reserves) and
  each one rebuilds the frontier from residual steps + residual budget;
- when no continuation is feasible, a model wraps up the review from the
  accumulated context, the run is flagged incomplete with the estimated
  shortfall, and can be resumed later with a budget top-up.
"""

from __future__ import annotations

import json
import logging

from open_referee.config import ModelRole
from open_referee.pipeline import prompts_committee
from open_referee.pipeline.budget import BudgetInfeasible
from open_referee.pipeline.state import Stage, StageStatus
from open_referee.providers import LLMError
from open_referee.providers.decisions import (
    ChoiceQuestion,
    PredicateQuestion,
    ScoreQuestion,
)

logger = logging.getLogger(__name__)


class ReviewIncomplete(Exception):
    """Raised by wrap-up: carries the best-possible report for an incomplete
    review (nothing further is feasible within the budget)."""

    def __init__(self, report, cause: str, shortfall_usd: float):
        self.report = report
        self.cause = cause
        self.shortfall_usd = shortfall_usd
        super().__init__(f"review incomplete: {cause} (+${shortfall_usd:.2f} needed)")


CONFIRM_THRESHOLD = 0.5  # DM confirmation probability above this commits
ESCALATION_CONFIDENCE = 0.5  # DM choice confidence below this escalates to the LLM


# ------------------------------------------------------------- claim ranking --


async def rank_claims(pipeline, pool, claims: list[dict]) -> list[dict]:
    """Total importance order over claims. Decision model rubric when
    configured; the inventory's coarse importance is the fallback order."""
    if not claims:
        return claims
    if getattr(pool, "decider", None) is None:
        order = {"central": 0, "supporting": 1, "peripheral": 2}
        return sorted(claims, key=lambda c: order.get(str(c.get("importance")).lower(), 1))

    lines = []
    for i, c in enumerate(claims, 1):
        comps = json.dumps(c.get("components", {}))[:220]
        lines.append(
            f"{i}. [{c.get('id')}] type={c.get('type')} importance={c.get('importance')} "
            f"anchor={(c.get('anchor') or '')[:100]} components={comps}"
        )
    questions = [
        ScoreQuestion(
            name=f"claim_{c.get('id', i)}",
            instructions=(
                f"Review value of verifying claim {i} ({c.get('type')}) with an isolated "
                "verification call: how much would a finding here change the review verdict?"
            ),
            levels=[
                {"label": "low", "description": "peripheral or mechanical"},
                {"label": "medium", "description": "supporting evidence"},
                {"label": "high", "description": "material to the main results"},
                {"label": "critical", "description": "central to the paper's validity"},
            ],
        )
        for i, c in enumerate(claims, 1)
    ]
    try:
        result = await pool.decide(
            "Claims proposed for verification:\n" + "\n".join(lines),
            questions,
            call_type="dm_rank",
        )
    except LLMError as e:
        logger.warning("claim ranking degraded to inventory order: %s", e)
        order = {"central": 0, "supporting": 1, "peripheral": 2}
        return sorted(claims, key=lambda c: order.get(str(c.get("importance")).lower(), 1))

    rank_score = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    scored = []
    for c in claims:
        a = result.answer(f"claim_{c.get('id')}")
        score = 0
        if a is not None and a.score is not None:
            idx = int(round(a.score))
            labels = ["low", "medium", "high", "critical"]
            score = rank_score.get(labels[min(idx, 3)], 0)
        scored.append((score, str(c.get("importance")).lower() != "central", c))
    scored.sort(key=lambda t: (-t[0], t[1]))
    ranked = [c for _, _, c in scored]
    await pipeline._emit(
        Stage.PLAN,
        StageStatus.RUNNING,
        f"claims ranked by decision model: top={ranked[0].get('id') if ranked else '-'}",
    )
    return ranked


# -------------------------------------------------------- strategy committee --


async def strategy_committee(pipeline, pool, claims: list[dict]) -> None:
    """LLM reflects → DM chooses a frontier path → DM confirms → (on
    rejection) one more DM round commits unconditionally."""
    harness = pool.harness
    if harness is None:
        return
    frontier = harness.frontier(limit=10)
    if not frontier:
        est = harness.preset_estimates()
        raise BudgetInfeasible(
            "No feasible frontier path fits the budget for this paper.",
            min_budget_usd=round(harness.fast_floor(), 4),
            standard_budget_usd=round(est["standard"], 4),
        )
    menu = "\n".join(f"- {p.menu_line()}" for p in frontier)

    # 1. LLM reflects (strong; no commit authority)
    reflection = {}
    try:
        reflect_user = prompts_committee.strategy_reflect_user(
            stats=harness.stats.model_dump(),
            triage=json.dumps(pipeline._triage_cache or {})[:2500],
            top_claims=_top_claims(claims, 12),
            frontier_menu=menu,
            budget=harness.budget_usd,
        )
        reflection = await pool.complete_json(
            ModelRole.STRONG,
            prompts_committee.STRATEGY_REFLECT_SYSTEM,
            reflect_user,
            call_type="planner",
        )
    except LLMError as e:  # reflection is best-effort (budget errors propagate)
        logger.warning("strategy reflection failed (%s); proceeding mechanical", e)
        reflection = {}
    if harness.plan is not None and reflection:
        harness.plan.planner_calls_used += 1

    summary = {
        "draft": str(reflection.get("draft", ""))[:1500],
        "critique": str(reflection.get("critique", ""))[:1500],
        "evaluation": str(reflection.get("evaluation_summary", ""))[:1500],
        "preferred": str(reflection.get("preferred_path", ""))[:80],
    }

    chosen = None
    if pool.decider is not None and frontier:
        # 2. DM chooses among the frontier paths
        chosen = await _dm_choose_path(pool, frontier, summary, context_note="initial strategy")
        # 3. DM confirms
        if chosen is not None:
            confirm = await _dm_confirm(pool, chosen, summary)
            if confirm is not None and confirm < CONFIRM_THRESHOLD and len(frontier) > 1:
                # 4. rejected: one more round over all paths; commits as-is
                second = await _dm_choose_path(
                    pool,
                    frontier,
                    summary,
                    context_note="rejection round",
                    extra=f"The first choice ({chosen.name}) was rejected by confirmation "
                    f"(probability {confirm:.2f}). Choose again over ALL paths.",
                )
                if second is not None:
                    chosen = second
    if chosen is None:
        # fallback: LLM's preferred path, else deterministic greedy pick
        names = [p.name for p in frontier]
        pref = summary.get("preferred", "")
        chosen = next((p for p in frontier if p.name == pref), None) or frontier[-1]
        logger.info("committee fallback: %s (names=%s)", chosen.name, names)

    plan = harness.path_to_plan(
        chosen,
        reason="strategy-committee",
        rationale=(summary.get("evaluation") or "")[:500] or chosen.description,
    )
    await pipeline._emit(
        Stage.PLAN,
        StageStatus.RUNNING,
        f"committee committed frontier path '{chosen.name}' "
        f"(~${chosen.est_cost_usd:.4f}, {chosen.claims_count} claims)",
        data={
            "plan": plan.model_dump(mode="json"),
            "budget": harness.status(),
            "committee": {"reflection": summary, "path": chosen.name},
        },
    )
    pipeline._sync_budget_to_state()


def _top_claims(claims: list[dict], n: int) -> str:
    if not claims:
        return "(none)"
    lines = [
        f"{i}. [{c.get('id')}] {c.get('type')} ({c.get('importance')}): "
        f"{(c.get('anchor') or '')[:90]}"
        for i, c in enumerate(claims[:n], 1)
    ]
    return "\n".join(lines)


async def _dm_choose_path(pool, frontier, summary: dict, context_note: str, extra: str = ""):
    from open_referee.providers.decisions import Choice

    briefing = {
        "context": context_note,
        "llm_reflection": summary,
        "frontier": [p.model_dump() for p in frontier],
        "budget_usd": pool.harness.budget_usd if pool.harness else None,
        "spent_usd": pool.harness.actual_cost() if pool.harness else 0,
    }
    q = ChoiceQuestion(
        name="path_choice",
        instructions=(
            "Choose the review plan for this paper given the strategist's reflection. "
            "Depth = claims verified individually; breadth = passes over the whole paper. "
            "Weigh the reflection's evaluation."
        ),
        choices=[
            Choice(value=p.name, description=f"~${p.est_cost_usd:.4f}; {p.description}")
            for p in frontier
        ],
    )
    try:
        result = await pool.decide(
            json.dumps(briefing, indent=1)[:24_000] + extra, [q], call_type="dm_steer"
        )
    except LLMError as e:
        logger.warning("DM path choice failed: %s", e)
        return None
    a = result.answer("path_choice")
    if a is None or a.choice is None:
        return None
    return next((p for p in frontier if p.name == a.choice), None)


async def _dm_confirm(pool, path, summary: dict) -> float | None:
    q = PredicateQuestion(
        name="confirm_path",
        instructions=(
            f"Given the strategist's evaluation, '{path.name}' "
            f"({path.description}, approx ${path.est_cost_usd:.4f}) is the right review "
            "plan for this paper and budget."
        ),
    )
    try:
        result = await pool.decide(
            json.dumps({"reflection": summary, "path": path.model_dump()}, indent=1)[:12_000],
            [q],
            call_type="dm_confirm",
        )
    except LLMError as e:
        logger.warning("DM confirmation failed: %s", e)
        return None
    a = result.answer("confirm_path")
    return a.probability if a and a.probability is not None else None


# ---------------------------------------------------------- replan committee --


async def replan_choice(pipeline, pool, reason: str) -> bool:
    """Rebuild the residual frontier and pick a continuation. Returns True if
    a continuation was applied, False when NOTHING is feasible (caller must
    wrap up). Deterministic-first: a single option applies without any call."""
    harness = pool.harness
    if harness is None or harness.plan is None:
        return True
    paths = harness.residual_frontier(limit=5)
    if not paths:
        return False
    if len(paths) == 1:
        plan = harness.apply_residual(paths[0], reason=f"steering:{paths[0].name}")
        await _announce(pipeline, harness, paths[0], plan, applied_by="steering")
        return True

    # multi-option: decision model over the mechanical context snapshot
    ctx = harness.replan_context(paths)
    chosen = None
    if getattr(pool, "decider", None) is not None:
        from open_referee.providers.decisions import Choice

        q = ChoiceQuestion(
            name="continuation",
            instructions=(
                "The review is over budget. Choose the continuation that maximizes review "
                "value within the residual budget. Depth = more claims verified; breadth = "
                "more passes. Use observed yield: keep spending where findings emerge."
            ),
            choices=[
                Choice(
                    value=p.name, description=f"${p.projected_final_usd:.4f} final; {p.description}"
                )
                for p in paths
            ],
        )
        try:
            result = await pool.decide(
                json.dumps(ctx, indent=1)[:24_000], [q], call_type="dm_steer"
            )
            a = result.answer("continuation")
            if a is not None and a.choice is not None:
                conf = a.confidence or 0.0
                chosen = next((p for p in paths if p.name == a.choice), None)
                if chosen is not None and conf < ESCALATION_CONFIDENCE:
                    chosen = None  # low confidence -> escalate below
        except LLMError as e:
            logger.warning("DM continuation choice failed: %s", e)

    if chosen is None:
        # escalation (or no decider): one reflective strong-model consultation
        chosen = await _llm_consultation(pool, paths, ctx)

    if chosen is None:
        chosen = paths[-1]  # leanest continuation as the safe default
    plan = harness.apply_residual(chosen, reason=f"replan:{chosen.name}")
    await _announce(pipeline, harness, chosen, plan, applied_by="committee")
    return True


async def _llm_consultation(pool, paths, ctx: dict):
    try:
        res = await pool.complete_json(
            ModelRole.STRONG,
            prompts_committee.REPLAN_CONSULT_SYSTEM,
            prompts_committee.replan_consult_user(ctx, paths),
            call_type="planner",
        )
        name = str(res.get("continuation", ""))
        return next((p for p in paths if p.name == name), None)
    except LLMError as e:
        logger.warning("replan consultation failed: %s", e)
        return None
    finally:
        h = getattr(pool, "harness", None)
        if h is not None and h.plan is not None:
            h.plan.planner_calls_used += 1


async def _announce(pipeline, harness, path, plan, applied_by: str) -> None:
    await pipeline._emit(
        Stage.PLAN,
        StageStatus.RUNNING,
        f"replan ({applied_by}): {path.name} — {path.description}",
        data={"plan": plan.model_dump(mode="json"), "budget": harness.status()},
    )
    pipeline._sync_budget_to_state()


# ---------------------------------------------------------------- wrap-up --


async def wrapup_incomplete(pipeline, pool, cause: str):
    """No feasible continuation: synthesize the best possible report from the
    accumulated context, flag the run incomplete with the estimated shortfall,
    and raise ReviewIncomplete(report) so run() can return it cleanly."""
    harness = pool.harness
    state = pipeline.state
    shortfall = harness.shortfall() if harness else 0.0
    cheapest = harness.cheapest_completion_cost() if harness else 0.0
    if state is not None:
        state.status = "incomplete"
        state.error = f"incomplete: {cause}"
        state.shortfall_usd = shortfall
        state.save()
    await pipeline._emit(
        Stage.PLAN,
        StageStatus.FAILED,
        f"no feasible continuation ({cause}); wrapping up — est. +${shortfall:.2f} "
        f"needed (cheapest completion ≈ ${cheapest:.2f})",
    )
    draft = {}
    try:
        draft = await pool.complete_json(
            ModelRole.STRONG,
            prompts_committee.WRAPUP_SYSTEM,
            prompts_committee.wrapup_user(pipeline, harness),
            call_type="wrapup",
        )
    except Exception as e:
        logger.warning("wrap-up synthesis failed (%s); mechanical drain", e)
        draft = {}
    draft.setdefault("paper_summary", "")
    draft.setdefault("overall_feedback", "")
    draft.setdefault("comments", [])
    draft.setdefault(
        "validator_notes",
        f"INCOMPLETE REVIEW — {cause}. Estimated additional budget to finish: "
        f"${shortfall:.2f}. Resume this run with a larger budget to continue from "
        f"where it stopped (completed stages are reused).",
    )
    doc = getattr(pipeline, "_ingested_doc", None)
    if doc is not None:
        report = await pipeline._assemble(draft, doc)
        state.status = "incomplete"
        state.save()
        await pipeline._emit(Stage.ASSEMBLE, StageStatus.DONE, "incomplete report ready (wrap-up)")
        raise ReviewIncomplete(report, cause, shortfall)
    from open_referee.providers import BudgetExceeded

    raise BudgetExceeded(shortfall or 1.0, harness.budget_usd if harness else 0.0)
