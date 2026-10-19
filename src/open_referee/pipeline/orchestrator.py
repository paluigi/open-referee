"""The review pipeline orchestrator.

Stages: ingest -> triage -> claims -> survey -> scout -> verify (parallel
per section/claim) -> challenge (parallel per section) -> bibliography ->
meta -> validate -> assemble. Every stage persists its artifact so a run
resumes where it left off. LLM role resolution and budget enforcement flow
through a RolePool; budget planning, tracking and steering flow through a
BudgetHarness (pipeline/budget.py) that owns the plan and charges every
call — including the planner's own — to the same budget.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from open_referee.config import Config, ModelRole
from open_referee.ingestion import Document, Figure, ingest_document
from open_referee.literature import (
    ContextPack,
    CrossrefClient,
    OpenAlexClient,
    PeerReviewScout,
    SearchRouter,
)
from open_referee.pipeline import prompts
from open_referee.pipeline.budget import (
    BudgetHarness,
    BudgetInfeasible,
    DocumentStats,
    PlanKnobs,
    PricingError,
    ReviewPlan,
)
from open_referee.pipeline.committee import ReviewIncomplete
from open_referee.pipeline.state import RunState, Stage, StageEvent, StageStatus
from open_referee.providers import (
    BudgetExceeded,
    ChatMessage,
    ChatProvider,
    LLMError,
    LLMResponse,
    ModelSpec,
    UsageLedger,
)
from open_referee.report import ReviewComment, ReviewReport
from open_referee.report.models import PlanSummary, UsageSummary
from open_referee.report.render import render_markdown, render_pdf

logger = logging.getLogger(__name__)

MAX_MANUSCRIPT_CHARS = 120_000


def _runs_dir() -> Path:
    from open_referee.config import DEFAULT_CONFIG_DIR

    return DEFAULT_CONFIG_DIR / "runs"


MAX_CITATION_SCANS = 6


@dataclass
class RolePool:
    """Resolved providers per role + shared ledger. Built once per run."""

    cfg: Config
    ledger: UsageLedger
    providers: dict[ModelRole, ChatProvider] = field(default_factory=dict)
    specs: dict[ModelRole, ModelSpec] = field(default_factory=dict)
    decider: object | None = None  # DecisionProvider when llm.roles.decision is set
    decider_spec: ModelSpec | None = None
    harness: BudgetHarness | None = None  # budget tracking back-reference

    @classmethod
    def build(cls, cfg: Config, roles: set[ModelRole], cap_usd: float | None = None) -> RolePool:
        from open_referee.providers.factory import build_decision_provider, build_provider

        cap = cap_usd if cap_usd is not None else cfg.review.max_cost_usd
        pool = cls(cfg=cfg, ledger=UsageLedger(cap, cfg.llm.pricing))
        for role in roles:
            role_cfg = cfg.llm.role(role)
            if role is ModelRole.DECISION:
                pool.decider = build_decision_provider(cfg.llm, role_cfg)
                pool.decider_spec = ModelSpec(
                    provider_name=role_cfg.provider,
                    provider_type="decision",
                    model=role_cfg.model,
                    base_url=cfg.llm.providers[role_cfg.provider].base_url,
                    api_key=cfg.llm.providers[role_cfg.provider].resolved_api_key(),
                    timeout_s=cfg.llm.request_timeout_s,
                    max_retries=cfg.llm.max_retries,
                )
                continue
            pool.providers[role] = build_provider(cfg.llm, role_cfg)
            pcfg = cfg.llm.providers[role_cfg.provider]
            pool.specs[role] = ModelSpec(
                provider_name=role_cfg.provider,
                provider_type=pcfg.type.value,
                model=role_cfg.model,
                base_url=pcfg.base_url,
                api_key=pcfg.resolved_api_key(),
                temperature=role_cfg.temperature,
                max_tokens=role_cfg.max_tokens,
                timeout_s=cfg.llm.request_timeout_s,
                max_retries=cfg.llm.max_retries,
            )
        return pool

    async def decide(self, input: str, questions, call_type: str = "dm_other"):
        """Typed decision call through the ledger: budget-checked, recorded,
        harness-notified — decision calls are model calls like any other."""
        if self.decider is None or self.decider_spec is None:
            raise LLMError("No decision model configured (llm.roles.decision)")
        est_in = max(1, len(input) // 3)
        await self.ledger.check_budget(self.decider_spec, est_in, 1)
        result = await self.decider.decide(input, questions)
        shim = LLMResponse(text="", usage=result.usage)
        rec = await self.ledger.record(self.decider_spec, "decision", shim, call_type=call_type)
        if self.harness is not None:
            self.harness.note_record(rec)
        return result

    async def complete_json(
        self, role: ModelRole, system: str, user: str, call_type: str = "other",
        max_tokens: int | None = None,
    ) -> dict:
        """JSON-mode completion with parse-repair retry, ledger check, usage record.

        ``max_tokens`` overrides the role's output budget for this call only
        (per-call-type budgets from the plan's output_budget level)."""
        provider = self.providers[role]
        spec = self.specs[role]
        override = max_tokens is not None and max_tokens != spec.max_tokens
        if override:
            # reuse the live provider (and its HTTP client); restore after
            provider.spec = spec.model_copy(update={"call_max_tokens": max_tokens})
        try:
            return await self._complete_json_inner(
                provider, spec, role, system, user, call_type
            )
        finally:
            if override:
                provider.spec = spec

    async def _complete_json_inner(
        self, provider, spec, role: ModelRole, system: str, user: str, call_type: str
    ) -> dict:
        est_in = (len(system) + len(user)) // 3
        # rough output estimate: proportional to input, bounded — avoids
        # pre-charging max_tokens/2 on every call and tripping the cap early
        est_out = min(spec.max_tokens, max(1_000, est_in // 4))
        await self.ledger.check_budget(spec, est_in, est_out)
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]
        resp = await provider.complete(messages, json_mode=True)
        rec = await self.ledger.record(spec, role.value, resp, call_type=call_type)
        if self.harness is not None:
            self.harness.note_record(rec)
        from open_referee.providers.adapters import extract_json

        try:
            return extract_json(resp.text)
        except json.JSONDecodeError:
            repair_user = (
                f"{user}\n\nYour previous reply was not valid JSON. Reply again with ONLY the "
                f"required JSON object, no prose, no code fences."
            )
            resp2 = await provider.complete(
                [
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=repair_user),
                ],
                json_mode=True,
            )
            rec2 = await self.ledger.record(spec, role.value, resp2, call_type=call_type)
            if self.harness is not None:
                self.harness.note_record(rec2)
            try:
                return extract_json(resp2.text)
            except json.JSONDecodeError as e:
                # never leak a raw decode error: stages catch LLMError and
                # degrade gracefully instead of failing the whole run
                raise LLMError(
                    f"{spec.provider_name}/{spec.model} returned no parseable JSON even "
                    f"after a repair attempt (last reply started: {resp2.text[:120]!r})"
                ) from e

    async def close(self) -> None:
        for p in self.providers.values():
            await p.close()


class ReviewPipeline:
    def __init__(
        self,
        cfg: Config,
        run_id: str | None = None,
        budget_usd: float | None = None,
        journal_profile: str | None = None,
        depth: str | None = None,
    ):
        self.cfg = cfg
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.requested_budget_usd = budget_usd
        self.journal_profile = (journal_profile or "").strip() or None
        self.requested_depth = (depth or cfg.review.depth).lower().strip()
        self.state: RunState | None = None
        self._subscribers: list[asyncio.Queue[StageEvent]] = []
        self._pool: RolePool | None = None
        self._triage_cache: dict | None = None
        self._claims_ranked = False
        self._venue_block: str = ""
        if self.journal_profile:
            self._venue_block = (
                "\n\n## Target venue profile (free-form, from the user)\n\n"
                f"{self.journal_profile[:8000]}\n\n"
                "Review with this venue's aims, scope and standards in mind.\n"
            )

    # ---------------------------------------------------------------- events --

    def subscribe(self) -> asyncio.Queue[StageEvent]:
        q: asyncio.Queue[StageEvent] = asyncio.Queue()
        self._subscribers.append(q)
        return q

    async def _emit(
        self,
        stage: Stage,
        status: StageStatus,
        message: str = "",
        progress: float | None = None,
        data: dict | None = None,
    ):
        ev = StageEvent(
            run_id=self.run_id,
            stage=stage,
            status=status,
            message=message,
            progress=progress,
            data=data or {},
        )
        for q in self._subscribers:
            await q.put(ev)
        if self.state is not None:
            self.state.events.append(ev)

    # ------------------------------------------------------------------- run --

    async def run(
        self, paper_path: str | Path, literature_paths: list[str | Path] | None = None
    ) -> ReviewReport:
        state = self._load_or_create_state(paper_path, literature_paths)
        resuming_incomplete = state.status == "incomplete"
        if resuming_incomplete:
            # restart-with-top-up: continue from persisted artifacts under the
            # new budget without rerunning completed stages
            state.status = "running"
            state.error = None
        self.state = state
        state.run_type = "review"
        state.requested_depth = self.requested_depth
        state.budget_usd = self.requested_budget_usd
        state.journal_profile = self.journal_profile
        state.save()
        try:
            pool = await self._ensure_pool()
            # 1. ingest
            await self._emit(Stage.INGEST, StageStatus.RUNNING)
            doc = await asyncio.to_thread(ingest_document, paper_path)
            self._ingested_doc = doc
            # artifacts: theorem environments + markdown tables (PDF ones come
            # from the reader)
            from open_referee.ingestion import extract_tables, extract_theorems

            doc.theorems = await asyncio.to_thread(extract_theorems, doc)
            doc.tables = doc.tables + await asyncio.to_thread(extract_tables, doc)
            lit_docs = [
                await asyncio.to_thread(ingest_document, p)
                for p in (literature_paths or [])[: self.cfg.review.max_user_literature_docs]
            ]
            state.paper_title = doc.title
            state.artifacts["ingest"] = {
                "title": doc.title,
                "n_blocks": len(doc.blocks),
                "n_figures": len(doc.figures),
                "lit_titles": [d.title for d in lit_docs],
            }
            state.set_stage(Stage.INGEST, StageStatus.DONE)
            state.save()
            await self._emit(Stage.INGEST, StageStatus.DONE, doc.title)

            # 1b. budget preflight + initial (deterministic) plan — free of LLM cost
            await self._init_budget(pool, doc)
            if resuming_incomplete and pool.harness is not None and pool.harness.plan is not None:
                from open_referee.pipeline import committee

                # pick up the continuation the new budget can afford
                await committee.replan_choice(self, pool, "resume-topup")

            # 2. triage
            triage = await self._stage_or_resume(
                Stage.TRIAGE, lambda: self._triage(pool, doc), skip_key="triage"
            )
            self._triage_cache = triage
            queries = triage.get("search_queries", [])[:8]
            key_citations = triage.get("key_citations", [])

            # 2b. claim inventory (atomic checkable claims) + artifact/theorem extraction
            claims = await self._stage_or_resume(
                Stage.CLAIMS, lambda: self._claim_inventory(pool, doc, triage), skip_key="claims"
            )
            # claims milestone: exact claim count is the dominant cost driver.
            # claims are importance-ranked (decision model when available),
            # then the committee commits a frontier path for the run.
            if pool.harness is not None:
                pool.harness.stats.n_claims = len(claims) if claims else 0
            if self._is_auto() and not state.stage_status(Stage.VERIFY) == StageStatus.DONE:
                from open_referee.pipeline import committee

                claims = await committee.rank_claims(self, pool, claims)
                self._claims_ranked = True
                state.artifacts["claims"] = claims  # ranked order persists
                await committee.strategy_committee(self, pool, claims)

            # 3+4. literature survey + community scout (concurrent)
            pack = await self._stage_or_resume(
                Stage.SURVEY,
                lambda: self._literature(pool, doc, triage, queries, key_citations, lit_docs),
                skip_key="survey",
            )
            if isinstance(pack, dict):
                # resumed runs load artifacts as JSON: rebuild the ContextPack
                pack = ContextPack.model_validate(pack)
            await self._stage_or_resume(
                Stage.SCOUT, lambda: self._scout(pool, doc.title, key_citations, pack)
            )

            # 5. verify: per-claim workers + section lenses (by plan) + artifacts + whole-paper
            candidates = await self._stage_or_resume(
                Stage.VERIFY,
                lambda: self._verify(pool, doc, pack, triage, claims),
                skip_key="candidates",
            )
            await self._replan_checkpoint(pool, "verify")
            # 6. challenge
            validated = await self._stage_or_resume(
                Stage.CHALLENGE,
                lambda: self._challenge(pool, doc, pack, candidates),
                skip_key="validated",
            )
            await self._replan_checkpoint(pool, "challenge")
            # 7. bibliography
            bib = await self._stage_or_resume(
                Stage.BIBLIOGRAPHY,
                lambda: self._bibliography(pool, doc, pack),
                skip_key="bibliography",
            )
            # 8. meta
            draft = await self._stage_or_resume(
                Stage.META,
                lambda: self._meta(pool, doc, triage, validated, bib, pack),
                skip_key="draft_review",
            )
            # 9. validate
            final = await self._stage_or_resume(
                Stage.VALIDATE, lambda: self._validate(pool, doc, draft), skip_key="final_review"
            )
            # 10. assemble
            report = await self._assemble(final, doc)
            state.status = "completed"
            state.save()
            await self._emit(Stage.ASSEMBLE, StageStatus.DONE, "report ready")
            self._record_history("completed")
            return report
        except (PricingError, BudgetInfeasible) as e:
            # pre-spend abort: no partial report is possible or meaningful
            if state:
                state.status = "failed"
                state.error = str(e)
                state.save()
            self._record_history("failed", str(e))
            await self._emit(Stage.ASSEMBLE, StageStatus.FAILED, f"cannot run: {e}")
            raise
        except ReviewIncomplete as e:
            # wrap-up produced the best possible report; the run is resumable
            # with a budget top-up (completed stages are reused)
            self._record_history("incomplete", e.cause)
            return e.report
        except BudgetExceeded as e:
            # Drain gracefully: assemble a partial report from whatever the
            # pipeline managed to produce before the spend cap tripped.
            if state:
                state.status = "failed"
                state.error = str(e)
                state.save()
            self._record_history("budget_exceeded", str(e))
            await self._emit(Stage.ASSEMBLE, StageStatus.FAILED, f"budget cap reached: {e}")
            doc: Document | None = getattr(self, "_ingested_doc", None)
            if doc is not None and state is not None:
                try:
                    draft = state.artifacts.get("draft_review")
                    if not isinstance(draft, dict):
                        draft = {}
                    draft.setdefault("paper_summary", "")
                    draft.setdefault("overall_feedback", "")
                    draft.setdefault("comments", [])
                    draft.setdefault(
                        "validator_notes",
                        f"Partial report: the run stopped early ({e}). "
                        "Stages completed: "
                        + ", ".join(
                            k
                            for k, v in (state.stages.items() if state else {})
                            if v.value == "done"
                        ),
                    )
                    report = await self._assemble(draft, doc)
                    return report
                except Exception:
                    logger.exception("partial-report assembly failed")
            raise
        except Exception as e:
            if state:
                state.status = "failed"
                state.error = str(e)
                state.save()
            self._record_history("failed", str(e))
            logger.exception("pipeline failed")
            raise
        finally:
            if self._pool:
                await self._pool.close()

    # ---------------------------------------------------------------- helpers --

    def _load_or_create_state(
        self, paper_path: str | Path, literature_paths: list[str | Path] | None
    ) -> RunState:
        """Resume an interrupted run (same run_id) or start a fresh state."""
        from open_referee.pipeline.state import _runs_dir

        existing = _runs_dir() / self.run_id / "state.json"
        if existing.exists():
            try:
                return RunState.load(self.run_id)
            except Exception:
                logger.warning("could not load state for %s; starting fresh", self.run_id)
        return RunState(
            run_id=self.run_id,
            paper_path=str(paper_path),
            literature_paths=[str(p) for p in (literature_paths or [])],
        )

    def _effective_depth(self) -> str:
        if self.state is not None and self.state.requested_depth:
            return self.state.requested_depth
        return self.requested_depth

    def _is_auto(self) -> bool:
        return self._effective_depth() == "auto"

    def _effective_budget(self) -> float:
        if self.requested_budget_usd is not None:
            return self.requested_budget_usd
        return self.cfg.review.max_cost_usd

    async def _ensure_pool(self) -> RolePool:
        if self._pool is None:
            roles = {ModelRole.STRONG, ModelRole.SMALL}
            for extra in (ModelRole.VISION, ModelRole.DECISION, ModelRole.AUDITOR):
                try:
                    self.cfg.llm.role(extra)
                    roles.add(extra)
                except KeyError:
                    pass
            self._pool = RolePool.build(self.cfg, roles, cap_usd=self._effective_budget())
        return self._pool

    # ---------------------------------------------------------------- budget --

    async def _init_budget(self, pool: RolePool, doc: Document) -> None:
        """Preflight + initial plan (no LLM cost).

        Auto mode: pricing must be configured (hard error) and the budget
        must clear the fast-preset floor for THIS paper (hard error listing
        all three preset estimates — rejected before any model call).
        Preset modes: uncapped by design — a deterministic estimate of the
        chosen preset is tracked for the live view and the final
        estimate-vs-actual report, but nothing is enforced in flight.
        """
        import math

        harness = BudgetHarness(
            self.cfg,
            pool.ledger,
            pool.specs,
            budget_usd=self._effective_budget(),
            auto=self._is_auto(),
        )
        harness.stats = DocumentStats.from_doc(doc)
        if pool.decider_spec is not None:
            harness.specs[ModelRole.DECISION] = pool.decider_spec
        pool.harness = harness
        if not self._is_auto():
            # uncapped preset mode: estimate-only tracking, no enforcement
            pool.ledger.cap_usd = math.inf
        # resume: restore plan + spend accounting from persisted state
        assert self.state is not None
        if self.state.ledger_calls:
            from open_referee.providers.usage import CallRecord

            pool.ledger.calls = [CallRecord.model_validate(c) for c in self.state.ledger_calls]
            for rec in pool.ledger.calls:
                harness.note_record(rec)
        if self.state.plan is not None:
            try:
                restored = ReviewPlan.model_validate(self.state.plan)
                restored.budget_usd = self._effective_budget()
                harness.set_plan(restored, reason="restored-resume")
            except Exception:
                logger.warning("could not restore persisted plan; replanning")
        if harness.plan is not None:
            return
        try:
            harness.require_pricing()
        except PricingError as e:
            if self._is_auto():
                raise
            logger.warning("pricing incomplete, budget tracking degraded: %s", e)
        if self._is_auto():
            # floor: reject budgets below the fast-preset estimate, zero model calls
            est = harness.preset_estimates()
            if harness.budget_usd < est["fast"]:
                raise BudgetInfeasible(
                    f"Budget ${harness.budget_usd:.2f} is below the cost of even a "
                    f"'fast' review of this paper (≈ ${est['fast']:.2f}). Preset "
                    f"estimates for this manuscript: fast ≈ ${est['fast']:.2f}, "
                    f"standard ≈ ${est['standard']:.2f}, deep ≈ ${est['deep']:.2f}.",
                    min_budget_usd=round(est["fast"], 2),
                    standard_budget_usd=round(est["standard"], 2),
                )
            plan = harness.deterministic_plan(reason="preflight")
            harness.set_plan(plan, reason="preflight-v0")
            await self._emit_plan_event("plan v0 (deterministic preflight)", plan)
        else:
            depth = self._effective_depth()
            knobs = PlanKnobs.from_preset(self.cfg.review.depth_preset())
            plan = harness.preset_plan(knobs, depth)
            harness.set_plan(plan, reason="preset-tracking")
            await self._emit_plan_event(
                f"preset '{depth}' — estimate ${plan.target_usd:.2f} (uncapped, "
                "spend reported at the end)",
                plan,
            )
        self._sync_budget_to_state()
        self.state.save()

    async def _emit_plan_event(self, message: str, plan: ReviewPlan) -> None:
        st = self._pool.harness.status() if self._pool and self._pool.harness else {}
        await self._emit(
            Stage.PLAN,
            StageStatus.RUNNING,
            message,
            data={"plan": plan.model_dump(mode="json"), "budget": st},
        )

    async def _replan_checkpoint(self, pool: RolePool, where: str) -> None:
        """Checkpoint after stages/sub-stages: telemetry when on-plan; when the
        harness flags a deviation (or a spike), rebuild the residual frontier
        and let the committee pick a continuation. When NOTHING is feasible,
        wrap up the review as incomplete (raises ReviewIncomplete)."""
        from open_referee.pipeline import committee

        harness = pool.harness
        if harness is None or harness.plan is None:
            return
        reason = harness.should_replan()
        st = harness.status()
        if reason is None:
            await self._emit(
                Stage.PLAN,
                StageStatus.RUNNING,
                f"budget after {where}: ${st['actual_usd']:.2f} spent, "
                f"${st['projected_final_usd']:.2f} projected of ${st['budget_usd']:.2f}",
                data={"budget": st},
            )
            self._sync_budget_to_state()
            return
        applied = await committee.replan_choice(self, pool, reason)
        if not applied:
            await committee.wrapup_incomplete(self, pool, reason)
        self.state.save()

    def _sync_budget_to_state(self) -> None:
        """Persist plan + ledger so a resumed run keeps spend accounting."""
        if self.state is None or self._pool is None:
            return
        h = self._pool.harness
        if h is not None:
            self.state.plan = h.plan.model_dump(mode="json") if h.plan else None
            self.state.ledger_calls = [c.model_dump(mode="json") for c in h.ledger.calls]

    def _record_history(self, status: str, error: str | None = None) -> None:
        """Append the per-phase estimation record (paper characteristics +
        budget/spend detail) to the run-history store. Best-effort: a history
        failure must never break the run itself."""
        try:
            from open_referee.pipeline.history import build_run_record, upsert_run_record

            pool = getattr(self, "_pool", None)
            if pool is None or pool.harness is None:
                return
            rec = build_run_record(
                run_id=self.run_id,
                run_type="review",
                status=status,
                harness=pool.harness,
                events=self.state.events if self.state is not None else [],
                paper_title=self.state.paper_title if self.state is not None else "",
                depth_mode=self._effective_depth(),
                error=error,
                models={r.value: f"{p.provider_name}/{p.model}" for r, p in pool.specs.items()},
            )
            upsert_run_record(rec)
        except Exception:
            logger.exception("failed to record run history")

    # --------------------------------------------------------- plan knobs --

    def plan_knobs(self) -> PlanKnobs:
        """Current plan knobs (harness plan if present, else depth preset)."""
        pool = getattr(self, "_pool", None)
        if pool is not None and pool.harness is not None:
            plan = pool.harness.plan
            if plan is not None:
                return plan.knobs
        return PlanKnobs.from_preset(self.cfg.review.depth_preset())

    def budget_allows(self, call_type: str) -> bool:
        """Fan-out gate: keeps the projection under the hard budget."""
        pool = getattr(self, "_pool", None)
        if pool is None or pool.harness is None:
            return True
        return pool.harness.allow_call(call_type)

    def _note_gate_stop(self, call_type: str) -> None:
        """Record + announce (once per call type) that the budget gate tripped."""
        pool = getattr(self, "_pool", None)
        if pool is None or pool.harness is None:
            return
        h = pool.harness
        first = call_type not in h.gate_stopped
        h.note_gate_stop(call_type, 1)
        if first:
            st = h.status()
            logger.info(
                "budget gate stopped %s (projected $%.2f of $%.2f)",
                call_type,
                st["projected_final_usd"],
                st["budget_usd"],
            )

    async def _stage_or_resume(self, stage: Stage, coro_fn, skip_key: str | None = None):
        assert self.state is not None, "run() must be invoked first"
        status = self.state.stage_status(stage)
        if status is StageStatus.DONE and skip_key and skip_key in self.state.artifacts:
            await self._emit(stage, StageStatus.DONE, "resumed from saved state", 1.0)
            return self.state.artifacts[skip_key]
        self.state.set_stage(stage, StageStatus.RUNNING)
        self.state.save()
        await self._emit(stage, StageStatus.RUNNING)
        try:
            result = await coro_fn()
        except (LLMError, BudgetExceeded):
            self.state.set_stage(stage, StageStatus.FAILED)
            self.state.save()
            raise
        self.state.set_stage(stage, StageStatus.DONE)
        if skip_key:
            self.state.artifacts[skip_key] = result
        self._sync_budget_to_state()
        self.state.save()
        await self._emit(stage, StageStatus.DONE, progress=1.0)
        return result

    def _manuscript_trunc(self, doc: Document) -> str:
        t = doc.full_text()
        return t[:MAX_MANUSCRIPT_CHARS]

    def _pack_for_section(self, pack: ContextPack | None) -> str:
        if not pack:
            return ""
        return pack.render(max_items=15)[:6000]

    # ----------------------------------------------------------------- stages --

    def _build_plan_summary(self) -> PlanSummary | None:
        """Telemetry for the report: how the run was budgeted vs actual."""
        if self._pool is None or self._pool.harness is None:
            return None
        h = self._pool.harness
        st = h.status()
        notes: list[str] = []
        for ct, skipped in sorted(h.gate_stopped.items()):
            notes.append(f"{ct}: {skipped} unit(s) skipped by the budget gate")
        plan = h.plan
        return PlanSummary(
            mode="auto" if self._is_auto() else self._effective_depth(),
            budget_usd=st["budget_usd"],
            target_usd=st["target_usd"],
            planned_usd=st["planned_usd"],
            spent_usd=st["actual_usd"],
            control_spent_usd=st["control_actual_usd"],
            projected_final_usd=st["projected_final_usd"],
            planner_calls_used=st["planner_calls_used"],
            adjustments=[
                f"[{a.source}] {a.action}: {a.detail or a.reason}"
                for a in (plan.adjustments if plan else [])
            ],
            notes="; ".join(notes) or None,
        )

    async def _dm_triage_classification(self, pool: RolePool, doc: Document) -> dict:
        """Closed-set triage classification via the decision model (batched,
        one call). Open-vocabulary fields stay with the LLM."""
        if getattr(pool, "decider", None) is None:
            return {}
        from open_referee.providers.decisions import ChoiceQuestion, ScoreQuestion

        abstract = doc.full_text()[:8000]
        questions = [
            ChoiceQuestion(
                name="paper_type",
                instructions="The manuscript's dominant type.",
                choices=[{"value": v} for v in ("theory", "empirical", "mixed", "survey", "other")],
            ),
            ScoreQuestion(
                name="mathematical_density",
                instructions="Density of formal mathematics (theorems, proofs, derivations).",
                levels=[{"label": v} for v in ("none", "light", "moderate", "heavy")],
            ),
            ChoiceQuestion(
                name="statistical_content",
                instructions="The kind of statistical/econometric content.",
                choices=[
                    {"value": v}
                    for v in ("none", "descriptive", "inferential", "causal", "forecasting")
                ],
            ),
            ChoiceQuestion(
                name="methodological_style",
                instructions="The dominant methodological style.",
                choices=[
                    {"value": v}
                    for v in (
                        "analytical-theory",
                        "econometrics",
                        "machine-learning",
                        "simulation",
                        "mixed",
                    )
                ],
            ),
        ]
        try:
            result = await pool.decide(
                f"Manuscript: {doc.title}\n\n{abstract}", questions, call_type="dm_triage"
            )
        except LLMError as e:
            logger.warning("DM triage classification failed (%s); LLM covers it", e)
            return {}
        out: dict = {}
        a = result.answer("paper_type")
        if a and a.choice:
            out["paper_type"] = a.choice
        a = result.answer("mathematical_density")
        if a and a.probabilities:
            # score level index -> label with the highest probability
            best = max(a.probabilities, key=lambda d: d.get("probability", 0))
            if isinstance(best.get("value"), int) and 0 <= best["value"] < 4:
                out["mathematical_density"] = ["none", "light", "moderate", "heavy"][best["value"]]
        for name in ("statistical_content", "methodological_style"):
            a = result.answer(name)
            if a and a.choice:
                out[name] = a.choice
        return out

    async def _triage(self, pool: RolePool, doc: Document) -> dict:
        # decision-model half: closed-set classification (skipped without a decider)
        dm_class = await self._dm_triage_classification(pool, doc)
        # LLM half: open-vocabulary domain + generative fields (queries, citations)
        user = (
            prompts.TRIAGE_USER.format(title=doc.title, manuscript=self._manuscript_trunc(doc))
            + self._venue_block
        )
        if dm_class:
            user += (
                "\n\n## Classification already determined (use as given)\n\n"
                + json.dumps(dm_class)
                + "\nFocus your reply on domain, search queries, key citations, main claims.\n"
            )
        triage = await pool.complete_json(
            ModelRole.STRONG, prompts.TRIAGE_SYSTEM, user, call_type="triage"
        )
        triage.update(dm_class)  # DM classification wins on shared keys
        if pool.harness is not None:
            pool.harness.stats.n_lenses = len(self._lenses_for(triage))
        return triage

    async def _claim_inventory(self, pool: RolePool, doc: Document, triage: dict) -> list[dict]:
        from open_referee.pipeline.claims import run_claim_inventory

        claims = await run_claim_inventory(
            pool, doc, triage, self.cfg,
            output_budget=str(getattr(self.plan_knobs(), "output_budget", "default")),
        )
        await self._emit(Stage.CLAIMS, StageStatus.RUNNING, f"{len(claims)} claims catalogued")
        return claims

    async def _literature(self, pool, doc, triage, queries, key_citations, lit_docs) -> ContextPack:
        pack = ContextPack(paper_title=doc.title, field_hint=triage.get("domain"))
        pack.user_docs = [d.title for d in lit_docs]
        items: list = []
        async with (
            OpenAlexClient(self.cfg.openalex) as oa,
            CrossrefClient(self.cfg.crossref) as cr,
            SearchRouter(self.cfg.search) as router,
        ):
            try:
                for q in queries:
                    items.extend(await oa.search_works(q, limit=5))
                    if len(items) >= 30:
                        break
                # user-supplied literature: try to match in OpenAlex for metadata
                for d in lit_docs:
                    hit = await oa.search_works(d.title, limit=1)
                    for h in hit:
                        h.source = "user_pdf"
                        h.why_relevant = "Author-supplied related work"
                        items.append(h)
                # web search for very recent / non-scholarly context
                for q in queries[:3]:
                    for hit in await router.search(q, limit=4):
                        items.append(
                            _hit_to_item(hit, why="Web result (recent or non-scholarly source)")
                        )
            except Exception as e:
                # literature enrichment is best-effort; never fail the run for it
                logger.warning("literature collection degraded: %s", e)
            # dedupe by normalized title
            seen: set[str] = set()
            for it in items:
                key = re.sub(r"\W+", "", it.title.lower())[:80]
                if key and key not in seen:
                    seen.add(key)
                    pack.items.append(it)
            # decision-model relevance filter (batched predicates, one call):
            # drop clearly irrelevant items before the surveyor reads the pack
            if getattr(pool, "decider", None) is not None and pack.items:
                from open_referee.providers.decisions import PredicateQuestion

                q_items = "\n".join(
                    f"{i}. {it.title[:120]} — {(it.why_relevant or '')[:80]}"
                    for i, it in enumerate(pack.items[:40], 1)
                )
                qs = [
                    PredicateQuestion(
                        name=f"item_{i}",
                        instructions=(
                            f"Literature item {i} is relevant background for reviewing a "
                            "manuscript in this field and should be kept in the context pack."
                        ),
                    )
                    for i in range(1, min(len(pack.items), 40) + 1)
                ]
                try:
                    dm = await pool.decide(
                        f"Field: {triage.get('domain', 'unknown')}\n\nItems:\n{q_items}",
                        qs,
                        call_type="dm_survey",
                    )
                    kept = []
                    for i, it in enumerate(pack.items[:40], 1):
                        a = dm.answer(f"item_{i}")
                        if a is None or a.probability is None or a.probability >= 0.5:
                            kept.append(it)
                    dropped = len(pack.items) - len(kept)
                    pack.items = kept + pack.items[40:]
                    if dropped:
                        await self._emit(
                            Stage.SURVEY,
                            StageStatus.RUNNING,
                            f"decision filter dropped {dropped} irrelevant literature items",
                        )
                except LLMError as e:
                    logger.warning("DM survey filter failed (%s); keeping all items", e)
            # surveyor refines relevance + missing refs (small role) — still inside
            # the client contexts so Crossref lookups below remain valid
            try:
                survey = await pool.complete_json(
                    ModelRole.SMALL,
                    prompts.SURVEY_SYSTEM,
                    pack.render(max_items=30),
                    call_type="surveyor",
                )
                for sel in survey.get("selected", []):
                    idx = sel.get("index")
                    if isinstance(idx, int) and 1 <= idx <= len(pack.items):
                        pack.items[idx - 1].why_relevant = sel.get("why_relevant")
                for m in survey.get("missing_references", [])[:5]:
                    extra = await cr.lookup(m.get("title", "")) if m.get("title") else None
                    if extra:
                        extra.why_relevant = m.get("why_important")
                        pack.items.append(extra)
                pack.field_hint = (pack.field_hint or "") + (
                    f" — {survey.get('state_of_the_art_notes', '')}"[:400]
                )
            except (LLMError, BudgetExceeded):
                raise
            except Exception as e:
                logger.warning("surveyor stage degraded: %s", e)
        return pack

    async def _scout(self, pool, title: str, key_citations: list, pack: ContextPack) -> None:
        router = SearchRouter(self.cfg.search)
        scout = PeerReviewScout(self.cfg.peer_review_sources, router)
        try:
            found = await scout.scan_title(title)
            for c in key_citations[:MAX_CITATION_SCANS]:
                cit_title = re.sub(r"^.*?:\s*", "", str(c))
                found.extend(await scout.scan_citation(cit_title))
            pack.items.extend(found[:15])
        finally:
            await router.close()

    def _reviewable_sections(self, doc: Document) -> list[tuple[str, str]]:
        """(title, text) for chunks worth verifying. Falls back to a
        heading-heuristic split when parsing produced no sections or a single
        degenerate one covering the whole document (plain-text conversions)."""
        out: list[tuple[str, str]] = []
        if doc.sections:
            for s in doc.sections:
                text = "\n\n".join(b.text for b in doc.blocks[s.start_block : s.end_block])
                if len(text) > 400:
                    out.append((s.title, text))
        if len(out) <= 1:
            split = _split_by_headings(doc.full_text())
            if len(split) > len(out):
                out = split
        if not out:
            out.append(("Full manuscript", doc.full_text()))
        return out

    def _lenses_for(self, triage: dict) -> list[tuple[str, str]]:
        """Select verifier lenses based on triage: (name, system prompt)."""
        math_density = str(triage.get("mathematical_density", "")).lower()
        stat_content = str(triage.get("statistical_content", "")).lower()
        style = str(triage.get("methodological_style", "")).lower()

        lenses: list[tuple[str, str]] = [("prose", prompts.PROSE_VERIFIER_SYSTEM)]
        if (
            math_density in ("light", "moderate", "heavy")
            or "theorem" in style
            or math_density in ("moderate", "heavy")
        ):
            lenses.append(("math", prompts.MATH_VERIFIER_SYSTEM))
        if stat_content not in ("", "none") or "empirical" in style or "causal" in stat_content:
            lenses.append(("empirical", prompts.EMPIRICAL_VERIFIER_SYSTEM))
        if stat_content not in ("", "none") or "empirical" in style:
            lenses.append(("stats", prompts.STATS_VERIFIER_SYSTEM))
        # literature coherence + clarity always run: cheap and central to the review
        lenses.append(("lit", prompts.LIT_VERIFIER_SYSTEM))
        lenses.append(("clarity", prompts.CLARITY_VERIFIER_SYSTEM))
        return lenses

    async def _verify(
        self,
        pool: RolePool,
        doc: Document,
        pack: ContextPack | None,
        triage: dict,
        claims: list[dict] | None = None,
    ) -> list[dict]:
        from open_referee.pipeline.checkers import run_deterministic_checks
        from open_referee.pipeline.claims import verify_artifacts, verify_claims
        from open_referee.pipeline.recompute import run_recompute

        knobs = self.plan_knobs()
        candidates: list[dict] = []

        # deterministic checkers: cross-references, number consistency, table
        # arithmetic — zero LLM cost, findings stored on the run state
        if not self.state.checker_findings:
            det = run_deterministic_checks(doc)
            self.state.checker_findings = [f.to_dict() for f in det]
            if det:
                kinds: dict[str, int] = {}
                for f in det:
                    kinds[f.check] = kinds.get(f.check, 0) + 1
                await self._emit(
                    Stage.VERIFY,
                    StageStatus.RUNNING,
                    "deterministic checks: "
                    + ", ".join(f"{k}×{v}" for k, v in sorted(kinds.items())),
                )
        if knobs.recompute and not any(
            f.get("check") == "recompute" for f in self.state.checker_findings
        ):
            try:
                rec = await run_recompute(self, pool, doc, self.cfg)
            except Exception as e:  # noqa: BLE001 - checker must never kill a run
                logger.warning("recompute stage failed: %s", e)
                rec = []
            if rec:
                self.state.checker_findings.extend(f.to_dict() for f in rec)
                await self._emit(
                    Stage.VERIFY,
                    StageStatus.RUNNING,
                    f"formula recomputation: {len(rec)} confirmed mismatch(es)",
                )

        # per-claim verification workers (the primary engine)
        if claims:
            await self._emit(Stage.VERIFY, StageStatus.RUNNING, "per-claim verification")
            candidates.extend(await verify_claims(self, pool, doc, claims, self.cfg))
        await self._replan_checkpoint(pool, "verify/claims")

        sections = self._reviewable_sections(doc)
        lenses = self._lenses_for(triage) if knobs.per_section_lenses else []
        sem = asyncio.Semaphore(self.cfg.review.max_parallel_calls)

        async def run_lens_section(lens_name: str, system: str, title: str, text: str):
            async with sem:
                if not self.budget_allows("lens_section"):
                    self._note_gate_stop("lens_section")
                    return []
                user = prompts.VERIFIER_USER.format(
                    section_title=f"{title} — {lens_name} lens",
                    context_block=self._pack_for_section(pack),
                    section_text=text[:40_000],
                )
                try:
                    res = await pool.complete_json(
                        ModelRole.SMALL, system, user, call_type="lens_section"
                    )
                except LLMError as e:
                    logger.warning("verifier (%s) failed on section %s: %s", lens_name, title, e)
                    return []
                await self._emit(
                    Stage.VERIFY, StageStatus.RUNNING, f"verified [{lens_name}]: {title}"
                )
                return res.get("comments", [])

        tasks = [
            run_lens_section(lens, system, t, txt)
            for (t, txt) in sections
            for (lens, system) in lenses
        ]
        results = await asyncio.gather(*tasks)
        for r in results:
            candidates.extend(r)
            if r and pool.harness is not None:
                pool.harness.note_yield(
                    "lens_section", len(r), max((float(c.get("score", 0)) for c in r), default=0.0)
                )
        await self._replan_checkpoint(pool, "verify/lenses")

        # artifact verification: ALL figures and ALL tables vs prose (vision role)
        if ModelRole.VISION in pool.providers and (doc.figures or doc.tables):
            await self._emit(Stage.VERIFY, StageStatus.RUNNING, "artifact verification")
            art_comments = await verify_artifacts(self, pool, doc, self.cfg)
            candidates.extend(art_comments)
            if art_comments and pool.harness is not None:
                pool.harness.note_yield("artifact_verify", len(art_comments), 0.5)
        await self._replan_checkpoint(pool, "verify/artifacts")

        # whole-paper pass: cross-section coherence (strong model, full text)
        if not knobs.whole_paper_passes:
            return candidates
        try:
            await self._emit(Stage.VERIFY, StageStatus.RUNNING, "whole-paper coherence pass")
            wp_user = prompts.WHOLE_PAPER_VERIFIER_USER.format(
                title=doc.title, manuscript=self._manuscript_trunc(doc)
            )
            wp = await pool.complete_json(
                ModelRole.STRONG,
                prompts.WHOLE_PAPER_VERIFIER_SYSTEM,
                wp_user,
                call_type="whole_paper_verify",
            )
            wp_comments = wp.get("comments", [])
            candidates.extend(wp_comments)
            if wp_comments and pool.harness is not None:
                pool.harness.note_yield(
                    "whole_paper_verify",
                    len(wp_comments),
                    max((float(c.get("score", 0)) for c in wp_comments), default=0.0),
                )
            await self._emit(
                Stage.VERIFY,
                StageStatus.RUNNING,
                f"whole-paper pass: {len(wp.get('comments', []))} findings",
            )
        except LLMError as e:
            logger.warning("whole-paper verifier failed: %s", e)
        return candidates

    async def _challenge(self, pool, doc, pack, candidates: list[dict]) -> list[dict]:
        sections = self._reviewable_sections(doc)
        sem = asyncio.Semaphore(self.cfg.review.max_parallel_calls)
        validated: list[dict] = []

        async def run_section(title: str, text: str):
            async with sem:
                if not self.budget_allows("section_challenger"):
                    self._note_gate_stop("section_challenger")
                    # challengers are validators: pass candidates through
                    # unvalidated rather than dropping them
                    return [
                        c
                        for c in candidates
                        if c.get("paragraph_anchor", "") and _anchor_in(c["paragraph_anchor"], text)
                    ]
                cands = [
                    c
                    for c in candidates
                    if c.get("paragraph_anchor", "") and _anchor_in(c["paragraph_anchor"], text)
                ]
                user = (
                    prompts.CHALLENGER_USER.format(
                        section_title=title,
                        context_block=self._pack_for_section(pack),
                        section_text=text[:40_000],
                        candidates_json=json.dumps(cands, indent=1)[:20_000],
                    )
                    + self._venue_block
                )
                try:
                    res = await pool.complete_json(
                        ModelRole.STRONG,
                        prompts.CHALLENGER_SYSTEM,
                        user,
                        call_type="section_challenger",
                    )
                except LLMError as e:
                    logger.warning("challenger failed on %s: %s", title, e)
                    return cands
                kept = [
                    c
                    for c in res.get("validated", [])
                    if isinstance(c, dict) and c.get("verdict") != "dropped"
                ]
                return kept + [c for c in res.get("new_comments", []) if isinstance(c, dict)]

        tasks = [run_section(t, txt) for t, txt in sections]
        results = await asyncio.gather(*tasks)
        for r in results:
            validated.extend(r)

        # whole-paper challenge: validate cross-section candidates (those no
        # section challenger claimed) and hunt missed global weaknesses
        from open_referee.pipeline.claims import defense_round

        try:
            claimed = {
                _norm(c.get("paragraph_anchor", ""))[:120]
                for r in results
                for c in (r if isinstance(r, list) else [])
                if isinstance(c, dict)
            }
            wp_candidates = [
                c
                for c in candidates
                if isinstance(c, dict) and _norm(c.get("paragraph_anchor", ""))[:120] not in claimed
            ]
            if not self.budget_allows("whole_paper_challenge"):
                self._note_gate_stop("whole_paper_challenge")
                validated.extend(wp_candidates)  # pass through unvalidated
                return await defense_round(self, pool, doc, validated, self.cfg)
            await self._emit(Stage.CHALLENGE, StageStatus.RUNNING, "whole-paper challenge")
            wp_user = (
                prompts.WHOLE_PAPER_CHALLENGER_USER.format(
                    title=doc.title,
                    manuscript=self._manuscript_trunc(doc),
                    candidates_json=json.dumps(wp_candidates, indent=1)[:30_000],
                )
                + self._venue_block
            )
            wp = await pool.complete_json(
                ModelRole.STRONG,
                prompts.WHOLE_PAPER_CHALLENGER_SYSTEM,
                wp_user,
                call_type="whole_paper_challenge",
            )
            kept = [
                c
                for c in wp.get("validated", [])
                if isinstance(c, dict) and c.get("verdict") != "dropped"
            ]
            validated.extend(kept + [c for c in wp.get("new_comments", []) if isinstance(c, dict)])
            await self._emit(Stage.CHALLENGE, StageStatus.RUNNING, "whole-paper challenge done")
        except LLMError as e:
            logger.warning("whole-paper challenger failed: %s", e)

        # defense-adjudication quality gate (per comment, standard/deep only)
        validated = await defense_round(self, pool, doc, validated, self.cfg)
        return validated

    async def _bibliography(self, pool, doc, pack) -> dict:
        empty = {
            "entries": [],
            "in_text_issues": [],
            "missing_key_references": [],
        }
        if not doc.references_text:
            return empty
        resolution = await self._resolve_references(doc)
        intext = _extract_intext_citations(doc)
        user = prompts.BIBLIOGRAPHY_USER.format(
            references=doc.references_text[:30_000],
            intext=intext[:15_000],
            resolution=resolution,
            context=pack.render(max_items=25),
        )
        try:
            result = await pool.complete_json(
                ModelRole.SMALL, prompts.BIBLIOGRAPHY_SYSTEM, user, call_type="bibliography"
            )
            result.setdefault("in_text_issues", [])
            return result
        except LLMError as e:
            logger.warning("bibliography audit failed: %s", e)
            return empty

    async def _resolve_references(self, doc: Document) -> str:
        """Free deterministic Crossref/OpenAlex pass over the reference list.

        Results are stored on the run state as ground truth and rendered into
        the bibliography prompt; unmatched entries become phantom-candidate
        findings in the checker stream."""
        if self.state.doi_resolutions:
            return _render_resolutions(self.state.doi_resolutions)
        from open_referee.literature.reference_resolution import resolve_references

        try:
            records = await resolve_references(doc.references_text or "", self.cfg)
        except Exception as e:  # noqa: BLE001 - resolution is best-effort
            logger.warning("reference resolution failed: %s", e)
            records = []
        self.state.doi_resolutions = records
        unmatched = [r for r in records if r.get("status") != "matched"]
        if unmatched:
            from open_referee.pipeline.checkers import CheckFinding

            for r in unmatched[:20]:
                self.state.checker_findings.append(
                    CheckFinding(
                        check="doi-resolution",
                        title="Reference not resolvable in Crossref/OpenAlex",
                        message=(
                            "The bibliography entry below could not be matched to "
                            "any Crossref or OpenAlex record (best score "
                            f"{r.get('score', 0)}). It may be phantom, garbled, or "
                            "use an unusual rendering — verify it exists."
                        ),
                        quote=str(r.get("ref", ""))[:240],
                        category="reference",
                        severity=0.4,
                        meta={"kind": "unmatched", "score": r.get("score", 0)},
                    ).to_dict()
                )
        return _render_resolutions(records)

    def _checker_comments(self) -> list[dict]:
        """Deterministic checker findings as candidate comments: they join the
        stream at meta (after defense, before validation) with their
        provenance kept — mechanical findings skip the author-defense gate."""
        out: list[dict] = []
        for f in getattr(self.state, "checker_findings", None) or []:
            try:
                sev = min(0.95, max(0.1, float(f.get("severity", 0.5))))
            except (TypeError, ValueError):
                sev = 0.5
            out.append(
                {
                    "title": str(f.get("title", "Finding"))[:160],
                    "message": str(f.get("message", ""))[:2500],
                    "paragraph_anchor": str(f.get("quote", ""))[:400],
                    "quote": str(f.get("quote", ""))[:240],
                    "score": sev,
                    "category": str(f.get("category", "consistency")),
                    "provenance": f.get("provenance", "deterministic"),
                }
            )
        return out

    async def _meta(self, pool, doc, triage, validated, bib, pack) -> dict:
        bib_summary = {
            "entries": bib.get("entries", []),
            "in_text_issues": bib.get("in_text_issues", []),
        }
        merged = list(validated) + self._checker_comments()
        user = (
            prompts.META_USER.format(
                title=doc.title,
                triage_json=json.dumps(triage)[:4000],
                comments_json=json.dumps(merged, indent=1)[:100_000],
                bib_summary=json.dumps(bib_summary)[:8000],
                sota_notes=(pack.field_hint or "")[:1500],
            )
            + self._venue_block
        )
        draft = await pool.complete_json(
            ModelRole.STRONG, prompts.META_SYSTEM, user, call_type="meta"
        )
        # synthesis: the lead-referee essay over the deduplicated comments,
        # replacing the merger's overall feedback with argued prose
        if getattr(self.plan_knobs(), "synthesis", True) and self.budget_allows("synthesis"):
            essay = await self._synthesis(pool, doc, triage, draft, bib, pack)
            if essay:
                draft["overall_feedback"] = essay
        return draft

    async def _synthesis(self, pool, doc, triage, draft, bib, pack) -> str | None:
        comments = [c for c in draft.get("comments", []) if isinstance(c, dict)]
        listing = "\n".join(
            f"- [{float(c.get('score', 0) or 0):.2f}] "
            f"({c.get('provenance', 'llm')}) {str(c.get('title', ''))[:110]} "
            f":: {str(c.get('paragraph_anchor', ''))[:80]}"
            for c in comments[:120]
        )
        bib_summary = json.dumps(
            {
                "entries": [
                    {"raw": e.get("raw_entry", ""), "status": e.get("status", "")}
                    for e in bib.get("entries", [])
                    if isinstance(e, dict)
                ][:40],
                "in_text_issues": bib.get("in_text_issues", [])[:20],
            }
        )[:8000]
        from open_referee.pipeline.budget import call_max_tokens

        try:
            res = await pool.complete_json(
                ModelRole.STRONG,
                prompts.SYNTHESIS_SYSTEM,
                prompts.SYNTHESIS_USER.format(
                    title=doc.title,
                    triage_json=json.dumps(triage)[:3000],
                    comments_listing=listing or "(no validated comments)",
                    bib_summary=bib_summary,
                    sota_notes=(pack.field_hint or "")[:1500],
                    venue_block=self._venue_block,
                ),
                call_type="synthesis",
                max_tokens=call_max_tokens(
                    "synthesis", str(getattr(self.plan_knobs(), "output_budget", "default"))
                ),
            )
        except LLMError as e:
            logger.warning("synthesis pass failed: %s", e)
            return None
        essay = str(res.get("overall_feedback", "")).strip()
        return essay or None

    async def _dm_validator_gate(self, pool, doc, comments: list[dict]) -> list[dict]:
        """Batched predicate per comment: is it grounded in the manuscript?
        Ungrounded comments are dropped before the (strong) validator pass.
        Batches of 60 questions per decision call, over ALL comments."""
        if getattr(pool, "decider", None) is None or not comments:
            return comments
        from open_referee.providers.decisions import PredicateQuestion

        kept: list[dict] = []
        dropped = 0
        for lo in range(0, len(comments), 60):
            chunk = comments[lo : lo + 60]
            listing = "\n".join(
                f"{i}. [{c.get('title', '')[:60]}] quote={(c.get('quote') or '')[:100]} "
                f"anchor={(c.get('paragraph_anchor') or '')[:80]}"
                for i, c in enumerate(chunk, 1)
            )
            qs = [
                PredicateQuestion(
                    name=f"comment_{i}",
                    instructions=(
                        f"Review comment {i} quotes or paraphrases text that actually "
                        "appears in the manuscript (not invented)."
                    ),
                )
                for i in range(1, len(chunk) + 1)
            ]
            try:
                dm = await pool.decide(
                    f"Manuscript: {doc.title}\n\nComments:\n{listing}", qs,
                    call_type="dm_validate",
                )
            except LLMError as e:
                logger.warning("DM validator gate failed (%s); keeping chunk", e)
                kept.extend(chunk)
                continue
            for i, c in enumerate(chunk, 1):
                a = dm.answer(f"comment_{i}")
                if a is None or a.probability is None or a.probability >= 0.5:
                    kept.append(c)
                else:
                    dropped += 1
        if dropped:
            await self._emit(
                Stage.VALIDATE,
                StageStatus.RUNNING,
                f"decision gate dropped {dropped} ungrounded comments before validation",
            )
        return kept

    async def _validate(self, pool, doc, draft: dict) -> dict:
        gated = await self._dm_validator_gate(pool, doc, list(draft.get("comments", [])))
        draft = dict(draft, comments=gated)
        user = (
            prompts.VALIDATOR_USER.format(
                title=doc.title,
                manuscript=self._manuscript_trunc(doc),
                comments_json=json.dumps(draft.get("comments", []), indent=1)[:60_000],
                overall=draft.get("overall_feedback", "")[:20_000],
            )
            + self._venue_block
        )
        result = await pool.complete_json(
            ModelRole.STRONG, prompts.VALIDATOR_SYSTEM, user, call_type="validate"
        )
        # merge any fields the validator omitted from the draft
        result.setdefault("paper_summary", draft.get("paper_summary", ""))
        result.setdefault("overall_feedback", draft.get("overall_feedback", ""))
        result.setdefault("comments", draft.get("comments", []))
        return result

    async def _assemble(self, final: dict, doc: Document) -> ReviewReport:
        comments: list[ReviewComment] = []
        for i, c in enumerate(final.get("comments", [])):
            try:
                score = max(0.0, min(1.0, float(c.get("score", 0.3))))
            except (TypeError, ValueError):
                score = 0.3
            anchor = c.get("paragraph_anchor") or ""
            quote = c.get("quote") or ""
            block_id, confidence = _resolve_anchor(doc, anchor, quote)
            comments.append(
                ReviewComment(
                    id=f"c{i:03d}",
                    title=c.get("title") or "Untitled comment",
                    section=_section_of_block(doc, block_id),
                    block_id=block_id,
                    paragraph_anchor=anchor or None,
                    quote=quote or None,
                    message=c.get("message") or "",
                    score=score,
                    category=c.get("category") or "general",
                    anchor_confidence=confidence,
                )
            )
        u = self._pool.ledger if self._pool else None
        by_role = u.by_role() if u else {}
        usage = UsageSummary(
            tokens_by_role=by_role,
            estimated_cost_usd=round(u.total_cost(), 4) if u else 0.0,
            total_input_tokens=int(u.total_tokens()[0]) if u else 0,
            total_output_tokens=int(u.total_tokens()[1]) if u else 0,
        )
        plan_summary = self._build_plan_summary()
        report = ReviewReport(
            run_id=self.run_id,
            paper_title=doc.title,
            paper_summary=final.get("paper_summary", ""),
            overall_feedback=final.get("overall_feedback", ""),
            comments=comments,
            usage=usage,
            model_roles={
                r.value: f"{p.provider_name}/{p.model}"
                for r, p in (self._pool.specs if self._pool else {}).items()
            },
            validator_notes=final.get("validator_notes"),
            plan_summary=plan_summary,
        )
        # persist report + exports into the run dir
        rdir = _runs_dir() / self.run_id
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "report.json").write_text(report.model_dump_json(indent=2))
        (rdir / "report.md").write_text(render_markdown(report))
        try:
            render_pdf(report, rdir / "report.pdf")
        except Exception as e:  # PDF export must never fail the run
            logger.warning("PDF export failed: %s", e)
        return report


# ------------------------------------------------------------------ anchor --


def _extract_intext_citations(doc: Document, max_citations: int = 60) -> str:
    """Extract in-text citation mentions with claim context for the audit.

    Matches the common author-year styles: (Smith, 2001), Smith (2001),
    (Smith and Jones, 2001; Doe, 2019), [1], [Smith2001]. For each match,
    keeps the surrounding sentence as the claim context.
    """
    text = doc.full_text()
    patterns = [
        # Author (1999) / Author and Other (1999) — runs FIRST so it claims the
        # full span before the bare-paren pattern can reduce it to "(1999)"
        r"([A-Z][A-Za-z'’\-]+(?:\s+(?:and|&|et al\.?)\s+[A-Z][A-Za-z'’\-]+)"
        r"{0,2}\s*\(\d{4}[a-z]?\))",
        # numeric [1] / [1, 2] / [1-3]
        r"(\[[0-9][0-9,\s\-–]*\])",
        # (Author, 1999) / (Author and Other, 1999; Third, 2020)
        r"\(([^()]{0,80}?\d{4}[a-z]?[^()]{0,80}?)\)",
        # natbib \citep-style leftovers: Smith2001 / Jones2020
        r"\b([A-Z][a-z]{2,}\d{2,4})\b",
    ]
    hits: list[tuple[str, str]] = []
    seen_spans: list[tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(s < be and e > bs for bs, be in seen_spans)

    for pat in patterns:
        for m in re.finditer(pat, text):
            if overlaps(m.start(), m.end()):
                continue
            # must contain either a year or a bracket number to count as a citation
            cite = m.group(0)
            if not (re.search(r"\d{4}", cite) or cite.startswith("[")):
                continue
            sent_lo = max(0, text.rfind(". ", 0, m.start()) + 1)
            sent_hi = text.find(". ", m.end())
            if sent_hi == -1:
                sent_hi = min(len(text), m.end() + 200)
            claim = " ".join(text[sent_lo:sent_hi].split())[:400]
            hits.append((cite, claim))
            seen_spans.append((m.start(), m.end()))
            if len(hits) >= max_citations:
                break
        if len(hits) >= max_citations:
            break
    if not hits:
        return "(no in-text citations detected)"
    return "\n".join(f"- {cite} :: {claim}" for cite, claim in hits)


def _figure_context(doc: Document, fig: Figure, window: int = 30) -> str:
    """Text around a figure: prefers the caption block, else blocks near the
    figure's page (approximated by proportional position in the block list)."""
    if fig.caption_block_id:
        for i, b in enumerate(doc.blocks):
            if b.id == fig.caption_block_id:
                lo, hi = max(0, i - window // 2), min(len(doc.blocks), i + window // 2)
                return "\n".join(b.text for b in doc.blocks[lo:hi])
    if len(doc.blocks) <= window:
        return doc.full_text()
    # pages are 1-based; approximate block index proportionally
    if fig.page:
        frac = (fig.page - 1) / max(1, fig.page)  # crude; falls back to middle
        center = min(len(doc.blocks) - window, int(frac * (len(doc.blocks) - window)))
        center = max(0, center)
    else:
        center = len(doc.blocks) // 2
    return "\n".join(b.text for b in doc.blocks[center : center + window])


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def _anchor_in(anchor: str, text: str) -> bool:
    return _norm(anchor)[:120] in _norm(text)


def _resolve_anchor(doc: Document, anchor: str, quote: str) -> tuple[str | None, float]:
    """Fuzzy-resolve a comment to a block. Returns (block_id, confidence)."""
    if anchor:
        hits = doc.find_blocks(anchor)
        if len(hits) == 1:
            return hits[0].id, 1.0
        if hits:
            return hits[0].id, 0.7
    if quote:
        hits = doc.find_blocks(quote)
        if hits:
            return hits[0].id, 0.6
    if anchor:
        # token-overlap fallback
        a_tokens = set(_norm(anchor).split())
        best, best_j = None, 0.0
        for b in doc.blocks:
            tokens = set(_norm(b.text).split())
            if not tokens:
                continue
            j = len(a_tokens & tokens) / max(1, len(a_tokens | tokens))
            if j > best_j:
                best, best_j = b, j
        if best is not None and best_j > 0.35:
            return best.id, round(best_j, 2)
    return None, 0.0


def _section_of_block(doc: Document, block_id: str | None) -> str | None:
    if not block_id:
        return None
    for b in doc.blocks:
        if b.id == block_id:
            return b.section_path[0] if b.section_path else None
    return None


_FALLBACK_MD_RE = re.compile(r"^#{1,3}\s+(\S.*)$")  # markdown
_FALLBACK_NUM_RE = re.compile(
    r"^(?:\d{1,2}(?:\.\d+)*\.?|Appendix\s+[A-Z])\s+[A-Z].{2,120}$"
)  # numbered section lines
_FALLBACK_KW_RE = re.compile(
    r"^(?:abstract|references|bibliography|appendix|conclusions?)\s*$", re.I
)
_FALLBACK_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9 :&\-]{3,60}$")
_FALLBACK_MIN_CHARS = 400
_FALLBACK_MAX_SECTIONS = 20


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split full text into (title, chunk) pairs when no sections were parsed.

    Detects markdown headings, numbered section lines ('2.3 Methods'), keyword
    headings (Abstract/References/...) and short ALL-CAPS title lines; skips
    author-list style caps lines (commas, long). Chunks below the minimum
    length merge into the previous section."""
    lines = text.split("\n")
    boundaries: list[tuple[int, str]] = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if len(s) < 3 or len(s) > 130:
            continue
        if m := _FALLBACK_MD_RE.match(s):
            boundaries.append((i, m.group(1).strip()))
            continue
        if _FALLBACK_NUM_RE.match(s):
            boundaries.append((i, s))
            continue
        if _FALLBACK_KW_RE.match(s):
            boundaries.append((i, s.title()))
            continue
        if _FALLBACK_CAPS_RE.match(s) and "," not in s and len(s.split()) <= 8:
            boundaries.append((i, s))
    if not boundaries:
        return []

    sections: list[list[str]] = []
    titles: list[str] = []
    if boundaries[0][0] > 0:
        sections.append(lines[: boundaries[0][0]])
        titles.append("Front matter")
    for bi, (start, title) in enumerate(boundaries):
        end = boundaries[bi + 1][0] if bi + 1 < len(boundaries) else len(lines)
        sections.append(lines[start:end])
        titles.append(title)

    out: list[tuple[str, str]] = []
    for title, chunk_lines in zip(titles, sections, strict=True):
        chunk = "\n".join(chunk_lines).strip()
        if not chunk:
            continue
        if len(chunk) < _FALLBACK_MIN_CHARS and out:
            out[-1] = (out[-1][0], out[-1][1] + "\n\n" + chunk)
        else:
            out.append((title, chunk))
    while len(out) > _FALLBACK_MAX_SECTIONS:
        last = out.pop()
        out[-1] = (out[-1][0], out[-1][1] + "\n\n" + last[1])
    return out


def _hit_to_item(hit, why: str):
    from open_referee.literature.context import LiteratureItem

    return LiteratureItem(
        source="web", title=hit.title, url=hit.url, abstract=hit.snippet, why_relevant=why
    )


def _render_resolutions(records: list[dict]) -> str:
    if not records:
        return "(resolution unavailable)"
    lines = []
    for r in records:
        if r.get("status") == "matched":
            lines.append(
                f"- MATCHED {r.get('doi') or 'no-doi'} :: {str(r.get('title'))[:110]} "
                f"({r.get('venue') or '?'}, {r.get('year') or '?'}) [score {r.get('score')}]"
            )
        else:
            lines.append(f"- UNMATCHED :: {str(r.get('ref'))[:110]} [best score {r.get('score')}]")
    return "\n".join(lines)[:20_000]
