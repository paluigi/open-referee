"""Journal-fit / desk-reject estimator: a standalone, cheap product.

Two-step protocol (project-owner decision): the chat LLM writes the editor
pass FIRST — six factor analyses (each hard-capped for brevity) plus overall
reasoning — and the DECISION MODEL then takes the venue profile plus those
factor texts as input and directly scores the desk-screen predicate. The
final number is the decision model's probability; the LLM never produces it.
Without a decision role configured, the LLM's own holistic estimate is used
and the report says so.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from open_referee.config import Config, ModelRole
from open_referee.ingestion import ingest_document
from open_referee.pipeline import prompts_fit
from open_referee.pipeline.budget import (
    BudgetHarness,
    BudgetInfeasible,
    DocumentStats,
    PricingError,
)
from open_referee.pipeline.state import RunState, Stage, StageStatus
from open_referee.providers import LLMError
from open_referee.providers.decisions import PredicateQuestion
from open_referee.report.models import (
    FitDecision,
    FitFactor,
    FitReport,
    PlanSummary,
    UsageSummary,
)
from open_referee.report.render import render_fit_markdown, render_fit_pdf

logger = logging.getLogger(__name__)

MAX_MANUSCRIPT_CHARS = 120_000
# hard character guard per factor analysis (~600 tokens x 4 chars/token)
FACTOR_CHAR_CAP = prompts_fit.FACTOR_TOKEN_CAP * 4


def _runs_dir() -> Path:
    from open_referee.config import DEFAULT_CONFIG_DIR

    return DEFAULT_CONFIG_DIR / "runs"


class FitPipeline:
    """Standalone desk-reject estimator. Reuses ingestion + RolePool + ledger."""

    def __init__(self, cfg: Config, run_id: str | None = None, budget_usd: float | None = None):
        self.cfg = cfg
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.requested_budget_usd = budget_usd
        self.state: RunState | None = None
        self._subscribers: list[asyncio.Queue] = []
        self._pool = None

    # ---------------------------------------------------------------- events --

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    async def _emit(
        self, stage: Stage, status: StageStatus, message: str = "", data: dict | None = None
    ):
        from open_referee.pipeline.state import StageEvent

        ev = StageEvent(
            run_id=self.run_id, stage=stage, status=status, message=message, data=data or {}
        )
        for q in self._subscribers:
            await q.put(ev)
        if self.state is not None:
            self.state.events.append(ev)

    # ------------------------------------------------------------------- run --

    async def run(self, paper_path: str | Path, journal_profile: str) -> FitReport:
        journal_profile = (journal_profile or "").strip()
        if not journal_profile:
            raise ValueError("A journal profile is required for the fit estimate.")
        state = RunState(run_id=self.run_id, paper_path=str(paper_path))
        self.state = state
        state.run_type = "fit"
        state.journal_profile = journal_profile
        state.budget_usd = self.requested_budget_usd
        state.save()
        try:
            from open_referee.pipeline.orchestrator import RolePool

            budget = (
                self.requested_budget_usd
                if self.requested_budget_usd is not None
                else self.cfg.review.max_cost_usd
            )
            fit_roles = {ModelRole.STRONG}
            try:
                self.cfg.llm.role(ModelRole.DECISION)
                fit_roles.add(ModelRole.DECISION)
            except KeyError:
                pass  # fallback mode: the editor LLM also estimates probability
            pool = RolePool.build(self.cfg, fit_roles, cap_usd=budget)
            self._pool = pool
            try:
                # 1. ingest (free)
                await self._emit(Stage.INGEST, StageStatus.RUNNING)
                doc = await asyncio.to_thread(ingest_document, paper_path)
                state.paper_title = doc.title
                state.set_stage(Stage.INGEST, StageStatus.DONE)
                state.save()
                await self._emit(Stage.INGEST, StageStatus.DONE, doc.title)

                # 2. budget preflight: pricing is mandatory; budget must cover the calls
                harness = BudgetHarness(
                    self.cfg, pool.ledger, pool.specs, budget_usd=budget, auto=True
                )
                harness.stats = DocumentStats.from_doc(doc)
                if pool.decider_spec is not None:
                    harness.specs[ModelRole.DECISION] = pool.decider_spec
                pool.harness = harness
                harness.require_pricing()  # PricingError if unpriced
                est = harness.unit_cost("fit_assessment")
                if budget < est:
                    raise BudgetInfeasible(
                        f"One fit assessment of this manuscript costs about ${est:.2f}, "
                        f"but the budget is ${budget:.2f}.",
                        min_budget_usd=round(est, 2),
                        standard_budget_usd=round(est, 2),
                    )

                # 3a. EDITOR PASS (strong LLM, first): six capped factor analyses
                await self._emit(Stage.FIT, StageStatus.RUNNING, "editor pass: factor analyses")
                user = prompts_fit.editor_user(
                    journal_profile,
                    doc.title,
                    doc.full_text()[:MAX_MANUSCRIPT_CHARS],
                    fallback=pool.decider is None,
                )
                res = await pool.complete_json(
                    ModelRole.STRONG, prompts_fit.EDITOR_SYSTEM, user, call_type="fit_assessment"
                )

                # 3b. DECISION PASS: the verdict over profile + factor texts
                decision = await self._decision_pass(pool, journal_profile, res)
                state.ledger_calls = [c.model_dump(mode="json") for c in pool.ledger.calls]

                state.artifacts["fit"] = {
                    k: res.get(k) for k in ("factors", "risks", "suggestions")
                }
                state.set_stage(Stage.FIT, StageStatus.DONE)
                state.save()
                await self._emit(Stage.FIT, StageStatus.DONE, "fit assessment done")

                # 4. assemble + persist
                report = self._assemble(res, decision, doc.title, journal_profile, harness)
                state.status = "completed"
                state.save()
                self._record_history("completed")
                await self._emit(Stage.ASSEMBLE, StageStatus.DONE, "fit report ready")
                return report
            finally:
                await pool.close()
        except (PricingError, BudgetInfeasible) as e:
            if state:
                state.status = "failed"
                state.error = str(e)
                state.save()
            self._record_history("failed", str(e))
            await self._emit(Stage.ASSEMBLE, StageStatus.FAILED, f"cannot run: {e}")
            raise
        except Exception as e:
            if state:
                state.status = "failed"
                state.error = str(e)
                state.save()
            self._record_history("failed", str(e))
            logger.exception("fit pipeline failed")
            raise

    async def _decision_pass(self, pool, journal_profile: str, editor: dict) -> FitDecision:
        """The decision model judges the LLM's analyses; without one, the
        LLM's own holistic estimate is the fallback (and is labelled as such)."""
        if getattr(pool, "decider", None) is not None:
            factor_texts = []
            for f in editor.get("factors", []) or []:
                if isinstance(f, dict) and f.get("name"):
                    label = prompts_fit.FACTOR_LABELS.get(str(f.get("name")), str(f.get("name")))
                    factor_texts.append(
                        f"## {label}\n{str(f.get('analysis', ''))[:FACTOR_CHAR_CAP]}"
                    )
            dm_input = (
                f"## Venue profile\n\n{journal_profile[:12_000]}\n\n"
                "## Editor factor analyses\n\n" + "\n\n".join(factor_texts)
            )
            try:
                result = await pool.decide(
                    dm_input,
                    [
                        PredicateQuestion(
                            name="desk_screen",
                            instructions=prompts_fit.DECISION_PREDICATE,
                        )
                    ],
                    call_type="dm_fit",
                )
                a = result.answer("desk_screen")
                if a is not None and a.probability is not None:
                    return FitDecision(
                        probability=a.probability,
                        confidence=a.confidence,
                        question=prompts_fit.DECISION_PREDICATE,
                        source="decision-model",
                    )
            except LLMError as e:
                logger.warning("decision pass failed (%s); falling back to LLM estimate", e)
        # fallback: the editor's own estimate (or neutral 0.5)
        try:
            prob = float(editor.get("llm_estimate", 0.5))
        except (TypeError, ValueError):
            prob = 0.5
        return FitDecision(
            probability=max(0.0, min(1.0, prob)),
            question=prompts_fit.DECISION_PREDICATE,
            source="llm-fallback",
        )

    # -------------------------------------------------------------- assemble --

    def _assemble(
        self,
        res: dict,
        decision: FitDecision,
        title: str,
        journal_profile: str,
        harness: BudgetHarness,
    ) -> FitReport:
        factors: list[FitFactor] = []
        seen: set[str] = set()
        for f in res.get("factors", []) or []:
            if not isinstance(f, dict):
                continue
            name = str(f.get("name", "")).strip().lower()
            if name not in prompts_fit.FACTOR_LABELS or name in seen:
                continue
            seen.add(name)
            factors.append(
                FitFactor(
                    name=name,
                    label=prompts_fit.FACTOR_LABELS[name],
                    analysis=str(f.get("analysis", ""))[:FACTOR_CHAR_CAP],
                )
            )
        factors.sort(key=lambda f: prompts_fit.FACTOR_ORDER.index(f.name))
        for name in prompts_fit.FACTOR_ORDER:
            if name not in seen:
                factors.append(
                    FitFactor(
                        name=name,
                        label=prompts_fit.FACTOR_LABELS[name],
                        analysis="(not assessed by the editor)",
                    )
                )
        u = self._pool.ledger
        usage = UsageSummary(
            tokens_by_role=u.by_role(),
            estimated_cost_usd=round(u.total_cost(), 4),
            total_input_tokens=int(u.total_tokens()[0]),
            total_output_tokens=int(u.total_tokens()[1]),
        )
        st = harness.status()
        report = FitReport(
            run_id=self.run_id,
            paper_title=title,
            journal_profile=journal_profile,
            decision=decision,
            probability_not_desk_rejected=round(decision.probability, 4),
            factors=factors,
            reasoning=res.get("reasoning", ""),
            risks=[str(r) for r in res.get("risks", []) or []],
            suggestions=[str(s) for s in res.get("suggestions", []) or []],
            usage=usage,
            model_roles={
                r.value: f"{p.provider_name}/{p.model}" for r, p in self._pool.specs.items()
            },
            plan_summary=PlanSummary(
                mode="fit",
                budget_usd=st["budget_usd"],
                spent_usd=st["actual_usd"],
                control_spent_usd=st["control_actual_usd"],
                projected_final_usd=st["projected_final_usd"],
            ),
        )
        if self._pool.decider is not None:
            report.model_roles["decision"] = (
                f"{self._pool.decider.provider_name}/{self._pool.decider.model}"
            )
        rdir = _runs_dir() / self.run_id
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "report.json").write_text(report.model_dump_json(indent=2))
        (rdir / "report.md").write_text(render_fit_markdown(report))
        try:
            render_fit_pdf(report, rdir / "report.pdf")
        except Exception as e:  # PDF export must never fail the run
            logger.warning("PDF export failed: %s", e)
        return report

    def _record_history(self, status: str, error: str | None = None) -> None:
        """Append the per-phase estimation record to the run-history store.
        Best-effort: a history failure must never break the run itself."""
        try:
            from open_referee.pipeline.history import build_run_record, upsert_run_record

            pool = getattr(self, "_pool", None)
            if pool is None or pool.harness is None:
                return
            rec = build_run_record(
                run_id=self.run_id,
                run_type="fit",
                status=status,
                harness=pool.harness,
                events=self.state.events if self.state is not None else [],
                paper_title=self.state.paper_title if self.state is not None else "",
                depth_mode="fit",
                error=error,
                models={r.value: f"{p.provider_name}/{p.model}" for r, p in pool.specs.items()},
            )
            upsert_run_record(rec)
        except Exception:
            logger.exception("failed to record run history")
