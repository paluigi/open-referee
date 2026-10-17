"""Journal-fit / desk-reject estimator: a standalone, cheap product.

One strong-model call compares the manuscript against a free-form venue
profile (aim/scope/metrics + any institutional considerations), producing
structured factor scores with evidence. A transparent deterministic
calibration (weighted logistic — weights printed in the report) maps the
factors to the probability of NOT being desk rejected.
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
from open_referee.report.models import FitFactor, FitReport, PlanSummary, UsageSummary
from open_referee.report.render import render_fit_markdown, render_fit_pdf

logger = logging.getLogger(__name__)

MAX_MANUSCRIPT_CHARS = 120_000

# Decision model: P(not desk rejected) = sigmoid(bias + Σ w_i · s_i).
# Weights favor scope/aim — what desk screens actually hinge on. These are
# printed verbatim in every report so institutions see how the number forms.
FIT_WEIGHTS: dict[str, float] = {
    "scope_fit": 2.2,
    "aim_alignment": 1.8,
    "novelty_threshold": 1.4,
    "methods_format_fit": 1.2,
    "presentation_readiness": 0.9,
    "policy_ethics": 1.0,
}
FIT_BIAS = -3.6

FACTOR_LABELS = {
    "scope_fit": "Scope fit",
    "aim_alignment": "Aim alignment",
    "methods_format_fit": "Methods & format fit",
    "novelty_threshold": "Novelty vs venue threshold",
    "presentation_readiness": "Presentation readiness",
    "policy_ethics": "Policy & ethics compliance",
}
FACTOR_ORDER = list(FACTOR_LABELS)


def calibrate_probability(factor_scores: dict[str, float]) -> float:
    """Weighted logistic over the factor scores; missing factors default 0.5."""
    import math

    z = FIT_BIAS
    for name, weight in FIT_WEIGHTS.items():
        s = factor_scores.get(name, 0.5)
        s = max(0.0, min(1.0, float(s)))
        z += weight * s
    return 1.0 / (1.0 + math.exp(-z))


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
            pool = RolePool.build(self.cfg, {ModelRole.STRONG}, cap_usd=budget)
            self._pool = pool
            try:
                # 1. ingest (free)
                await self._emit(Stage.INGEST, StageStatus.RUNNING)
                doc = await asyncio.to_thread(ingest_document, paper_path)
                state.paper_title = doc.title
                state.set_stage(Stage.INGEST, StageStatus.DONE)
                state.save()
                await self._emit(Stage.INGEST, StageStatus.DONE, doc.title)

                # 2. budget preflight: pricing is mandatory; budget must cover the call
                harness = BudgetHarness(
                    self.cfg, pool.ledger, pool.specs, budget_usd=budget, auto=True
                )
                harness.stats = DocumentStats.from_doc(doc)
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

                # 3. the fit assessment — the product's single LLM call
                await self._emit(Stage.FIT, StageStatus.RUNNING, "assessing journal fit")
                user = prompts_fit.fit_user(
                    journal_profile, doc.title, doc.full_text()[:MAX_MANUSCRIPT_CHARS]
                )
                res = await pool.complete_json(
                    ModelRole.STRONG, prompts_fit.FIT_SYSTEM, user, call_type="fit_assessment"
                )
                state.artifacts["fit"] = {
                    k: res.get(k) for k in ("factors", "risks", "suggestions")
                }
                state.set_stage(Stage.FIT, StageStatus.DONE)
                state.save()
                await self._emit(Stage.FIT, StageStatus.DONE, "fit assessment done")

                # 4. assemble + persist
                report = self._assemble(res, doc.title, journal_profile, harness)
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

    # -------------------------------------------------------------- assemble --

    def _assemble(
        self, res: dict, title: str, journal_profile: str, harness: BudgetHarness
    ) -> FitReport:
        scores: dict[str, float] = {}
        factors: list[FitFactor] = []
        for f in res.get("factors", []) or []:
            if not isinstance(f, dict):
                continue
            name = str(f.get("name", "")).strip().lower()
            if name not in FACTOR_LABELS:
                continue
            try:
                score = max(0.0, min(1.0, float(f.get("score", 0.5))))
            except (TypeError, ValueError):
                score = 0.5
            scores[name] = score
            factors.append(
                FitFactor(
                    name=name,
                    label=FACTOR_LABELS[name],
                    score=score,
                    rationale=str(f.get("rationale", "")),
                    evidence=f.get("evidence"),
                )
            )
        # stable order; ensure all six present for the calibration
        factors.sort(key=lambda f: FACTOR_ORDER.index(f.name))
        for name in FACTOR_ORDER:
            if name not in scores:
                scores[name] = 0.5
                factors.append(
                    FitFactor(
                        name=name,
                        label=FACTOR_LABELS[name],
                        score=0.5,
                        rationale="(not assessed — default score applied)",
                    )
                )
        try:
            holistic = res.get("llm_holistic_estimate")
            holistic = max(0.0, min(1.0, float(holistic))) if holistic is not None else None
        except (TypeError, ValueError):
            holistic = None
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
            probability_not_desk_rejected=round(calibrate_probability(scores), 4),
            llm_holistic_estimate=holistic,
            factors=factors,
            calibration={"bias": FIT_BIAS, **FIT_WEIGHTS},
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
        rdir = _runs_dir() / self.run_id
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "report.json").write_text(report.model_dump_json(indent=2))
        (rdir / "report.md").write_text(render_fit_markdown(report))
        try:
            render_fit_pdf(report, rdir / "report.pdf")
        except Exception as e:  # PDF export must never fail the run
            logger.warning("PDF export failed: %s", e)
        return report
