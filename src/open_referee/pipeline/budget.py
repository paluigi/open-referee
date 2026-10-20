"""Budget-driven planning harness: cost model, deterministic planner,
spend tracking, and steering for the review pipeline.

Two cost buckets are accounted separately:

- control plane: triage, claim inventory, surveyor, and the planner LLM
  calls themselves (plus a small replanning contingency reserve);
- review plane: per-claim/lens/artifact verification, challenges, the
  defense gate, bibliography audit, meta and validation.

Every LLM call in the run flows through the UsageLedger; the harness layers
plan-vs-actual tracking, adaptive unit costs, and deterministic steering on
top so that an LLM replanning call is a last resort, not the default
reaction to deviation — replanning costs money too, and that money comes
out of the same budget.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from open_referee.config import Config, ModelRole
from open_referee.providers.usage import CallRecord, UsageLedger

if TYPE_CHECKING:
    from open_referee.ingestion import Document
    from open_referee.providers.base import ModelSpec

# ----------------------------------------------------------------- constants --

CHARS_PER_TOKEN = 3  # matches RolePool's pre-call estimate convention
MAX_MANUSCRIPT_CHARS = 120_000
MAX_PLANNER_CALLS = 4  # 1 strategy call + up to 3 mid-run adjustments
PLANNER_CONTINGENCY_CALLS = 2  # reserve carved out for adjustments
MIN_CLAIMS = 6  # a review below this is not meaningful
CLAIM_VERIFY_STRONG_SHARE = 0.4  # math-ish claim types route to the strong role
OVERRUN_BUDGET_FRACTION = 0.95  # steer when projection exceeds 95% of budget
TARGET_TOLERANCE = 0.20  # the soft target may be exceeded by up to 20% silently
MIN_UNIT_SAMPLES = 3  # empirical unit costs kick in after N actual calls
VISION_TOKENS_PER_ARTIFACT = 4000  # fixed estimate, matches claims.py's gate
MAX_CLAIMS_CAP = 150
MAX_ARTIFACTS_CAP = 50

# call types that belong to the control plane (booked separately in the plan)
CONTROL_TYPES = frozenset({"triage", "claim_inventory", "surveyor", "planner"})

# review-plane call types, in the deterministic planner's quality order
REVIEW_TYPES = (
    "claim_verify",
    "whole_paper_verify",
    "section_challenger",
    "whole_paper_challenge",
    "lens_section",
    "defense",
    "adjudication",
    "artifact_verify",
    "bibliography",
    "meta",
    "validate",
)


class PricingError(RuntimeError):
    """A configured model has no pricing entry; budget planning is impossible."""


class BudgetInfeasible(RuntimeError):
    """The budget cannot buy a meaningful review."""

    def __init__(self, message: str, min_budget_usd: float, standard_budget_usd: float):
        self.min_budget_usd = min_budget_usd
        self.standard_budget_usd = standard_budget_usd
        super().__init__(message)


# --------------------------------------------------------------------- models --


class DocumentStats(BaseModel):
    """Free (no-LLM) statistics driving all cost estimates."""

    chars: int = 0
    n_sections: int = 1
    n_figures: int = 0
    n_tables: int = 0
    n_references: int = 0
    n_theorems: int = 0
    n_claims: int | None = None  # exact count known after the CLAIMS stage
    n_lenses: int = 3  # exact count known after TRIAGE

    @classmethod
    def from_doc(cls, doc: Document) -> DocumentStats:
        n_sections = 0
        if doc.sections:
            for s in doc.sections:
                text = "\n\n".join(b.text for b in doc.blocks[s.start_block : s.end_block])
                if len(text) > 400:
                    n_sections += 1
        refs = doc.references_text or ""
        return cls(
            chars=len(doc.full_text()),
            n_sections=max(1, n_sections),
            n_figures=len(doc.figures),
            n_tables=len(doc.tables),
            n_references=refs.count("\n") + (1 if refs.strip() else 0),
            n_theorems=len(doc.theorems),
        )

    @property
    def known_or_estimated_claims(self) -> int:
        if self.n_claims is not None:
            return self.n_claims
        return max(10, min(MAX_CLAIMS_CAP, self.chars // 2500))


class PlanKnobs(BaseModel):
    """The adjustable dials of a review plan (superset of the depth presets)."""

    max_claims: int = 60
    per_section_lenses: bool = True
    whole_paper_passes: bool = True
    defense_round: bool = True
    max_artifacts: int = 12
    recompute: bool = True  # hybrid formula recomputation (extract->sympy->verify)
    verify_passes: int = 1  # independent verification passes per claim
    claim_routing: str = "base"  # base | strong (numerical->strong) | all_strong
    output_budget: str = "default"  # low | default | high per-call-type budgets
    synthesis: bool = True  # lead-referee essay pass over deduplicated comments
    defense_second_round: bool = False  # second defense pass on upheld comments

    @classmethod
    def from_preset(cls, preset) -> PlanKnobs:
        return cls(
            max_claims=preset.max_claims,
            per_section_lenses=preset.per_section_lenses,
            whole_paper_passes=preset.whole_paper_passes,
            defense_round=preset.defense_round,
            max_artifacts=preset.max_artifacts,
            recompute=getattr(preset, "recompute", True),
            verify_passes=getattr(preset, "verify_passes", 1),
            claim_routing=getattr(preset, "claim_routing", "base"),
            output_budget=getattr(preset, "output_budget", "default"),
            synthesis=getattr(preset, "synthesis", True),
            defense_second_round=getattr(preset, "defense_second_round", False),
        )


# per-call-type output budgets by depth level (tokens); calls not listed use
# the role default. Reasoning-style models need generous budgets or they
# exhaust the output window mid-thought (observed 2-5x underestimates).
OUTPUT_BUDGETS: dict[str, dict[str, int]] = {
    "low": {
        "claim_inventory": 8192,
        "claim_verify": 4096,
        "claim_arbitrate": 2048,
        "recompute_extract": 4096,
        "recompute_verify": 2048,
        "whole_paper_verify": 8192,
        "whole_paper_challenge": 8192,
        "meta": 4096,
        "synthesis": 4096,
        "validate": 4096,
    },
    "default": {
        "claim_inventory": 12288,
        "claim_verify": 6144,
        "claim_arbitrate": 3072,
        "recompute_extract": 4096,
        "recompute_verify": 2048,
        "whole_paper_verify": 12288,
        "whole_paper_challenge": 12288,
        "section_challenger": 4096,
        "meta": 6144,
        "synthesis": 6144,
        "validate": 6144,
    },
    "high": {
        "claim_inventory": 16384,
        "claim_verify": 12288,
        "claim_arbitrate": 4096,
        "recompute_extract": 6144,
        "recompute_verify": 3072,
        "whole_paper_verify": 16384,
        "whole_paper_challenge": 16384,
        "section_challenger": 6144,
        "meta": 8192,
        "synthesis": 8192,
        "validate": 8192,
    },
}


def call_max_tokens(call_type: str, level: str) -> int | None:
    return OUTPUT_BUDGETS.get(level, {}).get(call_type)


class StageEstimate(BaseModel):
    call_type: str
    role: str  # role label; blends render as "strong+small"
    calls: int
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float


class PlanAdjustment(BaseModel):
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reason: str  # what triggered the change
    action: str  # what was changed
    detail: str = ""
    source: str = "planner"  # planner | steering | fallback | milestone


class ReviewPlan(BaseModel):
    plan_version: int = 1
    budget_usd: float  # hard ceiling, never exceeded
    target_usd: float  # soft target the run commits to
    knobs: PlanKnobs
    control_estimates: dict[str, StageEstimate] = Field(default_factory=dict)
    stage_estimates: dict[str, StageEstimate] = Field(default_factory=dict)
    planner_calls_max: int = MAX_PLANNER_CALLS
    planner_calls_used: int = 0
    rationale: str = ""
    based_on: str = "doc_stats"
    adjustments: list[PlanAdjustment] = Field(default_factory=list)

    def all_estimates(self) -> dict[str, StageEstimate]:
        out: dict[str, StageEstimate] = dict(self.control_estimates)
        out.update(self.stage_estimates)
        return out

    def planned_cost(self) -> float:
        return sum(e.est_cost_usd for e in self.all_estimates().values())


class RepairOption(BaseModel):
    index: int
    action: str  # machine name, applied via apply_option()
    label: str  # human description of what changes
    saves_usd: float  # projected savings on REMAINING calls
    impact: str  # quality-impact note for the planner LLM


class Steering:
    """Result of the deterministic steering decision at a checkpoint."""

    def __init__(
        self,
        action: str,  # none | applied | needs_planner | cant_fix
        applied: RepairOption | None = None,
        options: list[RepairOption] | None = None,
        note: str = "",
    ):
        self.action = action
        self.applied = applied
        self.options = options or []
        self.note = note


# -------------------------------------------------------------------- harness --


class BudgetHarness:
    """One per run. Owns planning + tracking on top of the UsageLedger.

    The harness is the primary steering mechanism: it computes plans and
    repair options deterministically and only defers to the planner LLM when
    there is a genuine judgment call (several viable options). All calls —
    including the planner's own — are charged to the same budget.
    """

    def __init__(
        self,
        cfg: Config,
        ledger: UsageLedger,
        specs: dict[ModelRole, ModelSpec],
        budget_usd: float,
        auto: bool,
    ):
        self.cfg = cfg
        self.ledger = ledger
        self.specs = specs
        self.budget_usd = budget_usd
        self.auto = auto
        self.stats = DocumentStats()
        self.plan: ReviewPlan | None = None
        # plan-version trajectory: one snapshot per set_plan() call, with the
        # spend at the moment of the change (estimation-history dataset)
        self.plan_log: list[dict] = []
        # call_type -> actual call count / empirical token averages
        self.done: dict[str, int] = {}
        self._emp_in: dict[str, float] = {}
        self._emp_out: dict[str, float] = {}
        # call_type -> units skipped by the affordance gate (for reporting)
        self.gate_stopped: dict[str, int] = {}
        # observed review yield per call type (findings, max severity) for replans
        self.yield_data: dict[str, dict] = {}
        # booked calls that cost >= SPIKE_FACTOR x their estimate
        self.spike_events: list[dict] = []

    # ------------------------------------------------------------ pricing --

    def _price(self, role: ModelRole) -> tuple[float, float]:
        spec = self.specs.get(role)
        if spec is None:
            return (0.0, 0.0)  # role not configured (e.g. no vision model)
        p = self.ledger.price_for(spec)
        return p.input_per_mtok, p.output_per_mtok

    def _cost(self, role: ModelRole, in_tok: int, out_tok: int) -> float:
        pi, po = self._price(role)
        return (in_tok * pi + out_tok * po) / 1_000_000

    def require_pricing(self) -> None:
        """Hard-error on models with no pricing entry (auto/fit modes).

        An explicit 0.0 entry counts as priced (local models are free); a
        missing key means the user never configured prices, and every cost
        downstream would silently be zero.
        """
        missing: list[str] = []
        for role in (ModelRole.STRONG, ModelRole.SMALL, ModelRole.VISION, ModelRole.DECISION):
            if role not in self.specs:
                continue
            spec = self.specs[role]
            key = f"{spec.provider_name}/{spec.model}"
            if key not in self.ledger.pricing:
                missing.append(f"{role.value}={key}")
        if missing:
            snippet = "\n".join(
                f'  "{m.split("=", 1)[1]}": {{input_per_mtok: 3.0, output_per_mtok: 15.0}}'
                for m in missing
            )
            raise PricingError(
                f"Budget-driven planning needs prices for: {', '.join(missing)}. "
                "Add entries to llm.pricing in your config, e.g.:\n"
                + snippet
                + ("\n(Local models can be priced explicitly at 0.0.)")
            )

    # --------------------------------------------------------- unit costs --

    def _manuscript_tokens(self) -> int:
        return min(self.stats.chars, MAX_MANUSCRIPT_CHARS) // CHARS_PER_TOKEN

    def _base_unit(self, ct: str) -> tuple[str, int, int]:
        """(role label, est input tokens, est output tokens) for one call."""
        m = self._manuscript_tokens()
        sec = max(1, self.stats.n_sections)
        avg_sec = max(2000, self.stats.chars // sec // CHARS_PER_TOKEN)
        lens_in = min(40_000 // CHARS_PER_TOKEN, avg_sec) + 6000 // CHARS_PER_TOKEN + 700
        match ct:
            case "triage":
                return ("strong", m + 800, 1500)
            case "claim_inventory":
                return ("strong", m + 1200, 5000)
            case "claim_verify":
                in_tok = 12_000 // CHARS_PER_TOKEN + 6600 // CHARS_PER_TOKEN + 600
                return ("strong+small", in_tok, 1200)
            case "lens_section":
                return ("small", lens_in, 2000)
            case "artifact_verify":
                return ("vision", VISION_TOKENS_PER_ARTIFACT + 2000, 2000)
            case "whole_paper_verify":
                return ("strong", m + 900, 3000)
            case "section_challenger":
                in_tok = min(40_000 // CHARS_PER_TOKEN, avg_sec) + 20_000 // CHARS_PER_TOKEN + 2700
                return ("strong", in_tok, 2500)
            case "whole_paper_challenge":
                return ("strong", m + 30_000 // CHARS_PER_TOKEN + 900, 3000)
            case "defense":
                return ("small", 2200, 900)
            case "adjudication":
                return ("strong", 2800, 900)
            case "bibliography":
                return ("small", 22_000, 3000)
            case "meta":
                return ("strong", 60_000 // CHARS_PER_TOKEN + 14_000, 4000)
            case "validate":
                return ("strong", m + 80_000 // CHARS_PER_TOKEN + 800, 4000)
            case "surveyor":
                return ("small", 4000, 1500)
            case "fit_assessment":
                return ("strong", m + 2000, 3000)
            case "planner":
                return ("small", 6000, 1200)
            case "wrapup":
                return ("strong", m + 80_000 // CHARS_PER_TOKEN + 1500, 4000)
            case "recompute_extract":
                return ("small", 12_000, 2500)
            case "recompute_verify":
                return ("strong", 3500, 1200)
            case "claim_arbitrate":
                return ("strong", 8000, 1500)
            case "synthesis":
                return ("strong", 18_000, 6000)
            case "dm_call":
                return ("decision", 5000, 1)
        return ("small", 4000, 1500)

    def unit_cost(self, ct: str) -> float:
        """Estimated USD for one more call of this type; empirical once we
        have enough actuals (self-correcting estimates)."""
        role, est_in, est_out = self._base_unit(ct)
        if self.done.get(ct, 0) >= MIN_UNIT_SAMPLES and ct not in ("planner",):
            in_tok, out_tok = int(self._emp_in[ct]), int(self._emp_out[ct])
        else:
            in_tok, out_tok = est_in, est_out
        if "+" in role:  # claim_verify blends strong/small
            strong = self._cost(ModelRole.STRONG, in_tok, out_tok)
            small = self._cost(ModelRole.SMALL, in_tok, out_tok)
            return CLAIM_VERIFY_STRONG_SHARE * strong + (1 - CLAIM_VERIFY_STRONG_SHARE) * small
        return self._cost(ModelRole(role), in_tok, out_tok)

    # -------------------------------------------------------------- counts --

    def _est_comments(self, knobs: PlanKnobs) -> int:
        n_claims_checked = min(knobs.max_claims, self.stats.known_or_estimated_claims)
        return min(200, math.ceil(n_claims_checked * 0.5) + self.stats.n_sections * 2)

    def _call_counts(self, knobs: PlanKnobs) -> dict[str, int]:
        claims = min(knobs.max_claims, self.stats.known_or_estimated_claims)
        has_vision = ModelRole.VISION in self.specs
        counts = {
            "triage": 1,
            "claim_inventory": 2 if self.stats.chars > 70_000 else 1,
            "surveyor": 1,
            "planner": 1 + PLANNER_CONTINGENCY_CALLS if self.auto else 0,
            "claim_verify": claims * max(1, knobs.verify_passes),
            "claim_arbitrate": (
                math.ceil(claims * 0.15) if knobs.verify_passes > 1 else 0
            ),
            "lens_section": (
                self.stats.n_sections * max(1, self.stats.n_lenses)
                if knobs.per_section_lenses
                else 0
            ),
            "artifact_verify": (
                min(knobs.max_artifacts, self.stats.n_figures + self.stats.n_tables)
                if has_vision
                else 0
            ),
            "whole_paper_verify": 1 if knobs.whole_paper_passes else 0,
            "section_challenger": self.stats.n_sections,
            "whole_paper_challenge": 1,
            "defense": (
                (self._est_comments(knobs) * (2 if knobs.defense_second_round else 1))
                if knobs.defense_round
                else 0
            ),
            "adjudication": self._est_comments(knobs) if knobs.defense_round else 0,
            "bibliography": 1 if self.stats.n_references else 0,
            "recompute_extract": 1 if knobs.recompute else 0,
            "recompute_verify": 3 if knobs.recompute else 0,
            "meta": 1,
            "synthesis": 1 if knobs.synthesis else 0,
            "validate": 1,
        }
        return {ct: n for ct, n in counts.items() if n > 0}

    def _estimates_for(
        self, knobs: PlanKnobs
    ) -> tuple[dict[str, StageEstimate], dict[str, StageEstimate]]:
        control: dict[str, StageEstimate] = {}
        stages: dict[str, StageEstimate] = {}
        for ct, n in self._call_counts(knobs).items():
            role, est_in, est_out = self._base_unit(ct)
            cost = round(n * self.unit_cost(ct), 6)
            est = StageEstimate(
                call_type=ct,
                role=role,
                calls=n,
                est_input_tokens=est_in * n,
                est_output_tokens=est_out * n,
                est_cost_usd=cost,
            )
            (control if ct in CONTROL_TYPES else stages)[ct] = est
        return control, stages

    # ------------------------------------------------------ plan builders --

    def min_viable_knobs(self) -> PlanKnobs:
        return PlanKnobs(
            max_claims=MIN_CLAIMS,
            per_section_lenses=False,
            whole_paper_passes=False,
            defense_round=False,
            max_artifacts=0,
            recompute=False,
            synthesis=False,  # starved runs keep the meta merger's own overall
        )

    def _cost_of(self, knobs: PlanKnobs) -> float:
        control, stages = self._estimates_for(knobs)
        return sum(e.est_cost_usd for e in {**control, **stages}.values())

    def min_viable_cost(self) -> float:
        return self._cost_of(self.min_viable_knobs())

    def standard_cost(self) -> float:
        return self._cost_of(PlanKnobs())

    def deterministic_plan(self, reason: str = "initial") -> ReviewPlan:
        """Greedy quality ordering within the budget; carves out the control
        reserve first. Also the fallback whenever the planner LLM fails."""
        control_only = self._cost_of(self.min_viable_knobs())
        if self.budget_usd < control_only:
            raise BudgetInfeasible(
                f"A meaningful review of this manuscript costs at least "
                f"${control_only:.2f} (control calls + {MIN_CLAIMS} claim verifications + "
                f"core synthesis), but the budget is ${self.budget_usd:.2f}.",
                min_budget_usd=round(control_only, 2),
                standard_budget_usd=round(self.standard_cost(), 2),
            )
        knobs = self.min_viable_knobs()
        est_claims = self.stats.known_or_estimated_claims
        # 1. claims: the primary engine — binary-search the affordable count
        lo, hi = MIN_CLAIMS, min(est_claims, MAX_CLAIMS_CAP)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._cost_of(knobs.model_copy(update={"max_claims": mid})) <= self.budget_usd:
                lo = mid
            else:
                hi = mid - 1
        knobs.max_claims = lo
        # 2-5. cheap high-value passes, then breadth
        for update in (
            {"whole_paper_passes": True},
            {"defense_round": True},
            {"recompute": True},
            {"per_section_lenses": True},
            {"max_artifacts": min(12, self.stats.n_figures + self.stats.n_tables)},
        ):
            candidate = knobs.model_copy(update=update)
            if self._cost_of(candidate) <= self.budget_usd:
                knobs = candidate
        control, stages = self._estimates_for(knobs)
        return ReviewPlan(
            budget_usd=self.budget_usd,
            target_usd=round(self._cost_of(knobs), 4),
            knobs=knobs,
            control_estimates=control,
            stage_estimates=stages,
            rationale="Deterministic fallback plan: greedy quality ordering within budget.",
            based_on="claims" if self.stats.n_claims is not None else "doc_stats",
            adjustments=[],
        )

    def preset_plan(self, knobs: PlanKnobs, depth_name: str) -> ReviewPlan:
        """Tracking-only plan for preset modes (no planner LLM involved)."""
        control, stages = self._estimates_for(knobs)
        return ReviewPlan(
            budget_usd=self.budget_usd,
            target_usd=round(self._cost_of(knobs), 4),
            knobs=knobs,
            control_estimates=control,
            stage_estimates=stages,
            planner_calls_max=0,
            rationale=f"Preset mode ({depth_name}); budget acts as cap only.",
            based_on="preset",
        )

    # ------------------------------------------------------- plan handling --

    def set_plan(self, plan: ReviewPlan, reason: str = "") -> None:
        self.plan = plan
        self.plan_log.append(
            {
                "ts": datetime.now(UTC).isoformat(),
                "plan_version": plan.plan_version,
                "based_on": plan.based_on,
                "reason": reason,
                "target_usd": plan.target_usd,
                "planned_usd": round(plan.planned_cost(), 6),
                "knobs": plan.knobs.model_dump(),
                "spend_usd_at_change": round(self.ledger.total_cost(), 6),
            }
        )

    def validate_plan(self, plan: ReviewPlan, source: str = "planner") -> ReviewPlan:
        """Never trust LLM arithmetic: clamp knobs, recompute cost with the
        cost model, and greedily repair anything that does not fit."""
        knobs = plan.knobs.model_copy(
            update={
                "max_claims": max(1, min(int(plan.knobs.max_claims), MAX_CLAIMS_CAP)),
                "max_artifacts": max(0, min(int(plan.knobs.max_artifacts), MAX_ARTIFACTS_CAP)),
            }
        )
        # greedy repair in reverse quality order until it fits
        if self._cost_of(knobs) > self.budget_usd:
            for update in (
                {"max_artifacts": 0},
                {"per_section_lenses": False},
                {"whole_paper_passes": False},
                {"defense_round": False},
            ):
                if self._cost_of(knobs) <= self.budget_usd:
                    break
                knobs = knobs.model_copy(update=update)
            lo, hi = MIN_CLAIMS, knobs.max_claims
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self._cost_of(knobs.model_copy(update={"max_claims": mid})) <= self.budget_usd:
                    lo = mid
                else:
                    hi = mid - 1
            knobs.max_claims = lo
        control, stages = self._estimates_for(knobs)
        return plan.model_copy(
            update={
                "knobs": knobs,
                "control_estimates": control,
                "stage_estimates": stages,
                "target_usd": round(self._cost_of(knobs), 4),
                "budget_usd": self.budget_usd,
                "plan_version": (plan.plan_version or 0) + 1,
                "based_on": "claims" if self.stats.n_claims is not None else "doc_stats",
                "adjustments": plan.adjustments
                + [
                    PlanAdjustment(
                        reason="validated",
                        action="clamp+recompute",
                        source=source,
                    )
                ],
            }
        )

    # ---------------------------------------------------------- tracking --

    def note_record(self, rec: CallRecord) -> None:
        ct = rec.call_type or "other"
        n = self.done.get(ct, 0)
        a = 0.3  # EMA weight for the new sample
        self._emp_in[ct] = (
            rec.input_tokens if n == 0 else (1 - a) * self._emp_in[ct] + a * rec.input_tokens
        )
        self._emp_out[ct] = (
            rec.output_tokens if n == 0 else (1 - a) * self._emp_out[ct] + a * rec.output_tokens
        )
        self.done[ct] = n + 1
        self._detect_spike(rec)

    def actual_cost(self) -> float:
        return self.ledger.total_cost()

    def projected_final(self, knobs: PlanKnobs | None = None) -> float:
        """Actual spend + estimated cost of the REMAINING planned calls."""
        if knobs is not None:
            control, stages = self._estimates_for(knobs)
            estimates: dict[str, StageEstimate] = {**control, **stages}
        elif self.plan is not None:
            estimates = self.plan.all_estimates()
        else:
            return self.actual_cost()
        proj = self.actual_cost()
        for ct, est in estimates.items():
            remaining = max(0, est.calls - self.done.get(ct, 0))
            if remaining:
                proj += remaining * self.unit_cost(ct)
        return proj

    def status(self) -> dict:
        plan = self.plan
        by_ct: dict[str, dict] = {}
        if plan is not None:
            for ct, est in plan.all_estimates().items():
                by_ct[ct] = {
                    "role": est.role,
                    "planned_calls": est.calls,
                    "done_calls": self.done.get(ct, 0),
                    "planned_usd": est.est_cost_usd,
                }
        actual = self.actual_cost()
        control_actual = 0.0
        for rec in self.ledger.calls:
            if (rec.call_type or "other") in CONTROL_TYPES:
                control_actual += rec.estimated_cost_usd
        return {
            "budget_usd": round(self.budget_usd, 4),
            "target_usd": round(plan.target_usd, 4) if plan else None,
            "planned_usd": round(plan.planned_cost(), 4) if plan else None,
            "actual_usd": round(actual, 4),
            "control_actual_usd": round(control_actual, 4),
            "projected_final_usd": round(self.projected_final(), 4),
            "planner_calls_used": plan.planner_calls_used if plan else 0,
            "planner_calls_max": plan.planner_calls_max if plan else 0,
            "by_call_type": by_ct,
        }

    # ---------------------------------------------------------- steering --

    def should_replan(self) -> str | None:
        """Reason string when a replan is warranted, else None.

        Gated: only in auto mode, and only on material deviation — the
        replan COUNT is deliberately unbounded (project-owner decision);
        anti-thrash comes from the deviation thresholds and the fact that
        every committee call draws from the same budget."""
        if not self.auto or self.plan is None:
            return None
        proj = self.projected_final()
        if proj > OVERRUN_BUDGET_FRACTION * self.budget_usd:
            return "projected overrun of the budget"
        if proj > (1 + TARGET_TOLERANCE) * self.plan.target_usd:
            return "projected spend exceeds the soft target by more than 20%"
        if self.spike_events:
            return "an unexpectedly expensive call was booked (spike)"
        return None

    def repair_options(self) -> list[RepairOption]:
        """Pre-costed adjustments to FUTURE calls (already-spent calls stay).
        Ordered by descending savings."""
        if self.plan is None:
            return []
        cur = self.plan.knobs
        base_proj = self.projected_final()
        candidates: list[tuple[str, str, str, PlanKnobs]] = []
        est_claims = self.stats.known_or_estimated_claims
        candidates.append(
            (
                "drop_artifacts",
                "Stop figure/table artifact checks",
                "Cheap to re-run later; findings are usually peripheral",
                cur.model_copy(update={"max_artifacts": 0}),
            )
        )
        candidates.append(
            (
                "drop_lenses",
                "Skip the per-section lens grid",
                "Loses breadth (per-section prose/math/stats reading)",
                cur.model_copy(update={"per_section_lenses": False}),
            )
        )
        candidates.append(
            (
                "drop_whole_paper_verify",
                "Skip the whole-paper coherence pass",
                "Loses cross-section consistency findings",
                cur.model_copy(update={"whole_paper_passes": False}),
            )
        )
        candidates.append(
            (
                "drop_defense",
                "Skip the defense/adjudication gate",
                "Keeps more unverified comments; report notes this",
                cur.model_copy(update={"defense_round": False}),
            )
        )
        for frac, name in ((0.7, "trim_claims_30"), (0.5, "trim_claims_50")):
            new_n = max(MIN_CLAIMS, min(cur.max_claims, int(est_claims * frac)))
            if new_n < cur.max_claims:
                candidates.append(
                    (
                        name,
                        f"Trim claim verification to ~{new_n} claims",
                        "Claims are importance-sorted; peripheral ones go first",
                        cur.model_copy(update={"max_claims": new_n}),
                    )
                )
        out: list[RepairOption] = []
        for i, (action, label, impact, knobs) in enumerate(candidates):
            saves = base_proj - self.projected_final(knobs=knobs)
            if saves > 1e-9:
                out.append(
                    RepairOption(
                        index=i,
                        action=action,
                        label=label,
                        saves_usd=round(saves, 4),
                        impact=impact,
                    )
                )
        out.sort(key=lambda o: o.saves_usd, reverse=True)
        for i, o in enumerate(out):
            o.index = i
        return out

    def apply_option(
        self, option: RepairOption, reason: str, source: str = "steering"
    ) -> ReviewPlan:
        assert self.plan is not None
        knobs = self.plan.knobs
        updates = {
            "drop_artifacts": {"max_artifacts": 0},
            "drop_lenses": {"per_section_lenses": False},
            "drop_whole_paper_verify": {"whole_paper_passes": False},
            "drop_defense": {"defense_round": False},
        }
        if option.action in updates:
            knobs = knobs.model_copy(update=updates[option.action])
        elif option.action.startswith("trim_claims"):
            frac = 0.7 if "30" in option.action else 0.5
            knobs = knobs.model_copy(
                update={
                    "max_claims": max(
                        MIN_CLAIMS,
                        min(knobs.max_claims, int(self.stats.known_or_estimated_claims * frac)),
                    )
                }
            )
        return self.validate_plan(
            self.plan.model_copy(
                update={
                    "knobs": knobs,
                    "adjustments": self.plan.adjustments
                    + [
                        PlanAdjustment(
                            reason=reason, action=option.action, detail=option.label, source=source
                        )
                    ],
                }
            ),
            source=source,
        )

    def steer(self) -> Steering:
        """Deterministic first: act without the LLM whenever the choice is
        forced, involve the planner only for real judgment calls."""
        reason = self.should_replan()
        if reason is None:
            return Steering("none", note="on plan")
        proj = self.projected_final()
        if proj <= self.budget_usd:
            # soft-target drift only: tolerated by design
            return Steering("none", note=f"within soft tolerance ({reason})")
        options = [o for o in self.repair_options() if proj - o.saves_usd <= self.budget_usd]
        if not options:
            return Steering(
                "cant_fix",
                note="no repair option closes the gap; relying on per-call gating and the cap",
            )
        if len(options) == 1:
            return Steering("applied", applied=options[0], options=options, note=reason)
        return Steering("needs_planner", options=options, note=reason)

    # ---------------------------------------------------------- gating --

    def allow_call(self, ct: str) -> bool:
        """Fan-out gate: can we afford one more unit of this call type while
        keeping the projection under the hard budget? Preset mode is uncapped
        by design, so its gate is always open."""
        if not self.auto or self.plan is None:
            return True
        return self.projected_final() + self.unit_cost(ct) <= self.budget_usd

    def note_gate_stop(self, ct: str, units_skipped: int) -> None:
        self.gate_stopped[ct] = self.gate_stopped.get(ct, 0) + units_skipped

    # ---------------------------------------------------------- briefing --

    def briefing(self, options: list[RepairOption] | None = None) -> dict:
        """Structured payload for the planner LLM: everything it needs to
        design or adjust the plan, including what its own call costs."""
        st = self.status()
        return {
            "mode": "adjust" if options is not None else "initial",
            "budget_usd": st["budget_usd"],
            "soft_target_usd": st["target_usd"],
            "spent_usd": st["actual_usd"],
            "projected_final_usd": st["projected_final_usd"],
            "planner_call_cost_usd": round(self.unit_cost("planner"), 4),
            "planner_calls_remaining": (
                (self.plan.planner_calls_max - self.plan.planner_calls_used)
                if self.plan
                else MAX_PLANNER_CALLS
            ),
            "document_stats": self.stats.model_dump(),
            "unit_costs_usd": {ct: round(self.unit_cost(ct), 4) for ct in REVIEW_TYPES},
            "current_knobs": self.plan.knobs.model_dump() if self.plan else None,
            "call_progress": st["by_call_type"],
            "repair_options": [o.model_dump() for o in (options or [])],
            "rules": (
                "Plan honestly under budget; a simple paper should target well below it. "
                "Your plan is validated and repaired deterministically if it does not fit. "
                "Returning action='noop' when things are on track is free of regret."
            ),
        }

    # ---------------------------------------------- frontier & replanning --

    def wrapup_reserve(self) -> float:
        """Cost kept aside so an incomplete review can always be wrapped up
        by a model call (strong, manuscript + findings as context)."""
        if not self.auto:
            return 0.0
        return round(self.unit_cost("wrapup"), 6)

    def effective_budget(self) -> float:
        return self.budget_usd - self.wrapup_reserve()

    def preset_estimates(self) -> dict[str, float]:
        out = {}
        for name, knobs in PRESET_KNOBS.items():
            control, stages = self._estimates_for(knobs)
            out[name] = round(sum(e.est_cost_usd for e in {**control, **stages}.values()), 6)
        return out

    def fast_floor(self) -> float:
        return self.preset_estimates()["fast"]

    def frontier(self, limit: int = 10) -> list[FrontierPath]:
        """Deterministic feasible-by-construction menu for the committee.
        Every returned path fits the (reserve-adjusted) budget."""
        cap = self.effective_budget()
        total_claims = self.stats.known_or_estimated_claims
        out: list[FrontierPath] = []
        for name, desc, knobs in FRONTIER_ARCHETYPES:
            k = knobs.model_copy(update={"max_claims": min(knobs.max_claims, total_claims)})
            if (
                k.max_claims < knobs.max_claims
                and knobs.max_claims > 0
                and name
                not in (
                    "fast-scan",
                    "triage-plus-lenses",
                )
            ):
                # rename to reflect the actual claim count when capped
                desc = desc.replace("Top-50", f"Top-{k.max_claims}").replace(
                    "All claims", f"Top-{k.max_claims} claims"
                )
            cost = self._cost_of(k)
            if cost <= cap:
                out.append(
                    FrontierPath(
                        name=name,
                        description=desc,
                        knobs=k,
                        est_cost_usd=round(cost, 6),
                        claims_count=k.max_claims,
                        passes=_passes_of(k),
                    )
                )
        # keep diversity: sort by cost, cap the list
        out.sort(key=lambda p: p.est_cost_usd)
        if len(out) > limit:
            # evenly spaced selection that always keeps the extremes
            idx = [round(i * (len(out) - 1) / (limit - 1)) for i in range(limit)]
            out = [out[i] for i in dict.fromkeys(idx)]
        return out

    def path_to_plan(self, path: FrontierPath, reason: str, rationale: str = "") -> ReviewPlan:
        """Commit a frontier path: feasible by construction, recomputed costs."""
        control, stages = self._estimates_for(path.knobs)
        plan = ReviewPlan(
            budget_usd=self.budget_usd,
            target_usd=round(path.est_cost_usd, 4),
            knobs=path.knobs,
            control_estimates=control,
            stage_estimates=stages,
            # no numeric cap on replans (project-owner decision); the counter
            # still tracks how many committee calls were spent
            planner_calls_max=10_000,
            planner_calls_used=(self.plan.planner_calls_used if self.plan else 0),
            rationale=rationale or f"frontier path '{path.name}': {path.description}",
            based_on="frontier",
        )
        plan.adjustments.append(
            PlanAdjustment(
                reason=reason,
                action=f"frontier:{path.name}",
                detail=path.description,
                source="committee",
            )
        )
        self.set_plan(plan, reason=f"frontier:{path.name}")
        return plan

    def apply_residual(self, path: ResidualPath, reason: str) -> ReviewPlan:
        """Commit a replan continuation: knobs applied, costs recomputed by
        the cost model (feasible by construction — built from the residual)."""
        control, stages = self._estimates_for(path.knobs)
        plan = ReviewPlan(
            budget_usd=self.budget_usd,
            target_usd=round(self.actual_cost() + self._remaining_cost(path.knobs), 4),
            knobs=path.knobs,
            control_estimates=control,
            stage_estimates=stages,
            planner_calls_max=10_000,
            planner_calls_used=(self.plan.planner_calls_used if self.plan else 0),
            rationale=f"continuation '{path.name}': {path.description}",
            based_on="residual-frontier",
        )
        plan.adjustments.append(
            PlanAdjustment(
                reason=reason,
                action=f"continuation:{path.name}",
                detail=path.description,
                source="committee",
            )
        )
        self.set_plan(plan, reason=f"continuation:{path.name}")
        return plan

    def _remaining_cost(self, knobs: PlanKnobs) -> float:
        """Cost of the calls still to be made under these knobs, given what
        is already done (per call type)."""
        control, stages = self._estimates_for(knobs)
        est = {**control, **stages}
        remaining = 0.0
        for ct, e in est.items():
            n = max(0, e.calls - self.done.get(ct, 0))
            if n:
                remaining += n * self.unit_cost(ct)
        return remaining

    def residual_frontier(self, limit: int = 5) -> list[ResidualPath]:
        """4-5 continuation plans built from residual steps + residual budget.
        Feasible by construction: actual + remaining <= budget - wrapup reserve."""
        if self.plan is None:
            return []
        cap = self.effective_budget()
        actual = self.actual_cost()
        cur = self.plan.knobs
        verified = self.done.get("claim_verify", 0)
        remaining_now = max(0, min(cur.max_claims, self.stats.known_or_estimated_claims) - verified)

        def variant(name, desc, knobs) -> ResidualPath | None:
            proj = actual + self._remaining_cost(knobs)
            if proj > cap:
                return None
            extra_claims = max(
                0, min(knobs.max_claims, self.stats.known_or_estimated_claims) - verified
            )
            passes = [p for p in _passes_of(knobs) if p not in _passes_of(cur)] or [
                "finish current passes"
            ]
            return ResidualPath(
                name=name,
                description=desc,
                knobs=knobs,
                projected_final_usd=round(proj, 6),
                claims_remaining=extra_claims,
                passes_remaining=passes,
            )

        candidates = [
            ("continue", "Continue the current plan as-is", cur),
            (
                "trim-claims",
                f"Verify ~half of the {remaining_now} remaining claims "
                "(least important dropped first)",
                cur.model_copy(update={"max_claims": verified + max(0, remaining_now // 2)}),
            ),
            (
                "stop-claims-hold-breadth",
                "Stop new claim verification; keep the planned breadth passes",
                cur.model_copy(update={"max_claims": verified}),
            ),
            (
                "drop-defense",
                "Keep claims; skip the defense/adjudication gate",
                cur.model_copy(update={"defense_round": False}),
            ),
            (
                "lean-finish",
                "Stop claims and breadth; challenger + meta + validation only",
                cur.model_copy(
                    update={
                        "max_claims": verified,
                        "per_section_lenses": False,
                        "whole_paper_passes": False,
                        "defense_round": False,
                        "max_artifacts": 0,
                    }
                ),
            ),
        ]
        # quality-ordered: keep the best feasible ones
        out = []
        for name, desc, knobs in candidates:
            v = variant(name, desc, knobs)
            if v is not None:
                out.append(v)
        # 'continue' first if feasible, else ordered by least cutting
        priority = {
            "continue": 0,
            "trim-claims": 1,
            "stop-claims-hold-breadth": 2,
            "drop-defense": 3,
            "lean-finish": 4,
        }
        out.sort(key=lambda p: priority.get(p.name, 9))
        return out[:limit]

    def cheapest_completion_cost(self) -> float:
        """Cost of the cheapest possible completion (lean-finish continuation)."""
        if self.plan is None:
            return 0.0
        verified = self.done.get("claim_verify", 0)
        lean = self.plan.knobs.model_copy(
            update={
                "max_claims": verified,
                "per_section_lenses": False,
                "whole_paper_passes": False,
                "defense_round": False,
                "max_artifacts": 0,
            }
        )
        return self.actual_cost() + self._remaining_cost(lean)

    def shortfall(self) -> float:
        """Additional budget needed to complete even the cheapest continuation."""
        if self.plan is None:
            return 0.0
        return round(max(0.0, self.cheapest_completion_cost() - self.effective_budget()), 6)

    def note_yield(self, call_type: str, findings: int, max_severity: float = 0.0) -> None:
        agg = self.yield_data.setdefault(
            call_type, {"findings": 0, "max_severity": 0.0, "units": 0}
        )
        agg["findings"] += findings
        agg["max_severity"] = max(agg["max_severity"], max_severity)
        agg["units"] += 1

    def _detect_spike(self, rec: CallRecord) -> None:
        if rec.call_type in CONTROL_TYPES or rec.call_type.startswith("dm_"):
            return
        est = self.unit_cost(rec.call_type)
        if est > 0 and rec.estimated_cost_usd >= SPIKE_FACTOR * est:
            self.spike_events.append(
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "call_type": rec.call_type,
                    "estimated_unit_usd": round(est, 6),
                    "actual_usd": round(rec.estimated_cost_usd, 6),
                }
            )

    def consume_spike(self) -> dict | None:
        """Return and clear the pending spike (checked at every checkpoint)."""
        if self.spike_events:
            return self.spike_events[-1]
        return None

    def replan_context(self, paths: list[ResidualPath] | None = None) -> dict:
        st = self.status()
        return {
            "budget_usd": st["budget_usd"],
            "target_usd": st["target_usd"],
            "spent_usd": st["actual_usd"],
            "projected_final_usd": st["projected_final_usd"],
            "residual_budget_usd": round(self.effective_budget() - st["actual_usd"], 6),
            "coverage": {
                "claims_planned": (
                    min(self.plan.knobs.max_claims, self.stats.known_or_estimated_claims)
                    if self.plan
                    else 0
                ),
                "claims_verified": self.done.get("claim_verify", 0),
                "claims_remaining": max(
                    0,
                    (
                        min(self.plan.knobs.max_claims, self.stats.known_or_estimated_claims)
                        if self.plan
                        else 0
                    )
                    - self.done.get("claim_verify", 0),
                ),
            },
            "observed_yield": dict(self.yield_data),
            "gate_stops": dict(self.gate_stopped),
            "spikes": list(self.spike_events[-3:]),
            "unit_costs_usd": {ct: round(self.unit_cost(ct), 6) for ct in REVIEW_TYPES},
            "residual_paths": [p.model_dump() for p in (paths if paths is not None else [])],
            "rules": (
                "Choose the continuation that maximizes review value within the "
                "residual budget. Depth = more claims verified; breadth = more "
                "passes. Use observed yield: keep spending where findings emerge, "
                "cut where they do not."
            ),
        }


# ------------------------------------------------------------------ frontier --

DEPTH_LEVELS = (0, 10, 25, 50, 150)  # claim counts by importance rank (capped)

PRESET_KNOBS: dict[str, PlanKnobs] = {
    # cheaper and less detailed than hosted deep-review products
    "fast": PlanKnobs(
        max_claims=20,
        per_section_lenses=False,
        whole_paper_passes=True,
        defense_round=False,
        max_artifacts=6,
        recompute=False,
        verify_passes=1,
        claim_routing="base",
        output_budget="low",
    ),
    # "match Refine": reproduction voting on every claim, strong-model routing
    # on numerical/table claims, full lens grid, defense, all checkers + DOI
    # loop, lead-referee synthesis
    "standard": PlanKnobs(
        verify_passes=2,
        claim_routing="strong",
        output_budget="default",
    ),
    # "deeper than Refine": triple-pass verification routed to the auditor
    # when configured, all claims on the strong model, double defense, max
    # reasoning budgets
    "deep": PlanKnobs(
        max_claims=150,
        max_artifacts=50,
        verify_passes=3,
        claim_routing="all_strong",
        output_budget="high",
        defense_second_round=True,
    ),
}

# (name, description, knobs) — ~14 archetypes over the two axes; the frontier
# keeps the ~10 most diverse FEASIBLE ones for the committee.
FRONTIER_ARCHETYPES: list[tuple[str, str, PlanKnobs]] = [
    (
        "fast-scan",
        "No claim verification; whole-paper coherence pass only",
        PlanKnobs(
            max_claims=0,
            per_section_lenses=False,
            whole_paper_passes=True,
            defense_round=False,
            max_artifacts=0,
        ),
    ),
    (
        "lean-core",
        "Minimal meaningful: 6 central claims, no extras",
        PlanKnobs(
            max_claims=MIN_CLAIMS,
            per_section_lenses=False,
            whole_paper_passes=False,
            defense_round=False,
            max_artifacts=0,
        ),
    ),
    (
        "claims-light",
        "Top-10 claims + whole-paper pass",
        PlanKnobs(
            max_claims=10,
            per_section_lenses=False,
            whole_paper_passes=True,
            defense_round=False,
            max_artifacts=0,
        ),
    ),
    (
        "claims-standard",
        "Top-25 claims + whole-paper pass",
        PlanKnobs(
            max_claims=25,
            per_section_lenses=False,
            whole_paper_passes=True,
            defense_round=False,
            max_artifacts=0,
        ),
    ),
    (
        "claims-deep",
        "Top-50 claims + whole-paper pass + defense gate",
        PlanKnobs(
            max_claims=50,
            per_section_lenses=False,
            whole_paper_passes=True,
            defense_round=True,
            max_artifacts=0,
        ),
    ),
    (
        "claims-max",
        "All claims + whole-paper pass + defense gate (depth-maximal)",
        PlanKnobs(
            max_claims=150,
            per_section_lenses=False,
            whole_paper_passes=True,
            defense_round=True,
            max_artifacts=0,
        ),
    ),
    (
        "balanced",
        "Top-25 claims + lenses + whole-paper + defense + some artifacts",
        PlanKnobs(
            max_claims=25,
            per_section_lenses=True,
            whole_paper_passes=True,
            defense_round=True,
            max_artifacts=6,
        ),
    ),
    (
        "breadth-standard",
        "Top-15 claims + full breadth (lenses, whole-paper, artifacts, defense)",
        PlanKnobs(
            max_claims=15,
            per_section_lenses=True,
            whole_paper_passes=True,
            defense_round=True,
            max_artifacts=12,
        ),
    ),
    (
        "breadth-max",
        "Top-10 claims + maximal breadth passes",
        PlanKnobs(
            max_claims=10,
            per_section_lenses=True,
            whole_paper_passes=True,
            defense_round=True,
            max_artifacts=25,
        ),
    ),
    (
        "deep-plus",
        "Top-50 claims + full breadth + defense",
        PlanKnobs(
            max_claims=50,
            per_section_lenses=True,
            whole_paper_passes=True,
            defense_round=True,
            max_artifacts=12,
        ),
    ),
    (
        "cross-check",
        "Top-20 claims, each verified twice (auditor if configured), strong "
        "numerical routing + defense",
        PlanKnobs(
            max_claims=20,
            per_section_lenses=False,
            whole_paper_passes=True,
            defense_round=True,
            verify_passes=2,
            claim_routing="strong",
        ),
    ),
    (
        "deep-verify",
        "Top-30 claims, triple-pass verification (auditor cross-examination), "
        "all-strong routing, defense + recomputation",
        PlanKnobs(
            max_claims=30,
            per_section_lenses=False,
            whole_paper_passes=True,
            defense_round=True,
            verify_passes=3,
            claim_routing="all_strong",
        ),
    ),
    (
        "rigor-max",
        "Top-40 claims × 2 passes + full breadth + defense + recomputation",
        PlanKnobs(
            max_claims=40,
            per_section_lenses=True,
            whole_paper_passes=True,
            defense_round=True,
            max_artifacts=6,
            verify_passes=2,
            claim_routing="strong",
        ),
    ),
    (
        "maximal",
        "Everything that fits: all claims, all passes",
        PlanKnobs(
            max_claims=150,
            per_section_lenses=True,
            whole_paper_passes=True,
            defense_round=True,
            max_artifacts=25,
        ),
    ),
    (
        "triage-plus-lenses",
        "No claims; lenses + whole-paper (pure breadth)",
        PlanKnobs(
            max_claims=0,
            per_section_lenses=True,
            whole_paper_passes=True,
            defense_round=False,
            max_artifacts=6,
        ),
    ),
]


class FrontierPath(BaseModel):
    """One feasible-by-construction plan option on the depth×breadth axes."""

    name: str
    description: str
    knobs: PlanKnobs
    est_cost_usd: float
    claims_count: int  # ranked claims verified under this path
    passes: list[str] = Field(default_factory=list)  # breadth passes included

    def menu_line(self) -> str:
        return (
            f"{self.name}: {self.description} — approx ${self.est_cost_usd:.4f} "
            f"({self.claims_count} claims; passes: {', '.join(self.passes) or 'none'})"
        )


def _passes_of(knobs: PlanKnobs) -> list[str]:
    passes = []
    if knobs.per_section_lenses:
        passes.append("section-lenses")
    if knobs.whole_paper_passes:
        passes.append("whole-paper")
    if knobs.max_artifacts > 0:
        passes.append(f"artifacts≤{knobs.max_artifacts}")
    if knobs.defense_round:
        passes.append("defense-gate")
    return passes


class ResidualPath(BaseModel):
    """A continuation option at replan time: feasible w.r.t. residual budget."""

    name: str
    description: str
    knobs: PlanKnobs  # the plan knobs AFTER applying this continuation
    projected_final_usd: float  # actual + remaining under these knobs
    claims_remaining: int  # additional claims verified if followed
    passes_remaining: list[str] = Field(default_factory=list)


SPIKE_FACTOR = 8.0  # a call costing >= 8x its estimate is a spike
