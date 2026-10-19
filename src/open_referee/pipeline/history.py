"""Run-history store: per-phase estimation dataset for future calibration.

Every run (review and fit, whatever the outcome) writes one structured
record pairing the paper's characteristics with what was planned and what
was actually spent — per pipeline stage, per call type, and per individual
call. Tokens are the calibration target: they are pricing-independent, so
the history stays valid when prices change.

Storage: a single JSONL file (``~/.open-referee/run_history.jsonl``), one
line per run_id, updated in place when a run resumes or re-exits, so the
file never double-counts a resumed run's spend.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from open_referee.pipeline.budget import BudgetHarness
    from open_referee.pipeline.state import StageEvent

logger = logging.getLogger(__name__)

HISTORY_FILENAME = "run_history.jsonl"
_write_lock = threading.Lock()

# static attribution of call types to pipeline stages (the "phases")
STAGE_OF_CALL_TYPE: dict[str, str] = {
    "triage": "triage",
    "claim_inventory": "claims",
    "surveyor": "survey",
    "planner": "plan",
    "claim_verify": "verify",
    "lens_section": "verify",
    "artifact_verify": "verify",
    "whole_paper_verify": "verify",
    "recompute_extract": "verify",
    "recompute_verify": "verify",
    "section_challenger": "challenge",
    "whole_paper_challenge": "challenge",
    "defense": "challenge",
    "adjudication": "challenge",
    "bibliography": "bibliography",
    "meta": "meta",
    "validate": "validate",
    "fit_assessment": "fit",
}

# stages whose calls belong to the control plane (planning overhead)
CONTROL_STAGES = frozenset({"triage", "claims", "survey", "plan"})

# every stage always present in a record (empty ones included) so downstream
# analysis can rely on a stable schema
CANONICAL_STAGES = (
    "ingest",
    "triage",
    "claims",
    "survey",
    "scout",
    "verify",
    "challenge",
    "bibliography",
    "meta",
    "validate",
    "assemble",
    "plan",
    "fit",
)


def history_path() -> Path:
    from open_referee.config import DEFAULT_CONFIG_DIR

    return DEFAULT_CONFIG_DIR / HISTORY_FILENAME


def _agg() -> dict:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


def _add(agg: dict, in_tok: int, out_tok: int, cost: float) -> None:
    agg["calls"] += 1
    agg["input_tokens"] += in_tok
    agg["output_tokens"] += out_tok
    agg["cost_usd"] += cost


def _round_agg(agg: dict) -> dict:
    agg["cost_usd"] = round(agg["cost_usd"], 6)
    return agg


def _stage_timing(events: Iterable[StageEvent]) -> dict[str, dict]:
    """Per-stage status + wall-clock boundaries from the event log.

    Tolerant of resumed stages (a DONE with no prior RUNNING in this
    process) and of repeated RUNNING emissions mid-stage (progress notes):
    started_at keeps the first RUNNING, ended_at the last terminal event.
    """
    out: dict[str, dict] = {}
    for ev in events:
        name = getattr(ev.stage, "value", str(ev.stage))
        status = getattr(ev.status, "value", str(ev.status))
        entry = out.setdefault(name, {"status": "pending", "started_at": None, "ended_at": None})
        if status == "running" and entry["started_at"] is None:
            entry["started_at"] = ev.ts.isoformat()
        elif status in ("done", "failed", "skipped"):
            entry["ended_at"] = ev.ts.isoformat()
            entry["status"] = status
    for entry in out.values():
        if entry["started_at"] and entry["ended_at"]:
            start = datetime.fromisoformat(entry["started_at"])
            end = datetime.fromisoformat(entry["ended_at"])
            entry["duration_s"] = round((end - start).total_seconds(), 3)
        else:
            entry["duration_s"] = None
    return out


def _stage_start_markers(events: Iterable[StageEvent]) -> list[tuple[datetime, str]]:
    """Ordered (ts, stage) markers used to attribute untyped calls to the
    stage that was running when the call was made."""
    markers: list[tuple[datetime, str]] = []
    for ev in events:
        status = getattr(ev.status, "value", str(ev.status))
        if status == "running":
            markers.append((ev.ts, getattr(ev.stage, "value", str(ev.stage))))
    markers.sort(key=lambda m: m[0])
    return markers


def _attribute_stage(ts: datetime, markers: list[tuple[datetime, str]]) -> str:
    name = "other"
    for m_ts, m_stage in markers:
        if ts >= m_ts:
            name = m_stage
        else:
            break
    return name


def build_run_record(
    run_id: str,
    run_type: str,
    status: str,
    harness: BudgetHarness,
    events: Iterable[StageEvent] = (),
    paper_title: str = "",
    depth_mode: str = "",
    error: str | None = None,
    models: dict[str, str] | None = None,
) -> dict:
    """Assemble the per-phase estimation record from a harness (+ its ledger).

    Three levels of detail: per stage (planned vs actual), per call type
    (estimated vs observed tokens per call), and per individual call
    (ordered spend timeline with cumulative cost).
    """
    events = list(events)
    plan = harness.plan
    estimates = plan.all_estimates() if plan is not None else {}
    timing = _stage_timing(events)
    markers = _stage_start_markers(events)

    # ---- per-stage planned (from the final plan) + actual (from the ledger)
    planned_by_stage: dict[str, dict] = {}
    for ct, est in estimates.items():
        stage = STAGE_OF_CALL_TYPE.get(ct, "other")
        agg = planned_by_stage.setdefault(stage, _agg())
        agg["calls"] += est.calls
        agg["input_tokens"] += est.est_input_tokens
        agg["output_tokens"] += est.est_output_tokens
        agg["cost_usd"] += est.est_cost_usd

    actual_by_stage: dict[str, dict] = {}
    for rec in harness.ledger.calls:
        ct = rec.call_type or "other"
        stage = STAGE_OF_CALL_TYPE.get(ct) or _attribute_stage(rec.ts, markers)
        _add(
            actual_by_stage.setdefault(stage, _agg()),
            rec.input_tokens,
            rec.output_tokens,
            rec.estimated_cost_usd,
        )

    stages: dict[str, dict] = {}
    for stage in sorted(
        set(CANONICAL_STAGES) | set(timing) | set(planned_by_stage) | set(actual_by_stage)
    ):
        t = timing.get(stage, {})
        p = _round_agg(planned_by_stage.get(stage, _agg()))
        a = _round_agg(actual_by_stage.get(stage, _agg()))
        stages[stage] = {
            "status": t.get("status", "pending"),
            "started_at": t.get("started_at"),
            "ended_at": t.get("ended_at"),
            "duration_s": t.get("duration_s"),
            "control_plane": stage in CONTROL_STAGES,
            "planned": p if p["calls"] else None,
            "actual": a if a["calls"] else None,
        }

    # ---- per-call-type calibration pairs
    actuals_by_ct: dict[str, dict] = {}
    for rec in harness.ledger.calls:
        ct = rec.call_type or "other"
        agg = actuals_by_ct.setdefault(
            ct,
            {
                "role": rec.role,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            },
        )
        agg["calls"] += 1
        agg["input_tokens"] += rec.input_tokens
        agg["output_tokens"] += rec.output_tokens
        agg["cost_usd"] += rec.estimated_cost_usd

    call_types: dict[str, dict] = {}
    for ct in sorted(set(estimates) | set(actuals_by_ct)):
        est = estimates.get(ct)
        act = actuals_by_ct.get(ct)
        call_types[ct] = {
            "role": est.role if est is not None else (act["role"] if act else "?"),
            "stage": STAGE_OF_CALL_TYPE.get(ct, "other"),
            "planned_calls": est.calls if est is not None else 0,
            "actual_calls": act["calls"] if act else 0,
            "planned_cost_usd": round(est.est_cost_usd, 6) if est is not None else 0.0,
            "actual_cost_usd": round(act["cost_usd"], 6) if act else 0.0,
            "est_input_per_call": (
                est.est_input_tokens // est.calls if est is not None and est.calls else None
            ),
            "est_output_per_call": (
                est.est_output_tokens // est.calls if est is not None and est.calls else None
            ),
            "actual_input_per_call": (
                act["input_tokens"] // act["calls"] if act and act["calls"] else None
            ),
            "actual_output_per_call": (
                act["output_tokens"] // act["calls"] if act and act["calls"] else None
            ),
        }

    # ---- per-call spend timeline (ordered, cumulative)
    timeline: list[dict] = []
    cumulative = 0.0
    for rec in harness.ledger.calls:
        ct = rec.call_type or "other"
        cumulative += rec.estimated_cost_usd
        timeline.append(
            {
                "ts": rec.ts.isoformat(),
                "stage": STAGE_OF_CALL_TYPE.get(ct) or _attribute_stage(rec.ts, markers),
                "call_type": ct,
                "role": rec.role,
                "model": f"{rec.provider}/{rec.model}",
                "input_tokens": rec.input_tokens,
                "output_tokens": rec.output_tokens,
                "cost_usd": round(rec.estimated_cost_usd, 6),
                "cumulative_cost_usd": round(cumulative, 6),
            }
        )

    st = harness.status()
    in_tok, out_tok = harness.ledger.total_tokens()
    return {
        "run_id": run_id,
        "recorded_at": datetime.now(UTC).isoformat(),
        "run_type": run_type,
        "status": status,  # completed | failed | budget_exceeded
        "error": error,
        "depth_mode": depth_mode,
        "paper_title": paper_title,
        # paper characteristics — the exact features the cost model uses
        "paper": harness.stats.model_dump(),
        "budget": {
            "budget_usd": st["budget_usd"],
            "target_usd": st["target_usd"],
            "planned_usd": st["planned_usd"],
            "actual_usd": st["actual_usd"],
            "control_actual_usd": st["control_actual_usd"],
            "projected_final_usd": st["projected_final_usd"],
            "planner_calls_used": st["planner_calls_used"],
            "planner_calls_max": st["planner_calls_max"],
            "adjustments": [
                f"[{a.source}] {a.action}: {a.detail or a.reason}"
                for a in (plan.adjustments if plan is not None else [])
            ],
            "gate_stops": dict(harness.gate_stopped),
        },
        "plan_log": harness.plan_log,
        "stages": stages,
        "call_types": call_types,
        "spend_timeline": timeline,
        "usage_totals": {
            "input_tokens": int(in_tok),
            "output_tokens": int(out_tok),
            "cost_usd": round(harness.ledger.total_cost(), 6),
        },
        "models": models or {},
        "pricing": harness.ledger.pricing,
    }


def upsert_run_record(record: dict, path: Path | None = None) -> Path:
    """One line per run_id: replace the existing line (resumed runs must not
    double-count) or append. Atomic write so concurrent runs don't corrupt."""
    p = path or history_path()
    lines: list[str] = []
    with _write_lock:
        if p.exists():
            try:
                lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
            except OSError as e:
                logger.warning("could not read run history %s: %s", p, e)
                lines = []
        run_id = record.get("run_id")
        replaced = False
        for i, ln in enumerate(lines):
            try:
                if json.loads(ln).get("run_id") == run_id:
                    lines[i] = json.dumps(record, sort_keys=False)
                    replaced = True
                    break
            except json.JSONDecodeError:
                continue  # keep foreign/corrupt lines untouched
        if not replaced:
            lines.append(json.dumps(record, sort_keys=False))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(p)
    return p


def load_run_history(path: Path | None = None) -> list[dict]:
    """All run records, oldest first. Future calibration code starts here."""
    p = path or history_path()
    if not p.exists():
        return []
    out: list[dict] = []
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            logger.warning("skipping corrupt history line in %s", p)
    return out
