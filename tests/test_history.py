"""Tests for the run-history estimation store: per-stage detail, calibration
pairs, spend timeline, plan trajectory, and upsert semantics."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from open_referee.config import Config, LLMConfig, ModelRole, ProviderConfig, RoleConfig
from open_referee.pipeline.budget import BudgetHarness, DocumentStats
from open_referee.pipeline.history import (
    build_run_record,
    load_run_history,
    upsert_run_record,
)
from open_referee.pipeline.orchestrator import RolePool
from open_referee.pipeline.state import Stage, StageEvent, StageStatus
from open_referee.providers.usage import CallRecord

PRICING = {
    "fake/strong-fake": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
    "fake/small-fake": {"input_per_mtok": 0.3, "output_per_mtok": 2.5},
}


def make_harness(budget: float = 20.0) -> BudgetHarness:
    cfg = Config()
    cfg.llm = LLMConfig(
        providers={"fake": ProviderConfig(type="fake")},  # type: ignore[arg-type]
        roles={
            "strong": RoleConfig(provider="fake", model="strong-fake"),
            "small": RoleConfig(provider="fake", model="small-fake"),
        },  # type: ignore[arg-type]
        pricing=dict(PRICING),
    )
    pool = RolePool.build(cfg, {ModelRole.STRONG, ModelRole.SMALL}, cap_usd=budget)
    h = BudgetHarness(cfg, pool.ledger, pool.specs, budget_usd=budget, auto=True)
    h.stats = DocumentStats(
        chars=60_000, n_sections=8, n_figures=3, n_tables=2, n_references=30, n_claims=40
    )
    return h


def _rec(
    ct: str,
    cost: float,
    in_tok: int = 1000,
    out_tok: int = 500,
    minutes: float = 0.0,
) -> CallRecord:
    return CallRecord(
        ts=datetime.now(UTC) + timedelta(minutes=minutes),
        role="strong",
        provider="fake",
        model="strong-fake",
        input_tokens=in_tok,
        output_tokens=out_tok,
        estimated_cost_usd=cost,
        call_type=ct,
    )


def _book(h: BudgetHarness, rec: CallRecord) -> None:
    h.ledger.calls.append(rec)
    h.note_record(rec)


def _events() -> list[StageEvent]:
    t0 = datetime.now(UTC)
    return [
        StageEvent(run_id="h1", stage=Stage.TRIAGE, status=StageStatus.RUNNING, ts=t0),
        StageEvent(
            run_id="h1", stage=Stage.TRIAGE, status=StageStatus.DONE, ts=t0 + timedelta(seconds=5)
        ),
        StageEvent(
            run_id="h1",
            stage=Stage.VERIFY,
            status=StageStatus.RUNNING,
            ts=t0 + timedelta(seconds=10),
        ),
        # resumed-stage shape: DONE with no RUNNING in this process
        StageEvent(
            run_id="h1", stage=Stage.META, status=StageStatus.DONE, ts=t0 + timedelta(seconds=99)
        ),
    ]


async def test_record_stages_planned_vs_actual():
    h = make_harness()
    h.set_plan(h.deterministic_plan(), reason="preflight-v0")
    _book(h, _rec("triage", 0.02, minutes=0))
    _book(h, _rec("claim_verify", 0.05, in_tok=7000, minutes=1))
    _book(h, _rec("claim_verify", 0.05, in_tok=9000, minutes=2))
    _book(h, _rec("lens_section", 0.01, minutes=3))
    _book(h, _rec("meta", 0.20, minutes=4))

    rec = build_run_record(
        run_id="h1",
        run_type="review",
        status="completed",
        harness=h,
        events=_events(),
        paper_title="Widgets",
        depth_mode="auto",
        models={"strong": "fake/strong-fake"},
    )

    # paper characteristics snapshot
    assert rec["paper"]["chars"] == 60_000
    assert rec["paper"]["n_claims"] == 40

    # stage attribution + detail
    st = rec["stages"]
    assert st["triage"]["actual"] == {
        "calls": 1,
        "input_tokens": 1000,
        "output_tokens": 500,
        "cost_usd": 0.02,
    }
    assert st["triage"]["planned"]["calls"] == 1  # control call planned
    assert st["triage"]["control_plane"] is True
    v = st["verify"]["actual"]
    assert v["calls"] == 3  # claim_verify x2 + lens_section
    assert v["input_tokens"] == 1000 + 7000 + 9000
    assert st["verify"]["planned"]["calls"] >= 3  # plan includes verify call types
    assert st["meta"]["actual"]["calls"] == 1
    assert st["ingest"]["planned"] is None and st["ingest"]["actual"] is None

    # stage timing from events (RUNNING->DONE); resumed stage has no duration
    assert st["triage"]["duration_s"] is not None and st["triage"]["duration_s"] >= 0
    assert st["triage"]["status"] == "done"
    assert st["meta"]["duration_s"] is None
    assert st["meta"]["status"] == "done"


async def test_record_calibration_pairs_and_timeline():
    h = make_harness()
    h.set_plan(h.deterministic_plan(), reason="preflight-v0")
    _book(h, _rec("triage", 0.02, minutes=0))
    _book(h, _rec("claim_verify", 0.05, in_tok=7000, minutes=1))
    _book(h, _rec("claim_verify", 0.05, in_tok=9000, minutes=2))
    _book(h, _rec("other", 0.01, minutes=3))  # untyped/unmodeled call

    rec = build_run_record(
        "h1", "review", "completed", h, events=_events(), models={"strong": "fake/strong-fake"}
    )

    ct = rec["call_types"]
    # calibration pair: estimated per call vs actual average per call
    assert ct["claim_verify"]["actual_calls"] == 2
    assert ct["claim_verify"]["actual_input_per_call"] == 8000
    assert ct["claim_verify"]["est_input_per_call"] is not None
    assert ct["claim_verify"]["stage"] == "verify"
    # unmodeled spend is kept visible with planned_calls 0
    assert ct["other"]["planned_calls"] == 0 and ct["other"]["actual_calls"] == 1

    # timeline: ordered, cumulative, untyped call attributed to the running stage
    tl = rec["spend_timeline"]
    assert [e["call_type"] for e in tl] == ["triage", "claim_verify", "claim_verify", "other"]
    assert tl[-1]["cumulative_cost_usd"] == 0.13
    assert tl[-1]["stage"] == "verify"  # verify was running at minute 3

    # budget block + plan trajectory
    assert rec["budget"]["actual_usd"] == 0.13
    assert rec["budget"]["control_actual_usd"] == 0.02
    assert rec["plan_log"][0]["reason"] == "preflight-v0"
    assert rec["plan_log"][0]["spend_usd_at_change"] == 0.0
    assert rec["plan_log"][0]["knobs"]["max_claims"] >= 6

    # pricing + models snapshot for future interpretation
    assert rec["pricing"]["fake/strong-fake"]["input_per_mtok"] == 3.0
    assert rec["models"]["strong"] == "fake/strong-fake"


async def test_plan_log_tracks_every_version_with_spend_at_change():
    h = make_harness()
    p0 = h.deterministic_plan()
    h.set_plan(p0, reason="preflight-v0")
    _book(h, _rec("triage", 0.02))
    p1 = h.validate_plan(
        p0.model_copy(update={"knobs": p0.knobs.model_copy(update={"max_claims": 10})}),
        source="planner",
    )
    h.set_plan(p1, reason="claims-milestone")
    _book(h, _rec("claim_inventory", 0.03))
    h.set_plan(
        h.apply_option(
            type("O", (), {"action": "drop_defense", "label": "Skip defense gate"})(),
            reason="overrun",
        ),
        reason="steering:drop_defense",
    )

    rec = build_run_record("h1", "review", "completed", h)
    reasons = [e["reason"] for e in rec["plan_log"]]
    assert reasons == ["preflight-v0", "claims-milestone", "steering:drop_defense"]
    assert rec["plan_log"][1]["spend_usd_at_change"] == 0.02
    assert rec["plan_log"][2]["spend_usd_at_change"] == 0.05
    assert rec["plan_log"][2]["knobs"]["defense_round"] is False


async def test_upsert_one_line_per_run(tmp_path):
    path = tmp_path / "run_history.jsonl"
    rec1 = {"run_id": "a", "status": "failed", "budget": {"actual_usd": 0.1}}
    upsert_run_record(rec1, path)
    rec1b = {"run_id": "a", "status": "completed", "budget": {"actual_usd": 0.9}}
    upsert_run_record(rec1b, path)  # resumed run updates its line
    upsert_run_record({"run_id": "b", "status": "completed"}, path)

    lines = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["run_id"] == "a" and lines[0]["status"] == "completed"
    assert lines[1]["run_id"] == "b"

    loaded = load_run_history(path)
    assert [r["run_id"] for r in loaded] == ["a", "b"]
    assert load_run_history(tmp_path / "missing.jsonl") == []


async def test_upsert_tolerates_corrupt_lines(tmp_path):
    path = tmp_path / "run_history.jsonl"
    path.write_text('{"run_id": "a"}\nNOT JSON AT ALL\n')
    upsert_run_record({"run_id": "b", "status": "completed"}, path)
    loaded = load_run_history(path)
    ids = [r["run_id"] for r in loaded]
    assert "a" in ids and "b" in ids  # corrupt line skipped on read, kept on write
    assert "NOT JSON" in path.read_text()
