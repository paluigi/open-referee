"""Tests for the journal-fit / desk-reject estimator product."""

from __future__ import annotations

import json

import pytest

from open_referee.config import Config, LLMConfig, ModelRole, ProviderConfig, RoleConfig
from open_referee.pipeline.fit import (
    FACTOR_ORDER,
    FIT_WEIGHTS,
    FitPipeline,
    calibrate_probability,
)
from open_referee.pipeline.state import Stage, StageStatus


def fit_config(tmp_path, priced: bool = True, budget: float = 10.0) -> Config:
    cfg = Config()
    cfg.llm = LLMConfig(
        providers={"fake": ProviderConfig(type="fake")},  # type: ignore[arg-type]
        roles={
            "strong": RoleConfig(provider="fake", model="strong-fake"),
            "small": RoleConfig(provider="fake", model="small-fake"),
        },  # type: ignore[arg-type]
        pricing=(
            {
                "fake/strong-fake": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
                "fake/small-fake": {"input_per_mtok": 0.3, "output_per_mtok": 2.5},
            }
            if priced
            else {}
        ),
    )
    cfg.review.max_cost_usd = budget
    return cfg


def fit_response_json() -> str:
    factors = [
        {
            "name": n,
            "score": 0.8 if n in ("scope_fit", "aim_alignment") else 0.4,
            "rationale": f"reason for {n}",
            "evidence": "quote from the manuscript",
        }
        for n in FACTOR_ORDER
    ]
    return json.dumps(
        {
            "factors": factors,
            "reasoning": "## Weighing\n\nScope matches well; novelty is borderline.",
            "risks": ["Novelty below the venue bar"],
            "suggestions": ["Sharpen the contribution statement"],
            "llm_holistic_estimate": 0.55,
        }
    )


# ------------------------------------------------------------- decision model --


def test_calibration_monotone_in_factors():
    base = calibrate_probability({n: 0.5 for n in FIT_WEIGHTS})
    better = calibrate_probability({n: 0.9 for n in FIT_WEIGHTS})
    worse = calibrate_probability({n: 0.1 for n in FIT_WEIGHTS})
    assert 0.0 <= worse < base < better <= 1.0


def test_calibration_missing_factors_default_to_neutral():
    full = calibrate_probability({n: 0.5 for n in FIT_WEIGHTS})
    empty = calibrate_probability({})
    assert empty == pytest.approx(full)


def test_calibration_scope_dominates():
    """Scope/aim carry the largest weights: they move the needle most."""
    scope_up = calibrate_probability({**{n: 0.5 for n in FIT_WEIGHTS}, "scope_fit": 1.0})
    polish_up = calibrate_probability(
        {**{n: 0.5 for n in FIT_WEIGHTS}, "presentation_readiness": 1.0}
    )
    assert scope_up > polish_up


# ----------------------------------------------------------------- pipeline --


async def test_fit_pipeline_e2e(test_config, sample_paper_file, monkeypatch, tmp_path):
    import open_referee.config as cfg_mod

    home = tmp_path / "home"
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_DIR", home)
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", home / "config.yaml")

    cfg = fit_config(tmp_path)
    profile = (
        "Journal of Widget Science: empirical and theoretical widget studies; "
        "novelty bar is high; no case-study-only papers."
    )
    pipeline = FitPipeline(cfg, run_id="fit01")
    from open_referee.pipeline.orchestrator import RolePool

    # script the FakeProvider by wrapping RolePool.build: the pipeline builds
    # its own pool inside run(), so we queue the response at build time
    real_build = RolePool.build

    def scripted_build(cls_cfg, roles, cap_usd=None):
        pool = real_build(cls_cfg, roles, cap_usd=cap_usd)
        pool.providers[ModelRole.STRONG].queue(fit_response_json())
        return pool

    import open_referee.pipeline.orchestrator as orch

    monkeypatch.setattr(orch.RolePool, "build", staticmethod(scripted_build))

    report = await pipeline.run(sample_paper_file, profile)

    assert 0.0 <= report.probability_not_desk_rejected <= 1.0
    assert report.llm_holistic_estimate == 0.55
    assert len(report.factors) == 6
    assert {f.name for f in report.factors} == set(FACTOR_ORDER)
    assert report.calibration["scope_fit"] == FIT_WEIGHTS["scope_fit"]
    assert report.risks and report.suggestions
    assert report.usage.total_input_tokens > 0
    assert report.model_roles["strong"] == "fake/strong-fake"
    assert report.attribution.url == "https://github.com/paluigi/open-referee"

    # persisted artifacts + state
    rdir = home / "runs" / "fit01"
    assert (rdir / "report.json").exists()
    md = (rdir / "report.md").read_text()
    assert "Probability of passing the desk screen" in md
    assert "Built with [open-referee](https://github.com/paluigi/open-referee)" in md
    assert "scope_fit" in md  # calibration weights are visible
    assert "## Build metadata" in md

    from open_referee.pipeline.state import RunState

    state = RunState.load("fit01")
    assert state.run_type == "fit"
    assert state.status == "completed"
    assert state.journal_profile == profile
    assert state.stage_status(Stage.FIT) is StageStatus.DONE

    # run history: per-phase estimation record for the fit product
    from open_referee.pipeline.history import load_run_history

    history = load_run_history(home / "run_history.jsonl")
    rec = next(r for r in history if r["run_id"] == "fit01")
    assert rec["run_type"] == "fit" and rec["status"] == "completed"
    assert rec["paper"]["chars"] > 0
    assert rec["stages"]["fit"]["actual"]["calls"] == 1
    assert rec["call_types"]["fit_assessment"]["actual_input_per_call"] > 0
    assert rec["budget"]["actual_usd"] > 0


async def test_fit_requires_journal_profile(test_config, sample_paper_file, tmp_path):
    cfg = fit_config(tmp_path)
    pipeline = FitPipeline(cfg, run_id="fit02")
    with pytest.raises(ValueError):
        await pipeline.run(sample_paper_file, "   ")


async def test_fit_unpriced_model_is_a_hard_error(
    test_config, sample_paper_file, monkeypatch, tmp_path
):
    import open_referee.config as cfg_mod
    from open_referee.pipeline.budget import PricingError

    home = tmp_path / "home"
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_DIR", home)
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", home / "config.yaml")
    cfg = fit_config(tmp_path, priced=False)
    pipeline = FitPipeline(cfg, run_id="fit03")
    with pytest.raises(PricingError) as ei:
        await pipeline.run(sample_paper_file, "Some venue profile")
    assert "llm.pricing" in str(ei.value)
    from open_referee.pipeline.state import RunState

    state = RunState.load("fit03")
    assert state.status == "failed"
    assert "llm.pricing" in (state.error or "")

    # aborted runs still record paper stats + zero spend for calibration
    from open_referee.pipeline.history import load_run_history

    history = load_run_history(home / "run_history.jsonl")
    rec = next(r for r in history if r["run_id"] == "fit03")
    assert rec["status"] == "failed"
    assert "llm.pricing" in (rec["error"] or "")
    assert rec["budget"]["actual_usd"] == 0.0
    assert rec["paper"]["chars"] > 0


async def test_fit_budget_below_single_call_is_infeasible(
    test_config, sample_paper_file, monkeypatch, tmp_path
):
    import open_referee.config as cfg_mod
    from open_referee.pipeline.budget import BudgetInfeasible

    home = tmp_path / "home"
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_DIR", home)
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", home / "config.yaml")
    cfg = fit_config(tmp_path, priced=True, budget=0.0001)
    pipeline = FitPipeline(cfg, run_id="fit04", budget_usd=0.0001)
    with pytest.raises(BudgetInfeasible):
        await pipeline.run(sample_paper_file, "Some venue profile")
