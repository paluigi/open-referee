"""Tests for the redesigned journal-fit estimator: LLM editor factors FIRST,
decision-model predicate verdict after."""

from __future__ import annotations

import json

import pytest

from open_referee.config import Config, LLMConfig, ProviderConfig, RoleConfig
from open_referee.pipeline.fit import FitPipeline
from open_referee.providers.decisions import DecisionAnswer, FakeDecisionProvider


def fit_config(priced: bool = True, budget: float = 10.0) -> Config:
    cfg = Config()
    cfg.llm = LLMConfig(
        providers={
            "fake": ProviderConfig(type="fake"),  # type: ignore[arg-type]
            "verceldec": ProviderConfig(  # type: ignore[arg-type]
                type="decision", base_url="http://fake.local"
            ),
        },
        roles={
            "strong": RoleConfig(provider="fake", model="strong-fake"),
            "small": RoleConfig(provider="fake", model="small-fake"),
            "decision": RoleConfig(provider="verceldec", model="pplx-decider"),
        },
        pricing=(
            {
                "fake/strong-fake": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
                "fake/small-fake": {"input_per_mtok": 0.3, "output_per_mtok": 2.5},
                "verceldec/pplx-decider": {"input_per_mtok": 0.02, "output_per_mtok": 0.0},
            }
            if priced
            else {
                "fake/strong-fake": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
                "fake/small-fake": {"input_per_mtok": 0.3, "output_per_mtok": 2.5},
            }
        ),
    )
    cfg.review.max_cost_usd = budget
    return cfg


def editor_response() -> str:
    factors = [
        {"name": n, "analysis": f"Editor analysis for {n}. " + "evidence " * 30}
        for n in (
            "scope_fit",
            "aim_alignment",
            "methods_format_fit",
            "novelty_threshold",
            "presentation_readiness",
            "policy_ethics",
        )
    ]
    return json.dumps(
        {
            "factors": factors,
            "reasoning": "## Weighing\n\nScope matches well; novelty is borderline.",
            "risks": ["Novelty below the venue bar"],
            "suggestions": ["Sharpen the contribution statement"],
            "llm_estimate": 0.55,  # only used in fallback mode
        }
    )


def script_pool(cfg, with_decider: bool, verdict: float = 0.72):
    """Patch RolePool.build: script the editor response and the decider."""
    from open_referee.config import ModelRole
    from open_referee.pipeline.orchestrator import RolePool
    from open_referee.providers.base import ModelSpec

    real_build = RolePool.build

    def scripted_build(cls_cfg, roles, cap_usd=None):
        pool = real_build(cls_cfg, roles - {ModelRole.DECISION}, cap_usd=cap_usd)
        pool.providers[ModelRole.STRONG].queue(editor_response())
        if with_decider:
            pool.decider = FakeDecisionProvider(model="pplx-decider")
            pool.decider.provider_name = "verceldec"
            pool.decider.queue(
                [
                    DecisionAnswer(
                        type="predicate", name="desk_screen", probability=verdict, confidence=0.8
                    )
                ]
            )
            pool.decider_spec = ModelSpec(
                provider_name="verceldec",
                provider_type="decision",
                model="pplx-decider",
                base_url="http://fake.local",
            )
        return pool

    return scripted_build


PROFILE = (
    "Empirical macroeconomics / forecasting journal: real-time nowcasting valued; "
    "theory-only out of scope. Institutional note: policy relevance a plus."
)


async def test_fit_llm_first_then_decision_verdict(
    test_config, sample_paper_file, monkeypatch, tmp_path
):
    """The protocol order is enforced: the editor LLM runs BEFORE the decision
    model, and the final probability is the decision model's."""
    import open_referee.config as cfg_mod
    from open_referee.pipeline.orchestrator import RolePool

    home = tmp_path / "home"
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_DIR", home)
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", home / "config.yaml")

    cfg = fit_config()
    monkeypatch.setattr(RolePool, "build", staticmethod(script_pool(cfg, with_decider=True)))

    pipeline = FitPipeline(cfg, run_id="fit10")
    report = await pipeline.run(sample_paper_file, PROFILE)

    # verdict comes from the decision model
    assert report.decision is not None
    assert report.decision.source == "decision-model"
    assert report.probability_not_desk_rejected == pytest.approx(0.72)
    assert report.decision.confidence == pytest.approx(0.8)
    # six textual factors, capped
    assert len(report.factors) == 6
    assert all(len(f.analysis) <= 600 * 4 for f in report.factors)
    assert report.risks and report.suggestions
    # both calls flowed through the ledger: editor (strong) + decision
    types = [c["call_type"] for c in pipeline.state.ledger_calls]
    assert "fit_assessment" in types
    assert "dm_fit" in types
    assert types.index("fit_assessment") < types.index("dm_fit")  # order enforced
    assert report.model_roles["decision"] == "verceldec/pplx-decider"

    md = (home / "runs" / "fit10" / "report.md").read_text()
    assert "72%" in md
    assert "decision model over the editor's factor analyses" in md
    assert "Built with [open-referee](https://github.com/paluigi/open-referee)" in md

    from open_referee.pipeline.state import RunState

    state = RunState.load("fit10")
    assert state.status == "completed" and state.run_type == "fit"


async def test_fit_fallback_without_decision_model(
    test_config, sample_paper_file, monkeypatch, tmp_path
):
    """No decision role: the LLM's own estimate is the labelled fallback."""
    import open_referee.config as cfg_mod
    from open_referee.pipeline.orchestrator import RolePool

    home = tmp_path / "home"
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_DIR", home)
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", home / "config.yaml")

    cfg = fit_config()
    cfg.llm.roles.pop("decision")  # type: ignore[union-attr]
    monkeypatch.setattr(RolePool, "build", staticmethod(script_pool(cfg, with_decider=False)))

    pipeline = FitPipeline(cfg, run_id="fit11")
    report = await pipeline.run(sample_paper_file, PROFILE)
    assert report.decision is not None
    assert report.decision.source == "llm-fallback"
    assert report.probability_not_desk_rejected == pytest.approx(0.55)
    assert all(c["call_type"] != "dm_fit" for c in pipeline.state.ledger_calls)


async def test_fit_requires_journal_profile(test_config, sample_paper_file, tmp_path):
    pipeline = FitPipeline(fit_config(), run_id="fit12")
    with pytest.raises(ValueError):
        await pipeline.run(sample_paper_file, "   ")


async def test_fit_unpriced_decision_model_is_a_hard_error(
    test_config, sample_paper_file, monkeypatch, tmp_path
):
    import open_referee.config as cfg_mod
    from open_referee.pipeline.budget import PricingError

    home = tmp_path / "home"
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_DIR", home)
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", home / "config.yaml")
    cfg = fit_config(priced=False)  # decision model has NO pricing entry
    from open_referee.pipeline.orchestrator import RolePool

    monkeypatch.setattr(RolePool, "build", staticmethod(script_pool(cfg, with_decider=True)))
    pipeline = FitPipeline(cfg, run_id="fit13")
    with pytest.raises(PricingError):
        await pipeline.run(sample_paper_file, PROFILE)
