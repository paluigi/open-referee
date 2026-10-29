"""Two-phase JSON completion: reasoning-exhaustion escalation."""

from __future__ import annotations

import pytest

from open_referee.config import Config, LLMConfig, ModelRole, ProviderConfig, RoleConfig
from open_referee.pipeline.orchestrator import RolePool
from open_referee.providers.base import LLMResponse, ModelSpec, Usage


def _spec(max_tokens: int = 4096) -> ModelSpec:
    return ModelSpec(
        provider_name="featherless",
        provider_type="openai_compatible",
        model="zai-org/GLM-5.3-Flash",
        max_tokens=max_tokens,
    )


class ScriptedProvider:
    """Returns queued (json_mode, text, output_tokens) tuples and records the
    spec snapshot at each call."""

    def __init__(self, spec, script: list[tuple[bool, str, int]]):
        self.spec = spec
        self.script = list(script)
        self.calls: list[dict] = []

    async def complete(self, messages, json_mode=False):

        self.calls.append(
            {
                "json_mode": json_mode,
                "spec": self.spec.model_copy(),
                "n_messages": len(messages),
                "user_tail": messages[-1].content[-80:],
            }
        )
        jmode, text, out_tok = self.script.pop(0)
        assert jmode == json_mode, f"expected json_mode={jmode}, got {json_mode}"
        return LLMResponse(text=text, usage=Usage(input_tokens=100, output_tokens=out_tok))

    async def close(self):
        pass


def _pool(provider, spec, two_phase: str = "auto") -> RolePool:
    cfg = Config()
    cfg.llm = LLMConfig(
        providers={"featherless": ProviderConfig(type="fake")},  # type: ignore[arg-type]
        roles={"strong": RoleConfig(provider="featherless", model="m")},  # type: ignore[arg-type]
        json_two_phase=two_phase,
        json_phase_reasoning_effort="low",
        json_phase_max_tokens=16384,
    )
    pool = RolePool.__new__(RolePool)
    pool.cfg = cfg
    from open_referee.providers.usage import UsageLedger

    pool.ledger = UsageLedger(100.0, {})
    pool.providers = {ModelRole.STRONG: provider}
    pool.specs = {ModelRole.STRONG: spec}
    pool.harness = None
    return pool


@pytest.mark.asyncio
async def test_auto_escalates_on_empty_content():
    spec = _spec()
    provider = ScriptedProvider(
        spec,
        [
            (True, "", 4096),  # single-call attempt: exhausted (empty, cap)
            (True, "", 200),  # low-effort retry: also empty here
            (False, "The analysis: the theorem lacks a boundedness step...", 9000),
            (True, '{"ok": true, "n": 2}', 120),
        ],
    )
    pool = _pool(provider, spec)
    out = await pool.complete_json(ModelRole.STRONG, "s", "u", call_type="meta")
    assert out == {"ok": True, "n": 2}
    modes = [c["json_mode"] for c in provider.calls]
    assert modes == [True, True, False, True]  # attempt, low retry, reasoning, emission
    # low retry and phase-2 emission run with the low-effort control
    assert provider.calls[1]["spec"].reasoning_effort == "low"
    assert provider.calls[3]["spec"].reasoning_effort == "low"
    # phase 1 reasoning ran with the generous phase budget
    assert provider.calls[2]["spec"].call_max_tokens == 16384
    # phase 2 prompt carries the analysis
    assert (
        "prior analysis" in provider.calls[2]["user_tail"]
        or len(provider.calls[2]["user_tail"]) > 0
    )
    # all four calls ledger-recorded under the same call type
    assert len(pool.ledger.calls) == 4
    assert all(c.call_type == "meta" for c in pool.ledger.calls)


@pytest.mark.asyncio
async def test_valid_single_call_never_escalates():
    spec = _spec()
    provider = ScriptedProvider(spec, [(True, '{"ok": 1}', 50)])
    pool = _pool(provider, spec)
    out = await pool.complete_json(ModelRole.STRONG, "s", "u", call_type="meta")
    assert out == {"ok": 1}
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_never_mode_keeps_single_call_ladder():
    spec = _spec()
    provider = ScriptedProvider(
        spec,
        [
            (True, "", 4096),  # exhausted
            (True, '{"ok": 2}', 30),  # repair retry
        ],
    )
    pool = _pool(provider, spec, two_phase="never")
    out = await pool.complete_json(ModelRole.STRONG, "s", "u", call_type="meta")
    assert out == {"ok": 2}
    assert [c["json_mode"] for c in provider.calls] == [True, True]


@pytest.mark.asyncio
async def test_always_mode_skips_single_call():
    spec = _spec()
    provider = ScriptedProvider(
        spec,
        [
            (False, "analysis text", 800),
            (True, '{"ok": 3}', 90),
        ],
    )
    pool = _pool(provider, spec, two_phase="always")
    out = await pool.complete_json(ModelRole.STRONG, "s", "u", call_type="meta")
    assert out == {"ok": 3}
    assert [c["json_mode"] for c in provider.calls] == [False, True]


@pytest.mark.asyncio
async def test_phase1_also_empty_falls_back_to_repair():
    spec = _spec()
    provider = ScriptedProvider(
        spec,
        [
            (True, "", 4096),  # single attempt exhausted
            (True, "", 300),  # low-effort retry empty
            (False, "", 24576),  # phase-1 reasoning also exhausted
            (True, '{"ok": 4}', 20),  # repair ladder saves it
        ],
    )
    pool = _pool(provider, spec)
    out = await pool.complete_json(ModelRole.STRONG, "s", "u", call_type="meta")
    assert out == {"ok": 4}
    assert len(pool.ledger.calls) == 4


@pytest.mark.asyncio
async def test_spec_restored_after_two_phase():
    spec = _spec()
    provider = ScriptedProvider(
        spec,
        [
            (True, "", 4096),
            (True, "", 100),
            (False, "analysis", 500),
            (True, '{"ok": 5}', 10),
        ],
    )
    pool = _pool(provider, spec)
    await pool.complete_json(ModelRole.STRONG, "s", "u", call_type="meta")
    assert provider.spec is spec
    assert provider.spec.reasoning_effort is None
    assert provider.spec.call_max_tokens is None
