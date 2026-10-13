import pytest

from open_referee.providers.adapters import FakeProvider, OpenAICompatibleProvider, extract_json
from open_referee.providers.base import ChatMessage, LLMError, ModelSpec
from open_referee.providers.usage import BudgetExceeded, UsageLedger


def spec(**kw):
    base: dict = dict(provider_name="fake", provider_type="openai_compatible", model="m")
    base.update(kw)
    return ModelSpec(**base)


async def test_fake_provider_queue():
    p = FakeProvider(spec(), ['{"a": 1}', "plain"])
    r1 = await p.complete([ChatMessage(role="user", content="x")], json_mode=True)
    r2 = await p.complete([ChatMessage(role="user", content="y")])
    assert r1.text == '{"a": 1}'
    assert r2.text == "plain"
    with pytest.raises(LLMError):
        await p.complete([ChatMessage(role="user", content="z")])


async def test_ledger_records_and_costs():
    led = UsageLedger(
        cap_usd=10.0, pricing={"fake/m": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}}
    )
    s = spec()
    from open_referee.providers.base import LLMResponse, Usage

    await led.record(
        s, "strong", LLMResponse(text="x", usage=Usage(input_tokens=1_000_000, output_tokens=0))
    )
    await led.record(
        s, "small", LLMResponse(text="y", usage=Usage(input_tokens=0, output_tokens=1_000_000))
    )
    assert led.total_cost() == pytest.approx(18.0)
    assert led.by_role()["strong"]["input_tokens"] == 1_000_000


async def test_budget_cap_enforced():
    led = UsageLedger(
        cap_usd=1.0, pricing={"fake/m": {"input_per_mtok": 100.0, "output_per_mtok": 100.0}}
    )
    s = spec()
    with pytest.raises(BudgetExceeded):
        await led.check_budget(s, est_input_tokens=100_000, est_output_tokens=100_000)
    await led.check_budget(s, 0, 0)


async def test_ledger_accumulates_across_checks():
    led = UsageLedger(
        cap_usd=2.0, pricing={"fake/m": {"input_per_mtok": 10.0, "output_per_mtok": 0.0}}
    )
    s = spec()
    await led.record(
        s,
        "small",
        __import__("open_referee.providers.base", fromlist=["LLMResponse"]).LLMResponse(
            text="",
            usage=__import__("open_referee.providers.base", fromlist=["Usage"]).Usage(
                input_tokens=150_000, output_tokens=0
            ),
        ),
    )
    with pytest.raises(BudgetExceeded):
        await led.check_budget(s, 100_000, 0)


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
    assert extract_json("[1, 2]") == [1, 2]
    import json

    with pytest.raises(json.JSONDecodeError):
        extract_json("no json here")


def test_openai_provider_requires_base_url():
    with pytest.raises(LLMError):
        OpenAICompatibleProvider(spec(base_url=None))
