import yaml

from open_referee.config import (
    Config,
    ModelRole,
    ProviderType,
    config_from_dict,
    example_config_text,
    load_config,
    save_config,
)


def test_defaults():
    cfg = Config()
    assert cfg.review.max_cost_usd == 10.0
    assert cfg.search.order == ["tavily", "tinyfish", "brave"]


def test_round_trip_dict_yaml(tmp_path):
    raw = {
        "llm": {
            "providers": {
                "x": {"type": "openai_compatible", "base_url": "http://x", "api_key_env": "X_KEY"}
            },
            "roles": {
                "strong": {"provider": "x", "model": "m1"},
                "small": {"provider": "x", "model": "m2"},
            },
        },
        "openalex": {"api_key_env": "OA", "email": "a@b.c"},
        "review": {"max_cost_usd": 3.5},
    }
    cfg = config_from_dict(raw)
    assert cfg.llm.role(ModelRole.STRONG).model == "m1"
    assert cfg.openalex.email == "a@b.c"
    assert cfg.review.max_cost_usd == 3.5
    # save -> load round trip
    p = save_config(cfg, tmp_path / "config.yaml")
    cfg2 = load_config(p)
    assert cfg2.llm.role(ModelRole.STRONG).model == "m1"
    assert cfg2.review.max_cost_usd == 3.5
    # secrets never persisted as values
    assert "sk-" not in p.read_text()


def test_role_coercion_and_provider_types():
    cfg = config_from_dict({"llm": {"roles": {"STRONG": {"provider": "p", "model": "m"}}}})
    assert cfg.llm.role(ModelRole.STRONG).model == "m"
    assert ProviderType("vllm").value == "vllm"


def test_example_config_parses():

    raw = yaml.safe_load(example_config_text())
    cfg = config_from_dict(raw)
    assert len(cfg.llm.providers) == 5
    assert cfg.search.order[0] == "tavily"


def test_search_enabled_engines_respects_order_and_keys(monkeypatch):
    from open_referee.config import SearchConfig

    sc = SearchConfig(order=["brave", "tavily", "tinyfish"])
    # no keys set -> none enabled
    assert sc.enabled_engines() == []
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    assert sc.enabled_engines() == ["tavily"]
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    assert sc.enabled_engines() == ["brave", "tavily"]  # user order wins
