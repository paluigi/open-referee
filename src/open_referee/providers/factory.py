"""Build providers from config and resolve roles to providers."""

from __future__ import annotations

from open_referee.config import Config, LLMConfig, ModelRole, ProviderType, RoleConfig
from open_referee.providers.base import ChatProvider, LLMError, ModelSpec
from open_referee.providers.usage import UsageLedger


def build_provider(llm_cfg: LLMConfig, role_cfg: RoleConfig) -> ChatProvider:
    pcfg = llm_cfg.providers.get(role_cfg.provider)
    if pcfg is None:
        raise LLMError(
            f"Role references provider '{role_cfg.provider}' which is not defined "
            f"under llm.providers (available: {sorted(llm_cfg.providers)})"
        )
    spec = ModelSpec(
        provider_name=role_cfg.provider,
        provider_type=pcfg.type.value,
        model=role_cfg.model,
        base_url=pcfg.base_url,
        api_key=pcfg.resolved_api_key(),
        temperature=role_cfg.temperature,
        max_tokens=role_cfg.max_tokens,
        timeout_s=llm_cfg.request_timeout_s,
        max_retries=llm_cfg.max_retries,
    )
    return _from_spec(spec)


def _from_spec(spec: ModelSpec) -> ChatProvider:
    # Local import keeps adapter import costs off the config-only path.
    from open_referee.providers.adapters import (
        AnthropicProvider,
        FakeProvider,
        GeminiProvider,
        OllamaProvider,
        OpenAICompatibleProvider,
    )

    ptype = ProviderType(spec.provider_type)
    if ptype is ProviderType.FAKE:
        return FakeProvider(spec)
    if ptype in (ProviderType.OPENAI_COMPATIBLE, ProviderType.VLLM):
        return OpenAICompatibleProvider(spec)
    if ptype is ProviderType.OLLAMA:
        return OllamaProvider(spec)
    if ptype is ProviderType.ANTHROPIC:
        if not spec.api_key:
            raise LLMError(f"Provider '{spec.provider_name}' has no API key resolved")
        return AnthropicProvider(spec)
    if ptype is ProviderType.GEMINI:
        if not spec.api_key:
            raise LLMError(f"Provider '{spec.provider_name}' has no API key resolved")
        return GeminiProvider(spec)
    raise LLMError(f"Unknown provider type: {spec.provider_type}")


def provider_for_role(
    cfg: Config,
    role: ModelRole,
    ledger: UsageLedger | None = None,
) -> tuple[ChatProvider, ModelSpec]:
    """Resolve a role to a live provider; registers it with the ledger if given."""
    role_cfg = cfg.llm.role(role)
    provider = build_provider(cfg.llm, role_cfg)
    pcfg = cfg.llm.providers[role_cfg.provider]
    spec = ModelSpec(
        provider_name=role_cfg.provider,
        provider_type=pcfg.type.value,
        model=role_cfg.model,
        base_url=pcfg.base_url,
        temperature=role_cfg.temperature,
        max_tokens=role_cfg.max_tokens,
        timeout_s=cfg.llm.request_timeout_s,
        max_retries=cfg.llm.max_retries,
    )
    if ledger is not None:
        spec.api_key = pcfg.resolved_api_key()
    return provider, spec


def fake_provider(model: str = "fake-model", responses: list[str] | None = None) -> ChatProvider:
    from open_referee.providers.adapters import FakeProvider

    spec = ModelSpec(provider_name="fake", provider_type="openai_compatible", model=model)
    return FakeProvider(spec, responses)
