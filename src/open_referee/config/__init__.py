"""Configuration model, loading, and validation.

One YAML file is the single source of truth (default
``~/.open-referee/config.yaml``). Secrets are never stored in the file: the
YAML holds the *name* of the environment variable holding each API key, and
values are resolved from the environment at load time. ``.env`` files are
honored at the CLI/server entry point via python-dotenv-compatible loading.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_DIR = Path(os.environ.get("OPEN_REFEREE_HOME", "~/.open-referee")).expanduser()
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"


class ProviderType(str, Enum):
    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    VLLM = "vllm"
    FAKE = "fake"


class ModelRole(str, Enum):
    STRONG = "strong"
    SMALL = "small"
    VISION = "vision"


class ProviderConfig(BaseModel):
    """A named, reusable LLM endpoint."""

    type: ProviderType
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = Field(default=None, exclude=True, repr=False)

    def resolved_api_key(self) -> str | None:
        if self.api_key is not None:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


class RoleConfig(BaseModel):
    """A model role bound to a specific provider + model."""

    provider: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 4096


class LLMConfig(BaseModel):
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    roles: dict[ModelRole, RoleConfig] = Field(default_factory=dict)
    pricing: dict[str, dict[str, float]] = Field(default_factory=dict)
    request_timeout_s: float = 300.0
    max_retries: int = 3

    @field_validator("roles", mode="before")
    @classmethod
    def _coerce_roles(cls, v: Any) -> Any:
        if isinstance(v, dict):
            out = {}
            for k, val in v.items():
                if isinstance(k, Enum):
                    key = k.value
                elif isinstance(k, str):
                    key = k.lower()
                else:
                    key = k
                out[key] = val
            return out
        return v

    def role(self, role: ModelRole) -> RoleConfig:
        try:
            return self.roles[role]
        except KeyError as e:
            raise KeyError(
                f"Model role '{role.value}' is not configured. "
                f"Configured roles: {sorted(r.value for r in self.roles)}"
            ) from e


class OpenAlexConfig(BaseModel):
    api_key_env: str = "OPENALEX_API_KEY"
    email: str | None = None

    def resolved_api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


class CrossrefConfig(BaseModel):
    email: str | None = None


class SearchEngineConfig(BaseModel):
    api_key_env: str | None = None
    enabled: bool = True

    def resolved_api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class SearchConfig(BaseModel):
    """Web-search engines in user-defined priority order (failover top-down)."""

    order: list[str] = Field(default_factory=lambda: ["tavily", "tinyfish", "brave"])
    tavily: SearchEngineConfig = SearchEngineConfig(api_key_env="TAVILY_API_KEY")
    tinyfish: SearchEngineConfig = SearchEngineConfig(api_key_env="TINYFISH_API_KEY")
    brave: SearchEngineConfig = SearchEngineConfig(api_key_env="BRAVE_API_KEY")

    def enabled_engines(self) -> list[str]:
        by_name = {"tavily": self.tavily, "tinyfish": self.tinyfish, "brave": self.brave}
        out: list[str] = []
        for name in self.order:
            cfg = by_name.get(name.strip().lower())
            if cfg and cfg.enabled and (cfg.api_key_env is None or cfg.resolved_api_key()):
                out.append(name.strip().lower())
        return out


class PeerReviewSourcesConfig(BaseModel):
    """Review-community sites scanned for prior discussion (PubPeer etc.)."""

    pubpeer: bool = True
    openreview: bool = True
    prereview: bool = True
    extra_sites: list[str] = Field(default_factory=list)


class DepthPreset(BaseModel):
    """Call-budget preset. 'deep' approaches the hundreds-of-calls regime of
    hosted deep-review tools; 'fast' trades depth for cost/latency."""

    max_claims: int = 60
    per_section_lenses: bool = True
    whole_paper_passes: bool = True
    defense_round: bool = True
    max_artifacts: int = 12


class ReviewConfig(BaseModel):
    max_cost_usd: float = 10.0
    max_parallel_calls: int = 8
    max_user_literature_docs: int = 10
    depth: str = "standard"  # fast | standard | deep

    def depth_preset(self) -> DepthPreset:
        presets = {
            "fast": DepthPreset(
                max_claims=20,
                per_section_lenses=False,
                whole_paper_passes=True,
                defense_round=False,
                max_artifacts=6,
            ),
            "standard": DepthPreset(),
            "deep": DepthPreset(
                max_claims=150,
                per_section_lenses=True,
                whole_paper_passes=True,
                defense_round=True,
                max_artifacts=25,
            ),
        }
        key = self.depth.lower().strip()
        if key not in presets:
            raise ValueError(f"Unknown depth preset '{self.depth}' (fast|standard|deep)")
        return presets[key]


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8410
    auth_token_env: str | None = None  # optional bearer token for LAN exposure


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    openalex: OpenAlexConfig = Field(default_factory=OpenAlexConfig)
    crossref: CrossrefConfig = Field(default_factory=CrossrefConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    peer_review_sources: PeerReviewSourcesConfig = Field(default_factory=PeerReviewSourcesConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def ensure_dirs(self) -> None:
        DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        (DEFAULT_CONFIG_DIR / "runs").mkdir(exist_ok=True)


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Shared parsing core: dict -> Config. YAML loading uses this too."""
    return Config.model_validate(raw)


def load_config(path: Path | str | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        return Config()
    raw = yaml.safe_load(p.read_text()) or {}
    return config_from_dict(raw)


def save_config(cfg: Config, path: Path | str | None = None) -> Path:
    """Write the config back to YAML. Secrets are stored as env-var names only."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.model_dump(mode="json", exclude_none=True)
    # pydantic serializes enum dict keys as "modelrole.strong"; restore plain values
    roles = data.get("llm", {}).get("roles")
    if isinstance(roles, dict):
        data["llm"]["roles"] = {k.split(".", 1)[-1]: v for k, v in roles.items()}
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return p


def example_config_text() -> str:
    return yaml.safe_dump(
        {
            "llm": {
                "providers": {
                    "openrouter": {
                        "type": "openai_compatible",
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_key_env": "OPENROUTER_API_KEY",
                    },
                    "anthropic": {"type": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
                    "gemini": {"type": "gemini", "api_key_env": "GEMINI_API_KEY"},
                    "ollama": {"type": "ollama", "base_url": "http://localhost:11434"},
                    "vllm": {"type": "vllm", "base_url": "http://localhost:8000/v1"},
                },
                "roles": {
                    "strong": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.5"},
                    "small": {"provider": "ollama", "model": "llama3.1:8b"},
                    "vision": {"provider": "gemini", "model": "gemini-2.5-flash"},
                },
                "pricing": {
                    "openrouter/anthropic/claude-sonnet-4.5": {
                        "input_per_mtok": 3.0,
                        "output_per_mtok": 15.0,
                    },
                    "ollama/llama3.1:8b": {"input_per_mtok": 0.0, "output_per_mtok": 0.0},
                },
            },
            "openalex": {"api_key_env": "OPENALEX_API_KEY", "email": "you@institution.edu"},
            "crossref": {"email": "you@institution.edu"},
            "search": {
                "order": ["tavily", "tinyfish", "brave"],
                "tavily": {"api_key_env": "TAVILY_API_KEY", "enabled": True},
                "tinyfish": {"api_key_env": "TINYFISH_API_KEY", "enabled": True},
                "brave": {"api_key_env": "BRAVE_API_KEY", "enabled": False},
            },
            "peer_review_sources": {"pubpeer": True, "openreview": True, "prereview": True},
            "review": {"max_cost_usd": 10.0, "max_parallel_calls": 8},
            "server": {"host": "127.0.0.1", "port": 8410},
        },
        sort_keys=False,
    )
