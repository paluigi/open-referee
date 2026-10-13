"""Core provider types."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str
    images: list[str] = Field(default_factory=list)  # data URLs / base64, vision role


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    usage: Usage = Field(default_factory=Usage)
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)


class ModelSpec(BaseModel):
    """Fully resolved model reference (provider + model + params)."""

    provider_name: str
    provider_type: str
    model: str
    base_url: str | None = None
    api_key: str | None = Field(default=None, exclude=True, repr=False)
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_s: float = 300.0
    max_retries: int = 3


class LLMError(RuntimeError):
    pass


@runtime_checkable
class ChatProvider(Protocol):
    """Uniform async chat interface implemented by every adapter."""

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
    ) -> LLMResponse: ...

    async def close(self) -> None: ...
