"""LLM provider layer.

Uniform async chat interface over five backend types plus a FakeProvider for
tests. Includes structured-output helper with a parse-repair retry loop and a
UsageLedger enforcing a global spend cap.
"""

from open_referee.providers.base import (
    ChatMessage,
    ChatProvider,
    LLMError,
    LLMResponse,
    ModelSpec,
    Usage,
)
from open_referee.providers.factory import build_provider, provider_for_role
from open_referee.providers.usage import BudgetExceeded, UsageLedger

__all__ = [
    "ChatMessage",
    "ChatProvider",
    "LLMError",
    "LLMResponse",
    "ModelSpec",
    "Usage",
    "UsageLedger",
    "BudgetExceeded",
    "build_provider",
    "provider_for_role",
]
