"""Provider adapters.

All adapters share retry/backoff and convert backend-specific payloads into
:class:`LLMResponse`. Images on messages are only honored by vision-capable
calls; text-only backends raise if images are present.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from collections.abc import Sequence
from typing import Any

import httpx

from open_referee.providers.base import (
    ChatMessage,
    ChatProvider,
    LLMError,
    LLMResponse,
    ModelSpec,
    Usage,
)


class HTTPChatProvider(ChatProvider):
    """Base with shared retry/backoff logic."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._client = httpx.AsyncClient(timeout=spec.timeout_s)

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(
        self, messages: Sequence[ChatMessage], *, json_mode: bool = False
    ) -> LLMResponse:
        delay = 1.0
        last_err: Exception | None = None
        for attempt in range(self.spec.max_retries + 1):
            try:
                return await self._complete_once(messages, json_mode=json_mode)
            except (httpx.HTTPError, LLMError) as e:
                last_err = e
                if attempt < self.spec.max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
        raise LLMError(
            f"{self.spec.provider_name}/{self.spec.model} failed after retries: {last_err}"
        )

    async def _complete_once(
        self, messages: Sequence[ChatMessage], *, json_mode: bool
    ) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError


def _check_no_images(messages: Sequence[ChatMessage]) -> None:
    if any(m.images for m in messages):
        raise LLMError("This provider does not support image inputs")


def _usage(inp: Any, out: Any) -> Usage:
    return Usage(input_tokens=int(inp or 0), output_tokens=int(out or 0))


class OpenAICompatibleProvider(HTTPChatProvider):
    """openai_compatible and vllm backends (and most aggregators)."""

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        if not spec.base_url:
            raise LLMError(f"Provider '{spec.provider_name}' needs a base_url")
        self._url = spec.base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {spec.api_key}"} if spec.api_key else {}

    async def _complete_once(
        self, messages: Sequence[ChatMessage], *, json_mode: bool
    ) -> LLMResponse:
        def content_of(m: ChatMessage) -> Any:
            if m.images:
                parts: list[dict[str, Any]] = [{"type": "text", "text": m.content}]
                for img in m.images:
                    parts.append({"type": "image_url", "image_url": {"url": img}})
                return parts
            return m.content

        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": [{"role": m.role, "content": content_of(m)} for m in messages],
            "temperature": self.spec.temperature,
            "max_tokens": self.spec.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = await self._client.post(self._url, json=payload, headers=self._headers)
        r.raise_for_status()
        data = r.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            return LLMResponse(
                text=text,
                usage=_usage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
                raw=data,
            )
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Malformed response from {self.spec.provider_name}: {e}") from e


class OllamaProvider(HTTPChatProvider):
    """Native Ollama /api/chat (no API key needed)."""

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        if not spec.base_url:
            spec.base_url = "http://localhost:11434"
            self.spec = spec
        self._url = spec.base_url.rstrip("/") + "/api/chat"

    async def _complete_once(
        self, messages: Sequence[ChatMessage], *, json_mode: bool
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    **({"images": [i.split(",")[-1] for i in m.images]} if m.images else {}),
                }
                for m in messages
            ],
            "stream": False,
            "format": "json" if json_mode else None,
            "options": {"temperature": self.spec.temperature, "num_predict": self.spec.max_tokens},
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        r = await self._client.post(self._url, json=payload)
        r.raise_for_status()
        data = r.json()
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            usage=_usage(data.get("prompt_eval_count"), data.get("eval_count")),
            raw=data,
        )


class AnthropicProvider(HTTPChatProvider):
    URL = "https://api.anthropic.com/v1/messages"

    async def _complete_once(
        self, messages: Sequence[ChatMessage], *, json_mode: bool
    ) -> LLMResponse:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.images:
                blocks: list[dict[str, Any]] = []
                for img in m.images:
                    header, b64 = _split_data_url(img)
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": header, "data": b64},
                        }
                    )
                blocks.append({"type": "text", "text": m.content})
                turns.append({"role": m.role, "content": blocks})
            else:
                turns.append({"role": m.role, "content": m.content})
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "max_tokens": self.spec.max_tokens,
            "temperature": self.spec.temperature,
            "messages": turns,
        }
        if system:
            payload["system"] = system
        if json_mode:
            # Anthropic has no json mode; instruct + prefill assistant brace.
            payload["messages"] = turns + [{"role": "assistant", "content": "{"}]
        headers = {
            "x-api-key": self.spec.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        r = await self._client.post(self.URL, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []))
        if json_mode and text and not text.lstrip().startswith("{"):
            text = "{" + text
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            usage=_usage(usage.get("input_tokens"), usage.get("output_tokens")),
            raw=data,
        )


class GeminiProvider(HTTPChatProvider):
    def _url(self, method: str) -> str:
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.spec.model}:{method}?key={self.spec.api_key}"
        )

    async def _complete_once(
        self, messages: Sequence[ChatMessage], *, json_mode: bool
    ) -> LLMResponse:
        contents: list[dict[str, Any]] = []
        system_parts: list[str] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            parts: list[dict[str, Any]] = []
            for img in m.images:
                header, b64 = _split_data_url(img)
                parts.append({"inline_data": {"mime_type": header, "data": b64}})
            parts.append({"text": m.content})
            contents.append({"role": "model" if m.role == "assistant" else "user", "parts": parts})
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.spec.temperature,
                "maxOutputTokens": self.spec.max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        r = await self._client.post(self._url("generateContent"), json=payload)
        r.raise_for_status()
        data = r.json()
        try:
            cand = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in cand)
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Malformed Gemini response: {e}") from e
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            usage=_usage(usage.get("promptTokenCount"), usage.get("candidatesTokenCount")),
            raw=data,
        )


class FakeProvider(HTTPChatProvider):
    """Scripted provider for tests: pops queued responses in order."""

    def __init__(self, spec: ModelSpec, responses: list[str] | None = None):
        super().__init__(spec)
        self.responses = list(responses or [])
        self.calls: list[list[ChatMessage]] = []

    def queue(self, *texts: str) -> None:
        self.responses.extend(texts)

    async def _complete_once(
        self, messages: Sequence[ChatMessage], *, json_mode: bool
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if not self.responses:
            raise LLMError("FakeProvider queue empty")
        text = self.responses.pop(0)
        return LLMResponse(
            text=text,
            usage=_usage(sum(len(m.content) // 4 for m in messages), len(text) // 4),
            raw={},
        )


def _split_data_url(data_url: str) -> tuple[str, str]:
    """Return (mime_type, base64_payload) from a data: URL (or raw base64)."""
    m = re.match(r"data:([^;]+);base64,(.*)", data_url, flags=re.S)
    if m:
        return m.group(1), m.group(2)
    return "image/png", data_url


def encode_image_bytes(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM reply (fenced, prefixed, trailing)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # first balanced object/array
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise json.JSONDecodeError("no valid JSON found", text, 0)
