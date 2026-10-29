"""Decision-model provider: typed evaluation questions against gateways
implementing the /v1/decisions API (verified on the Vercel AI Gateway).

Decision models do not generate free text: they evaluate one shared input
against predicate / choice / score questions and return typed answers with
probability distributions in a single forward pass. They are NOT
ChatProviders and cannot fill chat roles — they fill ModelRole.DECISION.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel, Field

from open_referee.providers.base import LLMError, Usage


class Choice(BaseModel):
    value: str
    description: str = ""


class Level(BaseModel):
    label: str
    description: str = ""


class PredicateQuestion(BaseModel):
    """Probability that the instruction's statement holds for the input."""

    type: str = "predicate"
    name: str
    instructions: str


class ChoiceQuestion(BaseModel):
    """Pick one option; answer includes the full probability distribution."""

    type: str = "choice"
    name: str
    instructions: str
    choices: list[Choice]


class ScoreQuestion(BaseModel):
    """Position on an ordered rubric; answer includes per-level distribution."""

    type: str = "score"
    name: str
    instructions: str
    levels: list[Level]


Question = PredicateQuestion | ChoiceQuestion | ScoreQuestion


class DecisionAnswer(BaseModel):
    type: str
    name: str | None = None
    probability: float | None = None  # predicate
    choice: str | None = None  # choice
    score: float | None = None  # score (level index)
    confidence: float | None = None
    probabilities: list[dict[str, Any]] = Field(default_factory=list)

    def by_name(self, answers: list[DecisionAnswer], name: str) -> DecisionAnswer | None:
        for a in answers:
            if a.name == name:
                return a
        return None


class DecisionResult(BaseModel):
    answers: list[DecisionAnswer] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    def answer(self, name: str) -> DecisionAnswer | None:
        for a in self.answers:
            if a.name == name:
                return a
        return None


def _question_payload(q: Question) -> dict[str, Any]:
    d: dict[str, Any] = {"type": q.type, "name": q.name, "instructions": q.instructions}
    if isinstance(q, ChoiceQuestion):
        d["choices"] = [
            {"value": c.value, **({"description": c.description} if c.description else {})}
            for c in q.choices
        ]
    elif isinstance(q, ScoreQuestion):
        d["levels"] = [
            {"label": lv.label, **({"description": lv.description} if lv.description else {})}
            for lv in q.levels
        ]
    return d


class DecisionProvider:
    """HTTP client for POST {base_url}/decisions (OpenAI-gateway-style auth)."""

    def __init__(
        self,
        provider_name: str,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout_s: float = 240.0,
        max_retries: int = 4,
    ):
        self.provider_name = provider_name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}" if api_key else "",
                "Content-Type": "application/json",
            },
            timeout=timeout_s,
        )

    async def decide(self, input: str, questions: Sequence[Question]) -> DecisionResult:
        payload = {
            "model": self.model,
            "input": input,
            "questions": [_question_payload(q) for q in questions],
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post("/decisions", json=payload)
                if resp.status_code >= 400:
                    raise LLMError(
                        f"{self.provider_name}/{self.model} decision call failed: "
                        f"{resp.status_code} {resp.text[:300]}"
                    )
                data = resp.json()
                usage = Usage(
                    input_tokens=int(data.get("usage", {}).get("input_tokens", 0)),
                    output_tokens=int(data.get("usage", {}).get("output_tokens", 0)),
                )
                answers = [DecisionAnswer(**a) for a in data.get("answers", [])]
                return DecisionResult(answers=answers, usage=usage, raw=data)
            except (httpx.HTTPError, LLMError) as e:
                last_err = e
                if attempt < self.max_retries:
                    import asyncio

                    await asyncio.sleep(1.5 * (2**attempt))
        raise LLMError(
            f"{self.provider_name}/{self.model} decision call failed after retries: {last_err}"
        )

    async def close(self) -> None:
        await self._client.aclose()


class FakeDecisionProvider(DecisionProvider):
    """Scripted decision provider for tests: queue(answers-list) per call."""

    def __init__(self, model: str = "fake-decider"):
        super().__init__("fake", model, "http://fake.local", api_key=None)
        self._queue: list[list[DecisionAnswer]] = []
        self.calls: list[dict] = []

    def queue(self, answers: list[DecisionAnswer]) -> None:
        self._queue.append(answers)

    async def decide(self, input: str, questions: Sequence[Question]) -> DecisionResult:
        self.calls.append({"input": input, "questions": [q.name for q in questions]})
        if not self._queue:
            raise LLMError("FakeDecisionProvider queue empty")
        answers = self._queue.pop(0)
        in_tok = max(1, len(input) // 4)
        return DecisionResult(answers=answers, usage=Usage(input_tokens=in_tok, output_tokens=1))
