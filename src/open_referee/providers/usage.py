"""Usage ledger with token accounting, cost estimation, and a spend cap."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from open_referee.providers.base import LLMResponse, ModelSpec


class CostEstimate(BaseModel):
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


class CallRecord(BaseModel):
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    role: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class BudgetExceeded(RuntimeError):
    def __init__(self, estimated_cost_usd: float, cap_usd: float):
        self.estimated_cost_usd = estimated_cost_usd
        self.cap_usd = cap_usd
        super().__init__(
            f"Estimated cost ${estimated_cost_usd:.4f} would exceed the configured "
            f"spend cap of ${cap_usd:.2f}. Raise review.max_cost_usd to continue."
        )


class UsageLedger:
    """Thread/async-safe ledger. Checked before every LLM call."""

    def __init__(self, cap_usd: float, pricing: dict[str, dict[str, float]] | None = None):
        self.cap_usd = cap_usd
        self.pricing = pricing or {}
        self.calls: list[CallRecord] = []
        self._lock = asyncio.Lock()

    def price_for(self, spec: ModelSpec) -> CostEstimate:
        entry = self.pricing.get(f"{spec.provider_name}/{spec.model}", {})
        return CostEstimate(
            input_per_mtok=float(entry.get("input_per_mtok", 0.0)),
            output_per_mtok=float(entry.get("output_per_mtok", 0.0)),
        )

    def estimate_cost(
        self, spec: ModelSpec, est_input_tokens: int, est_output_tokens: int
    ) -> float:
        p = self.price_for(spec)
        return (
            est_input_tokens * p.input_per_mtok + est_output_tokens * p.output_per_mtok
        ) / 1_000_000

    async def check_budget(
        self, spec: ModelSpec, est_input_tokens: int, est_output_tokens: int
    ) -> None:
        projected = self.total_cost() + self.estimate_cost(
            spec, est_input_tokens, est_output_tokens
        )
        if projected > self.cap_usd:
            raise BudgetExceeded(projected, self.cap_usd)

    async def record(self, spec: ModelSpec, role: str, resp: LLMResponse) -> CallRecord:
        cost = self.estimate_cost(spec, resp.usage.input_tokens, resp.usage.output_tokens)
        rec = CallRecord(
            role=role,
            provider=spec.provider_name,
            model=spec.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            estimated_cost_usd=round(cost, 6),
        )
        async with self._lock:
            self.calls.append(rec)
        return rec

    def total_tokens(self) -> tuple[int, int]:
        return sum(c.input_tokens for c in self.calls), sum(c.output_tokens for c in self.calls)

    def total_cost(self) -> float:
        return sum(c.estimated_cost_usd for c in self.calls)

    def by_role(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for c in self.calls:
            agg = out.setdefault(
                c.role, {"input_tokens": 0.0, "output_tokens": 0.0, "cost_usd": 0.0, "calls": 0.0}
            )
            agg["input_tokens"] += c.input_tokens
            agg["output_tokens"] += c.output_tokens
            agg["cost_usd"] += c.estimated_cost_usd
            agg["calls"] += 1
        return out
