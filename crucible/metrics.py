"""LLM-native metrics: tokens, latency, throughput, and cost-per-request.

These are the operational numbers the FDE role calls out explicitly
(tokens/sec, cost-per-request). The ledger collects one record per model call
and aggregates per-agent and per-run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config


def cost_usd(model: str, prompt_tokens: int, output_tokens: int) -> float:
    """Cost of a single call given the illustrative price table in config."""
    price = config.PRICING_USD_PER_1M.get(model)
    if not price:
        return 0.0
    return round(
        prompt_tokens / 1_000_000 * price["input"]
        + output_tokens / 1_000_000 * price["output"],
        6,
    )


@dataclass
class UsageRecord:
    agent: str
    model: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def cost(self) -> float:
        return cost_usd(self.model, self.prompt_tokens, self.output_tokens)

    @property
    def tokens_per_sec(self) -> float:
        return round(self.output_tokens / self.latency_s, 1) if self.latency_s > 0 else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 4)


@dataclass
class MetricsLedger:
    records: list[UsageRecord] = field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    @property
    def total_cost(self) -> float:
        return round(sum(r.cost for r in self.records), 6)

    @property
    def total_latency(self) -> float:
        return round(sum(r.latency_s for r in self.records), 4)

    def per_agent(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in self.records:
            a = out.setdefault(
                r.agent,
                {"calls": 0, "prompt_tokens": 0, "output_tokens": 0,
                 "cost": 0.0, "latency_s": 0.0, "model": r.model},
            )
            a["calls"] += 1
            a["prompt_tokens"] += r.prompt_tokens
            a["output_tokens"] += r.output_tokens
            a["cost"] = round(a["cost"] + r.cost, 6)
            a["latency_s"] = round(a["latency_s"] + r.latency_s, 4)
        return out

    def summary(self) -> dict:
        latencies = [r.latency_s for r in self.records]
        out_tokens = sum(r.output_tokens for r in self.records)
        return {
            "calls": len(self.records),
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "total_latency_s": self.total_latency,
            "p50_latency_s": _percentile(latencies, 0.50),
            "p95_latency_s": _percentile(latencies, 0.95),
            # Request-level output throughput (output tokens / summed call latency),
            # not raw decode tok/s, call latency includes the prompt round trip.
            "tokens_per_sec": round(out_tokens / self.total_latency, 1)
            if self.total_latency > 0 else 0.0,
            "cost_per_request_usd": round(self.total_cost / len(self.records), 6)
            if self.records else 0.0,
        }
