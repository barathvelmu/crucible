"""Eval result types and report rendering (markdown + JSON)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from ..rubric import RUBRIC_KEYS


@dataclass
class RunResult:
    case_id: str
    question: str
    topic: str
    in_scope: bool
    answer: str
    verdict: dict
    retrieved_ids: list[str]
    expected_ids: list[str]
    trace_text: str
    trace: dict
    metrics: dict          # {"summary": {...}, "per_agent": {...}}
    wall_time_s: float

    @property
    def overall(self) -> float:
        return float(self.verdict.get("overall") or 0.0)

    @property
    def errored(self) -> bool:
        return not self.verdict

    @property
    def retrieval_hit(self) -> bool:
        """Did retrieval surface at least one expected id (soft check)?"""
        if not self.expected_ids:
            return True
        return any(i in self.retrieved_ids for i in self.expected_ids)


@dataclass
class EvalReport:
    mode: str                       # "offline" | "online"
    model_map: dict
    results: list[RunResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # --- aggregates ---
    def _scored(self) -> list[RunResult]:
        return [r for r in self.results if r.in_scope and not r.errored]

    def mean_overall(self) -> float:
        scored = [r.overall for r in self._scored()]
        return round(sum(scored) / len(scored), 2) if scored else 0.0

    def mean_dim(self, key: str) -> float:
        vals = [float(r.verdict.get(key, 0)) for r in self._scored()]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    def retrieval_recall(self) -> float:
        scoped = [r for r in self.results if r.expected_ids and not r.errored]
        if not scoped:
            return 1.0
        return round(sum(1 for r in scoped if r.retrieval_hit) / len(scoped), 2)

    def totals(self) -> dict:
        cost = round(sum(r.metrics["summary"]["total_cost_usd"] for r in self.results), 6)
        tokens = sum(r.metrics["summary"]["total_tokens"] for r in self.results)
        out_tokens = sum(r.metrics["per_agent"][a]["output_tokens"]
                         for r in self.results for a in r.metrics["per_agent"])
        calls = sum(r.metrics["summary"]["calls"] for r in self.results)
        gen_latency = round(sum(r.metrics["summary"]["total_latency_s"] for r in self.results), 4)
        latency = round(sum(r.wall_time_s for r in self.results), 2)
        return {
            "questions": len(self.results),
            "model_calls": calls,
            "total_tokens": tokens,
            "total_cost_usd": cost,
            "wall_time_s": latency,
            "cost_per_question_usd": round(cost / len(self.results), 6) if self.results else 0.0,
            # Request-level output throughput: output tokens over summed model-call
            # latency (which includes the prompt round trip), not raw decode tok/s.
            "tokens_per_sec": round(out_tokens / gen_latency, 1) if gen_latency > 0 else 0.0,
        }

    # --- rendering ---
    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "created_at": self.created_at,
            "model_map": self.model_map,
            "aggregates": {
                "mean_overall": self.mean_overall(),
                **{f"mean_{k}": self.mean_dim(k) for k in RUBRIC_KEYS},
                "retrieval_recall": self.retrieval_recall(),
                **self.totals(),
            },
            "results": [asdict(r) for r in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        t = self.totals()
        lines: list[str] = []
        lines.append(f"# Crucible eval report ({self.mode})")
        lines.append("")
        lines.append(f"_Generated {self.created_at} · models: "
                     + ", ".join(f"{k}={v}" for k, v in self.model_map.items()) + "_")
        lines.append("")
        lines.append("## Scores")
        lines.append("")
        lines.append("| Q | Topic | Ground | Tool | Concise | Overall | Retrieved | Hit |")
        lines.append("|---|-------|:------:|:----:|:-------:|:-------:|-----------|:---:|")
        for r in self.results:
            if r.errored:
                lines.append(f"| {r.case_id} | {r.topic} |, |, |, |, | "
                             f"_{r.answer}_ |, |")
                continue
            v = r.verdict
            hit = ", " if not r.expected_ids else ("✓" if r.retrieval_hit else "✗")
            lines.append(
                f"| {r.case_id} | {r.topic} | {v.get('groundedness','-')} | "
                f"{v.get('tool_selection','-')} | {v.get('conciseness','-')} | "
                f"{r.overall} | {', '.join(r.retrieved_ids) or ', '} | {hit} |"
            )
        lines.append("")
        lines.append("## Aggregates")
        lines.append("")
        lines.append(f"- Mean overall (in-scope): **{self.mean_overall()} / 5**")
        lines.append(f"- Mean groundedness: {self.mean_dim('groundedness')} · "
                     f"tool-selection: {self.mean_dim('tool_selection')} · "
                     f"conciseness: {self.mean_dim('conciseness')}")
        lines.append(f"- Retrieval recall (expected id surfaced): **{self.retrieval_recall()}**")
        lines.append("")
        lines.append("## LLM-native metrics")
        lines.append("")
        lines.append(f"- Questions: {t['questions']} · model calls: {t['model_calls']}")
        lines.append(f"- Total tokens: {t['total_tokens']:,} · total cost: "
                     f"${t['total_cost_usd']} · cost/question: ${t['cost_per_question_usd']}")
        lines.append(f"- Throughput: {t['tokens_per_sec']} output tok/s (request-level) · "
                     f"wall time: {t['wall_time_s']}s")
        lines.append("")
        if self.results:
            ex = self.results[0]
            lines.append("## Example trace")
            lines.append("")
            lines.append("```")
            lines.append(ex.trace_text)
            lines.append("```")
            lines.append("")
            lines.append(f"**Q ({ex.case_id}):** {ex.question}")
            lines.append("")
            lines.append(f"**Answer:** {ex.answer}")
            lines.append("")
            lines.append(f"**Verdict:** {json.dumps(ex.verdict)}")
        return "\n".join(lines)
