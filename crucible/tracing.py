"""Lightweight cross-agent tracing.

ADK emits OpenTelemetry spans of its own; this is a small, inspectable trace
the eval runner builds from the event stream so a report can show, per run,
which agent did what, which tools fired, and what evidence was retrieved, 
the "granular tracing" an FDE leans on to localize where a trajectory broke.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    name: str
    kind: str  # "agent" | "tool" | "model"
    start: float
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end is None:
            return 0.0
        return round((self.end - self.start) * 1000, 1)


class Trace:
    """An ordered list of spans for a single run."""

    def __init__(self, label: str = "run"):
        self.label = label
        self.spans: list[Span] = []
        self._t0 = time.perf_counter()

    def event(self, name: str, kind: str, duration_s: float = 0.0, **attrs: Any) -> Span:
        now = time.perf_counter()
        span = Span(name=name, kind=kind, start=now - duration_s, end=now, attributes=attrs)
        self.spans.append(span)
        return span

    def by_kind(self, kind: str) -> list[Span]:
        return [s for s in self.spans if s.kind == kind]

    def tool_calls(self) -> list[str]:
        return [s.name for s in self.by_kind("tool")]

    def render(self) -> str:
        """Human-readable trace tree for the report / CLI output."""
        lines = [f"trace[{self.label}]"]
        for i, s in enumerate(self.spans):
            connector = "└─" if i == len(self.spans) - 1 else "├─"
            detail = ""
            if s.kind == "tool" and "ids" in s.attributes:
                detail = f"  → {', '.join(s.attributes['ids']) or 'no hits'}"
            elif s.kind == "agent" and "model" in s.attributes:
                detail = f"  ({s.attributes['model']})"
            timing = f"  {s.duration_ms}ms" if s.duration_ms else ""
            tokens = ""
            if s.attributes.get("tokens"):
                calls = s.attributes.get("calls")
                call_str = f"/{calls} calls" if calls and calls > 1 else ""
                tokens = f"  {s.attributes['tokens']} tok{call_str}"
            lines.append(f"  {connector} [{s.kind}] {s.name}{timing}{tokens}{detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "spans": [
                {"name": s.name, "kind": s.kind, "duration_ms": s.duration_ms,
                 "attributes": s.attributes}
                for s in self.spans
            ],
        }
