"""Headless evaluation runner.

Runs the pipeline over a question set, captures per-call metrics via ADK model
callbacks, builds a structural trace from the event stream, and emits a scored
report. Defaults to the offline scripted model so it runs with no API key and
burns no quota; pass --online to use real Gemini.

Usage:
    python -m crucible.eval.runner                 # offline, full dataset
    python -m crucible.eval.runner --online        # real Gemini
    python -m crucible.eval.runner -q "your question here"
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .. import config
from ..agents.pipeline import build_pipeline
from ..metrics import MetricsLedger, UsageRecord
from ..tools import RETRIEVED_IDS_KEY
from ..tracing import Trace
from .dataset import DATASET, EvalCase
from .report import EvalReport, RunResult

APP_NAME = "crucible"


class CallMetrics:
    """Collects one UsageRecord per model call via before/after callbacks."""

    def __init__(self, model_map: dict[str, str]):
        self.ledger = MetricsLedger()
        self.model_map = model_map
        self._stack: list[tuple[str, float]] = []

    def before(self, callback_context, llm_request):  # noqa: ANN001 (ADK callback signature)
        self._stack.append((callback_context.agent_name, time.perf_counter()))
        return None

    def after(self, callback_context, llm_response):  # noqa: ANN001
        agent, t0 = self._stack.pop() if self._stack else (callback_context.agent_name, time.perf_counter())
        latency = time.perf_counter() - t0
        um = getattr(llm_response, "usage_metadata", None)
        prompt = (um.prompt_token_count or 0) if um else 0
        output = (um.candidates_token_count or 0) if um else 0
        self.ledger.add(UsageRecord(
            agent=agent,
            model=self.model_map.get(agent, "unknown"),
            prompt_tokens=prompt,
            output_tokens=output,
            latency_s=latency,
        ))
        return None


def _short_error(e: Exception) -> str:
    s = str(e)
    if "RESOURCE_EXHAUSTED" in s or "429" in s:
        scope = "daily" if "PerDay" in s else "per-minute" if "PerMinute" in s else ""
        return f"rate limit hit (429{', ' + scope + ' free-tier quota' if scope else ''})"
    return f"{type(e).__name__}: {s[:120]}"


def _model_name(agent: LlmAgent) -> str:
    m = agent.model
    return m if isinstance(m, str) else getattr(m, "model", "scripted-offline")


def _attach_metrics(pipeline, collector: CallMetrics) -> None:
    for sub in pipeline.sub_agents:
        sub.before_model_callback = collector.before
        sub.after_model_callback = collector.after


def _build_trace(events, model_map: dict[str, str], ledger: MetricsLedger, label: str) -> Trace:
    """Build a structural trace from the event stream.

    Tool spans come from function-call/response events. Each agent's span is
    annotated with that agent's *aggregate* tokens and latency from the metrics
    ledger (summed over all of its model calls), so the numbers are correct even
    when a single model response batches multiple tool calls (parallel calling).
    """
    trace = Trace(label=label)
    per_agent = ledger.per_agent()
    annotated: set[str] = set()
    last_tool_span = None
    for ev in events:
        if not (ev.content and ev.content.parts):
            continue
        for part in ev.content.parts:
            if part.function_call is not None:
                span = trace.event(part.function_call.name, "tool")
                span.attributes["args"] = dict(part.function_call.args or {})
                last_tool_span = span
            elif part.function_response is not None:
                resp = part.function_response.response or {}
                ids = [it["id"] for it in (resp.get("results", []) + resp.get("notes", []))
                       if isinstance(it, dict) and it.get("id")]
                if last_tool_span is not None:
                    last_tool_span.attributes["ids"] = ids
            elif part.text and ev.author not in annotated:
                agg = per_agent.get(ev.author)
                model = model_map.get(ev.author, ev.author)
                span = trace.event(ev.author, "agent", model=model)
                if agg:
                    span.end = span.start + agg["latency_s"]
                    span.attributes["tokens"] = agg["prompt_tokens"] + agg["output_tokens"]
                    span.attributes["calls"] = agg["calls"]
                annotated.add(ev.author)
    return trace


async def run_case(case_question: str, use_offline: bool, label: str) -> tuple[RunResult, dict]:
    pipeline = build_pipeline(use_offline=use_offline)
    model_map = {sub.name: _model_name(sub) for sub in pipeline.sub_agents}
    collector = CallMetrics(model_map)
    _attach_metrics(pipeline, collector)

    ss = InMemorySessionService()
    sid = f"s-{label}"
    await ss.create_session(app_name=APP_NAME, user_id="evaluator", session_id=sid)
    runner = Runner(agent=pipeline, app_name=APP_NAME, session_service=ss)
    msg = types.Content(role="user", parts=[types.Part(text=case_question)])

    t0 = time.perf_counter()
    events: list = []
    run_error: str | None = None
    try:
        async for ev in runner.run_async(user_id="evaluator", session_id=sid, new_message=msg):
            events.append(ev)
    except Exception as e:  # most often a 429 RESOURCE_EXHAUSTED on the free tier
        run_error = _short_error(e)
    wall = round(time.perf_counter() - t0, 3)

    sess = await ss.get_session(app_name=APP_NAME, user_id="evaluator", session_id=sid)
    state = sess.state
    verdict = dict(state.get("verdict") or {})
    if verdict and "overall" not in verdict:
        scores = [verdict.get(k, 0) for k in ("groundedness", "tool_selection", "conciseness")]
        verdict["overall"] = round(sum(scores) / 3, 2)
    answer = state.get("revised_answer") or state.get("research_answer") or ""
    if run_error and not answer:
        answer = f"[run did not complete: {run_error}]"
    retrieved = list(state.get(RETRIEVED_IDS_KEY, []))

    trace = _build_trace(events, model_map, collector.ledger, label)

    metrics = {"summary": collector.ledger.summary(), "per_agent": collector.ledger.per_agent()}
    result = RunResult(
        case_id=label,
        question=case_question,
        topic="",
        in_scope=True,
        answer=answer,
        verdict=verdict,
        retrieved_ids=retrieved,
        expected_ids=[],
        trace_text=trace.render(),
        trace=trace.to_dict(),
        metrics=metrics,
        wall_time_s=wall,
    )
    return result, model_map


async def run_eval(cases: list[EvalCase], use_offline: bool, delay_s: float = 0.0) -> EvalReport:
    results: list[RunResult] = []
    model_map: dict = {}
    for i, case in enumerate(cases):
        res, model_map = await run_case(case.question, use_offline, label=case.id)
        res.topic = case.topic
        res.in_scope = case.in_scope
        res.expected_ids = list(case.expects_ids)
        results.append(res)
        if delay_s and i < len(cases) - 1:
            await asyncio.sleep(delay_s)
    return EvalReport(mode="offline" if use_offline else "online", model_map=model_map, results=results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crucible eval runner")
    parser.add_argument("--online", action="store_true", help="Use real Gemini (needs GOOGLE_API_KEY).")
    parser.add_argument("-q", "--question", help="Run a single ad-hoc question instead of the dataset.")
    parser.add_argument("--out", default="reports", help="Directory for report.md / report.json.")
    parser.add_argument("--no-write", action="store_true", help="Print only; do not write files.")
    parser.add_argument("--delay", type=float, default=None,
                        help="Seconds between questions (default: 0 offline, 4 online).")
    args = parser.parse_args()

    use_offline = not args.online
    delay = args.delay if args.delay is not None else (0.0 if use_offline else 4.0)

    if args.online:
        # Mirror `adk web`: load crucible/.env so GOOGLE_API_KEY is available.
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        except ImportError:
            pass

    if args.question:
        cases = [EvalCase(id="Q1", question=args.question, topic="ad-hoc")]
    else:
        cases = DATASET

    report = asyncio.run(run_eval(cases, use_offline=use_offline, delay_s=delay))
    md = report.to_markdown()
    print(md)

    if not args.no_write:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(md)
        (out / "report.json").write_text(report.to_json())
        print(f"\n[written] {out/'report.md'}  and  {out/'report.json'}")


if __name__ == "__main__":
    main()
