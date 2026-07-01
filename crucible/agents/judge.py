"""The judge agent: LLM-as-judge with a structured, multi-dimensional verdict."""
from __future__ import annotations

from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext

from .. import config
from ..rubric import RUBRIC, JudgeVerdict


def _rubric_block() -> str:
    return "\n".join(f"- {d.name} ({d.key}): {d.description}" for d in RUBRIC)


def judge_instruction(ctx: ReadonlyContext) -> str:
    answer = ctx.state.get("revised_answer") or ctx.state.get("research_answer", "")
    retrieved = ctx.state.get("crucible:retrieved_ids", [])
    return f"""\
You are an evaluation judge. Score the answer below on a 1-5 scale for each
rubric dimension. Be calibrated and discriminating, do not give everything a 4.

Rubric:
{_rubric_block()}

Source ids that were actually retrieved this run: {retrieved or "none"}

Answer under evaluation:
\"\"\"
{answer}
\"\"\"

Return your verdict in the required structured format: integer scores 1-5 for
groundedness, tool_selection, and conciseness; a one-or-two sentence rationale;
and cited_ids listing the source ids the answer cites. Penalize groundedness if
the answer cites ids that were not retrieved, or makes uncited claims."""


def build_judge(model: Optional[object] = None) -> LlmAgent:
    return LlmAgent(
        name="judge",
        model=model or config.JUDGE_MODEL,
        description="Scores the final answer on a multi-dimensional rubric and returns a structured verdict.",
        instruction=judge_instruction,
        output_schema=JudgeVerdict,
        output_key="verdict",
    )
