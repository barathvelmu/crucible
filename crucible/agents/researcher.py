"""The researcher agent: a ReAct loop over the knowledge-base tools."""
from __future__ import annotations

from typing import Optional

from google.adk.agents import LlmAgent

from .. import config
from ..tools import ALL_TOOLS

RESEARCHER_INSTRUCTION = """\
You are a research agent answering questions about agentic-system reliability,
grounded strictly in the Agent Reliability knowledge base.

Work in a reason-then-act loop:
1. If you are unsure what the knowledge base covers, call `list_corpus_topics`.
2. Call `search_papers` to find the most relevant entries for the question.
3. When you have a specific topic, call `get_lab_notes` for operational detail.
4. Only then write your answer.

Rules:
- Ground every claim in retrieved evidence. Do not use outside knowledge.
- Cite the source ids you used inline, e.g. (KB-003), (NOTE-001).
- Be concise: aim for under 120 words. No preamble, no restating the question.
- If the knowledge base does not cover the question, say so plainly.
"""


def build_researcher(model: Optional[object] = None) -> LlmAgent:
    return LlmAgent(
        name="researcher",
        model=model or config.RESEARCHER_MODEL,
        description="Answers reliability questions grounded in the knowledge base, using retrieval tools.",
        instruction=RESEARCHER_INSTRUCTION,
        tools=list(ALL_TOOLS),
        output_key="research_answer",
    )
