"""The reviser agent: one self-reflection pass over the researcher's draft."""
from __future__ import annotations

from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext

from .. import config


def reviser_instruction(ctx: ReadonlyContext) -> str:
    draft = ctx.state.get("research_answer", "")
    return f"""\
You are a critical reviewer performing one self-reflection pass on a draft
answer about agentic-system reliability.

Draft answer to review:
\"\"\"
{draft}
\"\"\"

Critique the draft against these checks, then output ONLY the improved answer:
- Remove any claim not supported by a cited source id; keep the citations.
- Cut padding, hedging, and repetition (counter verbosity).
- Preserve correct content, do not rewrite a good answer into a worse one.
- Keep it under 120 words and end with a "Sources:" line listing the ids.

Output only the revised answer text. Do not include your critique."""


def build_reviser(model: Optional[object] = None) -> LlmAgent:
    return LlmAgent(
        name="reviser",
        model=model or config.REVISER_MODEL,
        description="Performs a self-reflection pass that tightens and grounds the draft answer.",
        instruction=reviser_instruction,
        output_key="revised_answer",
    )
