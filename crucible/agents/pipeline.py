"""Assembles the researcher → reviser → judge pipeline.

Hierarchical delegation via a SequentialAgent: each stage writes its result to
shared session state under an explicit key, and the next stage reads it. This is
the "make the state contract explicit" lesson from the knowledge base, applied
to the pipeline itself.
"""
from __future__ import annotations

from google.adk.agents import SequentialAgent

from .. import config, offline
from .judge import build_judge
from .researcher import build_researcher
from .reviser import build_reviser


def build_pipeline(use_offline: bool | None = None) -> SequentialAgent:
    """Build the full eval pipeline.

    Args:
        use_offline: force the scripted offline model. Defaults to config.OFFLINE,
            which reads the CRUCIBLE_OFFLINE env var.
    """
    if use_offline is None:
        use_offline = config.OFFLINE

    if use_offline:
        researcher = build_researcher(offline.researcher_model())
        reviser = build_reviser(offline.reviser_model())
        judge = build_judge(offline.judge_model())
    else:
        researcher = build_researcher()
        reviser = build_reviser()
        judge = build_judge()

    # SequentialAgent is the canonical, well-documented composition primitive and
    # is what we want here: a fixed researcher → reviser → judge order with an
    # explicit shared-state contract. ADK 2.x also ships a newer lower-level graph
    # API (google.adk.workflow) for dynamic/branching topologies; this pipeline is
    # strictly linear, so the sequential primitive is the right fit.
    return SequentialAgent(
        name="crucible",
        description=(
            "Crucible: a multi-agent evaluation forge. A ReAct researcher answers "
            "from a knowledge base, a reviser self-reflects on the draft, and an "
            "LLM-as-judge scores the result on a multi-dimensional rubric."
        ),
        sub_agents=[researcher, reviser, judge],
    )
