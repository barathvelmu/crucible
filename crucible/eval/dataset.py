"""The evaluation dataset.

A small, held-out question set spanning the knowledge base, including one
deliberately out-of-scope question to test whether the agent abstains instead
of hallucinating. `expects_ids` is a soft signal used to sanity-check whether
retrieval surfaced the entries we'd expect, it is not a hard assertion.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    topic: str
    expects_ids: list[str] = field(default_factory=list)
    in_scope: bool = True


DATASET: list[EvalCase] = [
    EvalCase(
        id="Q1",
        question="What is the dominant failure mode in tool-use agents?",
        topic="tool-use failures",
        expects_ids=["KB-003"],
    ),
    EvalCase(
        id="Q2",
        question="When does a self-reflection pass actually improve an answer, and when does it hurt?",
        topic="self-reflection",
        expects_ids=["KB-002"],
    ),
    EvalCase(
        id="Q3",
        question="Why is a single quality score a bad idea for LLM-as-judge evaluation?",
        topic="llm-as-judge",
        expects_ids=["KB-006"],
    ),
    EvalCase(
        id="Q4",
        question="What matters more for RAG quality: the vector store or the chunking strategy?",
        topic="rag chunking",
        expects_ids=["KB-007"],
    ),
    EvalCase(
        id="Q5",
        question="Why isn't final-answer-only evaluation enough for long-horizon agents?",
        topic="long-horizon agents",
        expects_ids=["KB-008"],
    ),
    EvalCase(
        id="Q6",
        question="What is the airspeed velocity of an unladen swallow?",
        topic="out-of-scope",
        expects_ids=[],
        in_scope=False,
    ),
]
