"""Evaluation rubric and the judge's structured verdict schema.

The rubric is deliberately multi-dimensional: a single quality score collapses
to "4/5 everything" and tells you nothing. Scoring groundedness, tool-selection,
and conciseness separately makes the judge actionable and surfaces verbosity
bias instead of hiding it.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RubricDimension(BaseModel):
    key: str
    name: str
    description: str


RUBRIC: list[RubricDimension] = [
    RubricDimension(
        key="groundedness",
        name="Groundedness",
        description=(
            "Is every claim supported by the retrieved evidence, and are the "
            "source ids cited? Unsupported or fabricated claims score low."
        ),
    ),
    RubricDimension(
        key="tool_selection",
        name="Tool selection",
        description=(
            "Did the agent call the right tools in a sensible order, and use "
            "the observations it got back? Guessing without retrieval scores low."
        ),
    ),
    RubricDimension(
        key="conciseness",
        name="Conciseness",
        description=(
            "Is the answer as short as it can be while staying complete? "
            "Padding and repetition score low; this dimension counters "
            "verbosity bias."
        ),
    ),
]

RUBRIC_KEYS = [d.key for d in RUBRIC]


class JudgeVerdict(BaseModel):
    """Structured score the judge agent must return (output_schema)."""

    groundedness: int = Field(ge=1, le=5, description="1-5 score for groundedness.")
    tool_selection: int = Field(ge=1, le=5, description="1-5 score for tool selection.")
    conciseness: int = Field(ge=1, le=5, description="1-5 score for conciseness.")
    rationale: str = Field(description="One or two sentences justifying the scores.")
    cited_ids: list[str] = Field(
        default_factory=list,
        description="Source ids the answer actually cited, e.g. ['KB-003'].",
    )

    @property
    def overall(self) -> float:
        return round((self.groundedness + self.tool_selection + self.conciseness) / 3, 2)

    def as_row(self) -> dict:
        return {
            "groundedness": self.groundedness,
            "tool_selection": self.tool_selection,
            "conciseness": self.conciseness,
            "overall": self.overall,
            "cited_ids": list(self.cited_ids),
            "rationale": self.rationale,
        }
