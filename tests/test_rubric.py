import pytest
from pydantic import ValidationError

from crucible.rubric import RUBRIC, RUBRIC_KEYS, JudgeVerdict


def test_rubric_has_three_named_dimensions():
    assert RUBRIC_KEYS == ["groundedness", "tool_selection", "conciseness"]
    assert all(d.name and d.description for d in RUBRIC)


def test_verdict_overall_is_mean():
    v = JudgeVerdict(groundedness=5, tool_selection=4, conciseness=3, rationale="ok", cited_ids=["KB-001"])
    assert v.overall == round((5 + 4 + 3) / 3, 2)
    assert v.as_row()["overall"] == v.overall


def test_verdict_rejects_out_of_range_scores():
    with pytest.raises(ValidationError):
        JudgeVerdict(groundedness=9, tool_selection=4, conciseness=3, rationale="bad")


def test_verdict_roundtrips_json():
    v = JudgeVerdict(groundedness=5, tool_selection=5, conciseness=5, rationale="great", cited_ids=["KB-003"])
    again = JudgeVerdict.model_validate_json(v.model_dump_json())
    assert again.cited_ids == ["KB-003"]
