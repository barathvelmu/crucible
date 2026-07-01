"""End-to-end pipeline tests using the scripted offline model, no API key."""
import asyncio

from crucible.agents.pipeline import build_pipeline
from crucible.eval.dataset import DATASET
from crucible.eval.runner import run_case, run_eval


def test_pipeline_structure():
    p = build_pipeline(use_offline=True)
    assert [s.name for s in p.sub_agents] == ["researcher", "reviser", "judge"]


def test_offline_run_grounds_and_scores_well():
    result, _ = asyncio.run(run_case(
        "What is the dominant failure mode in tool-use agents?",
        use_offline=True, label="T1",
    ))
    assert "KB-003" in result.retrieved_ids
    assert result.verdict["groundedness"] == 5
    assert result.verdict["tool_selection"] == 5
    assert result.overall == 5.0
    assert "search_papers" in result.trace_text
    assert "Sources:" in result.answer


def test_offline_abstains_on_out_of_scope():
    result, _ = asyncio.run(run_case(
        "What is the airspeed velocity of an unladen swallow?",
        use_offline=True, label="OOS",
    ))
    assert result.retrieved_ids == []
    # No grounded citations -> judge should not award full groundedness.
    assert result.verdict["groundedness"] < 5


def test_full_offline_eval_report():
    report = asyncio.run(run_eval(DATASET, use_offline=True))
    assert len(report.results) == len(DATASET)
    assert report.mean_overall() >= 4.5
    assert report.retrieval_recall() == 1.0
    totals = report.totals()
    assert totals["model_calls"] > 0
    assert totals["total_tokens"] > 0
    # Markdown and JSON render without error.
    assert "Crucible eval report" in report.to_markdown()
    assert report.to_json().startswith("{")
