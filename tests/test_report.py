"""Report aggregation, including graceful handling of errored (e.g. 429) runs."""
from crucible.eval.report import EvalReport, RunResult


def _ok(case_id, overall, verdict, retrieved, expected, in_scope=True):
    return RunResult(
        case_id=case_id, question="q", topic="t", in_scope=in_scope,
        answer="a", verdict={**verdict, "overall": overall},
        retrieved_ids=retrieved, expected_ids=expected,
        trace_text="", trace={},
        metrics={"summary": {"total_cost_usd": 0.0, "total_tokens": 10,
                             "total_latency_s": 1.0, "calls": 5},
                 "per_agent": {"researcher": {"output_tokens": 5}}},
        wall_time_s=1.0,
    )


def _errored(case_id):
    return RunResult(
        case_id=case_id, question="q", topic="t", in_scope=True,
        answer="[run did not complete: rate limit hit (429, daily free-tier quota)]",
        verdict={}, retrieved_ids=[], expected_ids=["KB-001"],
        trace_text="", trace={},
        metrics={"summary": {"total_cost_usd": 0.0, "total_tokens": 0,
                             "total_latency_s": 0.0, "calls": 0},
                 "per_agent": {}},
        wall_time_s=0.0,
    )


def test_errored_run_excluded_from_aggregates_and_marked():
    report = EvalReport(
        mode="online", model_map={"researcher": "gemini-2.5-flash"},
        results=[
            _ok("Q1", 5.0, {"groundedness": 5, "tool_selection": 5, "conciseness": 5}, ["KB-001"], ["KB-001"]),
            _errored("Q2"),
        ],
    )
    # Errored run must not drag the mean to ~2.5; only the good run counts.
    assert report.mean_overall() == 5.0
    assert report.retrieval_recall() == 1.0  # Q2's expected id is excluded, not a miss

    md = report.to_markdown()
    assert "rate limit hit" in md
    assert "Crucible eval report" in md
