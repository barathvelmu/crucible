from crucible.tools import (
    RETRIEVED_IDS_KEY,
    get_lab_notes,
    list_corpus_topics,
    search_papers,
)


class FakeToolContext:
    """Stand-in for ADK's ToolContext so tools can be unit-tested offline."""

    def __init__(self):
        self.state = {}


def test_search_papers_returns_grounded_results():
    out = search_papers("tool-use failure modes", top_k=2)
    assert out["result_count"] >= 1
    assert all("id" in r and "snippet" in r and "score" in r for r in out["results"])
    assert out["results"][0]["id"] == "KB-003"


def test_search_papers_records_ids_in_state():
    ctx = FakeToolContext()
    search_papers("llm as judge bias", top_k=2, tool_context=ctx)
    assert ctx.state[RETRIEVED_IDS_KEY]
    assert all(i.startswith(("KB-", "NOTE-")) for i in ctx.state[RETRIEVED_IDS_KEY])


def test_get_lab_notes_returns_body():
    out = get_lab_notes("self-reflection")
    assert out["note_count"] >= 1
    assert out["notes"][0]["body"]


def test_list_corpus_topics_nonempty():
    out = list_corpus_topics()
    assert "tool-use failures" in out["topics"]
    assert len(out["topics"]) >= 6


def test_tools_work_without_tool_context():
    # tool_context defaults to None; must not raise.
    assert search_papers("rag chunking")["result_count"] >= 0
