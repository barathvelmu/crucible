"""Tools the researcher agent can call.

These are plain Python functions with explicit docstrings, ADK turns the
docstring and type hints into the tool schema the model sees, so the wording
here is part of the prompt surface. Each tool also records which source ids it
returned into session state, so the judge can later assess groundedness and
the trace can show what evidence the answer was built on.
"""
from __future__ import annotations

from typing import Any, Optional

from .retrieval import get_index

# Session-state key holding the list of source ids retrieved this run.
RETRIEVED_IDS_KEY = "crucible:retrieved_ids"


def _record(tool_context: Any, ids: list[str]) -> None:
    if tool_context is None:
        return
    try:
        existing = list(tool_context.state.get(RETRIEVED_IDS_KEY, []))
    except Exception:
        return
    for i in ids:
        if i not in existing:
            existing.append(i)
    tool_context.state[RETRIEVED_IDS_KEY] = existing


def search_papers(query: str, top_k: int = 3, tool_context: Optional[Any] = None) -> dict:
    """Search the Agent Reliability knowledge base for entries matching a query.

    Use this to find techniques, definitions, and failure modes for agentic
    systems (e.g. ReAct, tool-use failures, LLM-as-judge, RAG chunking).

    Args:
        query: A natural-language search query describing what you need.
        top_k: How many of the most relevant entries to return (default 3).

    Returns:
        A dict with a "results" list; each result has id, title, topic, tags,
        a text snippet, and a relevance score. Always cite the returned ids.
    """
    hits = get_index().search(query, top_k=top_k)
    results = [
        {
            "id": h.document.id,
            "title": h.document.title,
            "topic": h.document.topic,
            "tags": h.document.tags,
            "snippet": h.document.snippet(),
            "score": round(h.score, 4),
        }
        for h in hits
    ]
    _record(tool_context, [r["id"] for r in results])
    return {"query": query, "result_count": len(results), "results": results}


def get_lab_notes(topic: str, tool_context: Optional[Any] = None) -> dict:
    """Fetch operational field notes for a specific topic.

    Field notes are the "what actually bit us in production" observations, more
    candid than the knowledge-base entries. Use this once you know the topic.

    Args:
        topic: The topic to fetch notes for (e.g. "tool-use failures",
            "self-reflection", "llm-as-judge", "rag chunking").

    Returns:
        A dict with a "notes" list; each note has id, topic, and body text.
    """
    notes = get_index().notes_for_topic(topic)
    results = [
        {"id": n.id, "topic": n.topic, "body": n.payload.get("body", n.text)}
        for n in notes
    ]
    _record(tool_context, [r["id"] for r in results])
    return {"topic": topic, "note_count": len(results), "notes": results}


def list_corpus_topics(tool_context: Optional[Any] = None) -> dict:
    """List every topic available in the knowledge base.

    Call this first when you are unsure what the knowledge base covers, so you
    can choose precise search terms instead of guessing.

    Returns:
        A dict with a "topics" list of topic strings.
    """
    return {"topics": get_index().topics()}


ALL_TOOLS = [search_papers, get_lab_notes, list_corpus_topics]
