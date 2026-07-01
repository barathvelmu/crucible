from crucible.retrieval import get_index, tokenize


def test_corpus_loads_papers_and_notes():
    idx = get_index()
    papers = [d for d in idx.documents if d.kind == "paper"]
    notes = [d for d in idx.documents if d.kind == "note"]
    assert len(papers) == 8
    assert len(notes) >= 5
    assert all(d.id for d in idx.documents)


def test_tokenize_drops_stopwords_and_depluralizes():
    toks = tokenize("What are the dominant failures in agents?")
    assert "the" not in toks and "are" not in toks
    assert "failure" in toks  # de-pluralized
    assert "agent" in toks


def test_search_ranks_relevant_entry_first():
    idx = get_index()
    hits = idx.search("dominant failure mode in tool-use agents", top_k=3)
    assert hits, "expected at least one hit"
    assert hits[0].document.id == "KB-003"
    assert hits[0].score > 0


def test_search_is_deterministic():
    idx = get_index()
    a = [h.document.id for h in idx.search("rag chunking strategy", top_k=3)]
    b = [h.document.id for h in idx.search("rag chunking strategy", top_k=3)]
    assert a == b


def test_out_of_scope_query_returns_no_strong_hits():
    idx = get_index()
    hits = idx.search("airspeed velocity of an unladen swallow", top_k=3)
    assert hits == [] or all(h.score < 0.15 for h in hits)


def test_notes_for_topic_matches_exact_topic():
    idx = get_index()
    notes = idx.notes_for_topic("tool-use failures")
    assert notes and all(n.kind == "note" for n in notes)
    assert notes[0].topic == "tool-use failures"
