"""RAG-like retrieval over the local Agent Reliability knowledge base.

A dependency-free TF-IDF + cosine index. The point is not to out-engineer a
vector database; it is to show the retrieval-grounding contract an FDE cares
about: every returned chunk carries a stable source id so answers can be
audited back to evidence.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

from . import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "be", "with", "that", "this", "it", "as", "at", "by", "from", "into", "than",
    "not", "no", "do", "does", "what", "which", "when", "how", "why", "vs",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords, light de-pluralize."""
    tokens = _TOKEN_RE.findall(text.lower())
    out = []
    for tok in tokens:
        if tok in _STOPWORDS or len(tok) == 1:
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


@dataclass
class Document:
    id: str
    kind: str  # "paper" | "note"
    title: str
    topic: str
    text: str
    tags: list[str] = field(default_factory=list)
    payload: dict = field(default_factory=dict)

    def snippet(self, limit: int = 240) -> str:
        clean = " ".join(self.text.split())
        return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"


@dataclass
class Hit:
    document: Document
    score: float


def _load_papers() -> list[Document]:
    data = json.loads(config.PAPERS_PATH.read_text())
    docs: list[Document] = []
    for e in data["entries"]:
        parts = [e["title"], e["summary"]]
        parts += e.get("key_points", [])
        parts += e.get("failure_modes", [])
        parts += e.get("tags", [])
        parts.append(e.get("topic", ""))
        docs.append(
            Document(
                id=e["id"],
                kind="paper",
                title=e["title"],
                topic=e.get("topic", ""),
                text=" ".join(parts),
                tags=e.get("tags", []),
                payload=e,
            )
        )
    return docs


def _load_notes() -> list[Document]:
    raw = config.LAB_NOTES_PATH.read_text()
    docs: list[Document] = []
    current_topic: str | None = None
    buf: list[str] = []
    idx = 1

    def flush():
        nonlocal idx, buf, current_topic
        if current_topic and buf:
            body = "\n".join(buf).strip()
            if body:
                docs.append(
                    Document(
                        id=f"NOTE-{idx:03d}",
                        kind="note",
                        title=f"Field note: {current_topic}",
                        topic=current_topic,
                        text=f"{current_topic} {body}",
                        tags=[current_topic],
                        payload={"topic": current_topic, "body": body},
                    )
                )
                idx += 1
        buf = []

    for line in raw.splitlines():
        m = re.match(r"^##\s+topic:\s*(.+?)\s*$", line)
        if m:
            flush()
            current_topic = m.group(1).strip()
        elif current_topic is not None:
            buf.append(line)
    flush()
    return docs


class RetrievalIndex:
    """TF-IDF cosine index over papers + notes."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._df: dict[str, int] = {}
        self._doc_vectors: list[dict[str, float]] = []
        self._build()

    def _build(self) -> None:
        tokenized = [tokenize(d.text) for d in self.documents]
        for toks in tokenized:
            for term in set(toks):
                self._df[term] = self._df.get(term, 0) + 1
        n = len(self.documents)
        self._idf = {t: math.log((1 + n) / (1 + df)) + 1.0 for t, df in self._df.items()}
        for toks in tokenized:
            self._doc_vectors.append(self._vector(toks))

    def _vector(self, tokens: Iterable[str]) -> dict[str, float]:
        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0
        vec = {t: f * self._idf.get(t, math.log(len(self.documents) + 1) + 1.0)
               for t, f in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(w * b.get(t, 0.0) for t, w in a.items())

    def search(self, query: str, top_k: int = 3) -> list[Hit]:
        qvec = self._vector(tokenize(query))
        scored = [
            Hit(doc, self._cosine(qvec, dvec))
            for doc, dvec in zip(self.documents, self._doc_vectors)
        ]
        scored.sort(key=lambda h: (h.score, h.document.id), reverse=True)
        return [h for h in scored[:top_k] if h.score > 0.0]

    def notes_for_topic(self, topic: str) -> list[Document]:
        q = topic.lower().strip()
        notes = [d for d in self.documents if d.kind == "note"]
        exact = [d for d in notes if d.topic.lower() == q]
        if exact:
            return exact
        ranked = self.search(topic, top_k=len(notes) or 1)
        return [h.document for h in ranked if h.document.kind == "note"][:2]

    def topics(self) -> list[str]:
        return sorted({d.topic for d in self.documents if d.topic})


@lru_cache(maxsize=1)
def get_index() -> RetrievalIndex:
    return RetrievalIndex(_load_papers() + _load_notes())
