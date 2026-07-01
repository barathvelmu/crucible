"""Deterministic offline model so the whole pipeline runs with no API key.

`ScriptedLLM` is a `BaseLlm` whose responses are produced by a Python callable
instead of a network call. The responders below read the conversation the same
way a real model would, the researcher decides which tool to call next from
what it has already seen; the judge scores the answer it can read in context, 
so the offline run exercises the real ADK orchestration (tool calls, state
hand-offs, structured output), just with the model swapped out.

Timings in offline mode are simulated (a small fixed sleep per call) purely so
the trace and throughput fields are populated; cost is $0 by construction.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

from .rubric import JudgeVerdict

ID_RE = re.compile(r"\b(?:KB|NOTE)-\d{3}\b")
# ADK prefixes another agent's text with "[name] said:" when it lands in the
# next agent's context. Strip it so offline responders see the bare answer.
ATTRIBUTION_RE = re.compile(r"\[[^\]]+\]\s*said:\s*")
DEFAULT_OFFLINE_LATENCY_S = 0.12


@dataclass
class RequestView:
    """A model-agnostic view of an LlmRequest for the responders."""

    system_text: str
    text_parts: list[str]
    all_text: str
    function_responses: list[tuple[str, Any]] = field(default_factory=list)

    @property
    def responded_tools(self) -> set[str]:
        return {name for name, _ in self.function_responses}

    @property
    def first_user_question(self) -> str:
        for t in self.text_parts:
            if t.strip():
                return t.strip()
        return ""

    @property
    def last_model_answer(self) -> str:
        for t in reversed(self.text_parts):
            if t.strip():
                return t.strip()
        return ""


def _system_text(request: LlmRequest) -> str:
    cfg = request.config
    si = getattr(cfg, "system_instruction", None) if cfg else None
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    parts = getattr(si, "parts", None)
    if parts:
        return " ".join(p.text for p in parts if getattr(p, "text", None))
    return str(si)


def view_request(request: LlmRequest) -> RequestView:
    text_parts: list[str] = []
    fn_responses: list[tuple[str, Any]] = []
    for content in request.contents or []:
        for part in content.parts or []:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            fr = getattr(part, "function_response", None)
            if fr is not None:
                fn_responses.append((fr.name, fr.response))
    sys_text = _system_text(request)
    all_text = sys_text + "\n" + "\n".join(text_parts)
    return RequestView(
        system_text=sys_text,
        text_parts=text_parts,
        all_text=all_text,
        function_responses=fn_responses,
    )


@dataclass
class Reply:
    text: str | None = None
    function_calls: list[tuple[str, dict]] = field(default_factory=list)


def _retrieved_from(view: RequestView) -> tuple[list[str], list[dict]]:
    """Collect (ids, hit-dicts) from any search/notes tool responses seen so far."""
    ids: list[str] = []
    hits: list[dict] = []
    for name, response in view.function_responses:
        if not isinstance(response, dict):
            continue
        for item in response.get("results", []) + response.get("notes", []):
            if isinstance(item, dict) and item.get("id"):
                if item["id"] not in ids:
                    ids.append(item["id"])
                    hits.append(item)
    return ids, hits


# --- Per-agent offline responders -------------------------------------------

def researcher_responder(view: RequestView) -> Reply:
    """ReAct offline: search, then pull field notes, then answer with citations."""
    responded = view.responded_tools
    question = view.first_user_question

    if "search_papers" not in responded:
        return Reply(function_calls=[("search_papers", {"query": question, "top_k": 3})])

    ids, hits = _retrieved_from(view)
    if "get_lab_notes" not in responded and hits:
        topic = hits[0].get("topic") or "tool-use failures"
        return Reply(function_calls=[("get_lab_notes", {"topic": topic})])

    if not hits:
        return Reply(text=(
            "I could not find supporting entries in the knowledge base for that "
            "question, so I am not able to answer it with grounded evidence."
        ))

    top = hits[0]
    cites = ", ".join(ids[:3])
    body = " ".join((top.get("snippet") or top.get("body", "")).split())
    sentences = re.split(r"(?<=[.!?])\s+", body)
    body = " ".join(sentences[:2]).strip()
    answer = (
        f"{body} The field notes reinforce that this is an operational issue, "
        f"not a theoretical one. Sources: {cites}."
    )
    return Reply(text=answer)


def reviser_responder(view: RequestView) -> Reply:
    """Self-reflection offline: tighten the draft and guarantee a citation line."""
    draft = ATTRIBUTION_RE.sub("", view.last_model_answer).strip()
    if not draft:
        return Reply(text="No draft answer was available to revise.")
    ids = list(dict.fromkeys(ID_RE.findall(draft)))
    core = re.split(r"\s*Sources:\s*", draft)[0].strip().rstrip(".")
    core = " ".join(core.split())
    sources = ", ".join(ids) if ids else "none cited"
    return Reply(text=f"{core}. Sources: {sources}.")


def judge_responder(view: RequestView) -> Reply:
    """LLM-as-judge offline: score the answer it can read in context."""
    answer = ATTRIBUTION_RE.sub("", view.last_model_answer).strip()
    # The judge's own instruction embeds the retrieved-id list from session
    # state, so read it from the system text, tool observations from other
    # agents are not replayed into this agent's contents.
    retrieved_ids = list(dict.fromkeys(ID_RE.findall(view.system_text)))
    if not retrieved_ids:
        retrieved_ids, _ = _retrieved_from(view)
    cited_ids = list(dict.fromkeys(ID_RE.findall(answer)))
    word_count = len(answer.split())

    grounded_cites = [c for c in cited_ids if c in set(retrieved_ids)] if retrieved_ids else cited_ids
    groundedness = 5 if grounded_cites else (3 if cited_ids else 2)
    tool_selection = 5 if retrieved_ids else 2
    if word_count <= 130:
        conciseness = 5
    elif word_count <= 200:
        conciseness = 4
    else:
        conciseness = 3

    verdict = JudgeVerdict(
        groundedness=groundedness,
        tool_selection=tool_selection,
        conciseness=conciseness,
        rationale=(
            f"Answer cites {len(cited_ids)} source(s); "
            f"{len(retrieved_ids)} entries were retrieved; "
            f"{word_count} words."
        ),
        cited_ids=cited_ids,
    )
    return Reply(text=verdict.model_dump_json())


class ScriptedLLM(BaseLlm):
    """A BaseLlm whose output is computed by a responder callable."""

    model: str = "scripted-offline"
    agent_label: str = "scripted"
    latency_s: float = DEFAULT_OFFLINE_LATENCY_S

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    def __init__(self, responder: Callable[[RequestView], Reply], **kwargs: Any):
        super().__init__(**kwargs)
        object.__setattr__(self, "_responder", responder)

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"scripted-offline"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        view = view_request(llm_request)
        reply: Reply = self._responder(view)

        if reply.function_calls:
            parts = [
                types.Part(function_call=types.FunctionCall(name=name, args=args))
                for name, args in reply.function_calls
            ]
            out_text = ""
        else:
            parts = [types.Part(text=reply.text or "")]
            out_text = reply.text or ""

        if self.latency_s:
            await asyncio.sleep(self.latency_s)

        prompt_tokens = max(1, len(view.all_text) // 4)
        output_tokens = max(1, len(out_text) // 4 + 8 * len(reply.function_calls))
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            total_token_count=prompt_tokens + output_tokens,
        )
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            usage_metadata=usage,
            turn_complete=True,
        )


def researcher_model() -> ScriptedLLM:
    return ScriptedLLM(researcher_responder, agent_label="researcher")


def reviser_model() -> ScriptedLLM:
    return ScriptedLLM(reviser_responder, agent_label="reviser")


def judge_model() -> ScriptedLLM:
    return ScriptedLLM(judge_responder, agent_label="judge")
