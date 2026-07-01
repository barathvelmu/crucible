# Crucible: System Design

**A multi-agent evaluation forge on Google ADK.**

A ReAct researcher answers questions grounded in a knowledge base; a reviser does one self-reflection pass; an LLM-as-judge scores the result on a multi-dimensional rubric. The whole thing is wrapped in an evaluation harness that captures groundedness, retrieval recall, and LLM-native operating metrics (tokens/sec, cost-per-request), with per-call tracing, and it runs end to end with **no API key** when you want it to.

> **TL;DR.** Most agent demos stop at a chat loop. The hard, valuable part is the layer *around* the agent: did it call the right tool, is every claim grounded in retrieved evidence, what did it cost, and where in the trajectory did it break. Crucible *is* that layer, kept small enough to read in one sitting and honest enough to run without faking results.

---

## Table of contents

1. [The story: why this exists](#1-the-story-why-this-exists)
2. [Design goals and non-goals](#2-design-goals-and-non-goals)
3. [System overview](#3-system-overview)
4. [The pipeline: hierarchical delegation over an explicit state contract](#4-the-pipeline-hierarchical-delegation-over-an-explicit-state-contract)
5. [The three agents](#5-the-three-agents)
6. [The retrieval and grounding layer](#6-the-retrieval-and-grounding-layer)
7. [The rubric and the structured verdict](#7-the-rubric-and-the-structured-verdict)
8. [Offline mode: ScriptedLLM](#8-offline-mode-scriptedllm)
9. [Observability: metrics and tracing](#9-observability-metrics-and-tracing)
10. [The evaluation harness](#10-the-evaluation-harness)
11. [Request lifecycle: one question, end to end](#11-request-lifecycle-one-question-end-to-end)
12. [Configuration and runtime modes](#12-configuration-and-runtime-modes)
13. [Testing strategy](#13-testing-strategy)
14. [Failure modes and how the design absorbs them](#14-failure-modes-and-how-the-design-absorbs-them)
15. [The path to production](#15-the-path-to-production)
16. [Design decisions and trade-offs](#16-design-decisions-and-trade-offs)
17. [Extensibility and future work](#17-extensibility-and-future-work)
18. [Appendix](#18-appendix)

---

## 1. The story: why this exists

Picture the demo everyone has seen. An agent gets a question, it "thinks," it calls a tool or two, it writes back a confident paragraph. The room nods. It looks like magic.

Then it goes to production and the magic curdles. The agent picks the wrong tool because two tools have overlapping descriptions. It cites a source that was never retrieved. It pads a thin answer into a verbose one and the eval, a single 1-to-5 quality score, rates it 4/5, same as everything else, because a single number cannot tell you *what* is wrong. Three weeks later someone asks "why did run #4,812 give the wrong answer?" and nobody can say, because the only thing logged was the final string.

That gap, between *the agent works in the demo* and *I can trust, measure, and debug the agent in production*, is the entire subject of this project. Crucible is built around a single conviction:

> The interesting engineering in agentic systems is not the chat loop. It is the **evaluation, grounding, and observability layer** that turns a plausible-looking generator into a system you can actually operate.

To make that concrete rather than abstract, Crucible builds a small but complete instance of the problem. It picks a domain it can be honest about, *agent reliability itself*, and assembles a three-stage pipeline where each stage deliberately exercises a named agentic pattern the field actually cares about:

| Stage | Pattern | What it does |
|-------|---------|--------------|
| `researcher` | **ReAct** (reason ↔ act) | Loops over retrieval tools, then answers with inline source citations. |
| `reviser` | **Self-reflection** | One critique pass: strips unsupported claims, cuts verbosity, keeps citations. |
| `judge` | **LLM-as-judge** | Scores groundedness / tool-selection / conciseness (1–5) as a structured verdict. |

Around those three agents sits the part that makes them trustworthy: a retrieval layer where **every chunk carries a source id**, a judge with a **multi-dimensional rubric** so quality does not collapse to one meaningless number, a **metrics ledger** that knows what each call cost and how fast it ran, a **trace** that shows which tool fired and what evidence came back, and an **offline mode** that runs the *real* orchestration with the model swapped out so the whole thing is reproducible with zero setup.

The knowledge base is synthesized for demonstration and says so plainly. The orchestration, the eval harness, the metrics, and the tracing are real.

---

## 2. Design goals and non-goals

### Goals

1. **Be the layer around the agent, not just the agent.** Grounding, scoring, tracing, and cost accounting are first-class, not afterthoughts.
2. **Make every claim auditable.** Every retrieved chunk has a stable source id (`KB-003`, `NOTE-001`). Answers cite ids; the judge checks that cited ids were actually retrieved. Provenance is the spine of the system.
3. **Refuse to collapse quality into one number.** Score groundedness, tool-selection, and conciseness separately, on purpose, so the eval is *actionable* and verbosity bias is *surfaced* rather than hidden.
4. **Run honestly with zero setup.** `make eval` produces a real scored report with no API key, no quota, and deterministic output, by swapping the *model*, not by faking the *pipeline*.
5. **Stay readable.** The whole system should fit in one sitting. Small, dependency-light, explicit.
6. **Map cleanly onto a production trajectory.** The same code path goes from offline prototype → AI Studio key → `adk web` developer UI → Vertex AI / Cloud Run, with nothing thrown away in between.

### Non-goals

- **Not** a vector-database benchmark. The retriever is a dependency-free TF-IDF + cosine index. The point is the *grounding contract* (source ids, auditability), not out-engineering a production vector store.
- **Not** a general agent framework. It is a focused reference implementation of one well-shaped problem.
- **Not** a real research corpus. The knowledge base is curated, synthesized briefing notes, labeled as such.
- **Not** dynamic/branching orchestration. The topology is strictly linear by design (see [§16](#16-design-decisions-and-trade-offs)).

### Principles that fall out of the goals

- **Provenance over cleverness.** A retrieved fact you cannot trace to a source is a fact you cannot trust in an enterprise deployment.
- **Explicit state contracts.** Agents communicate through named session-state keys, written and read in a fixed order. The contract is documented and testable, not implied.
- **The offline model reads context the way a real model would.** It is a stand-in, not a script of canned answers (see [§8](#8-offline-mode-scriptedllm)).
- **Measure what production measures.** Tokens/sec and cost-per-request are surfaced in every report because those are the operational numbers that matter when an agent ships.

---

## 3. System overview

```
                              user question
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │             SequentialAgent              │   ← hierarchical delegation
              │                "crucible"                │     over a shared-state contract
              └─────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐           ┌───────────────┐
│  researcher   │   tools  │    reviser    │           │     judge     │
│   (ReAct)     │───┐      │ (reflection)  │           │ (LLM-as-judge)│
└───────┬───────┘   │      └───────┬───────┘           └───────┬───────┘
        │           │              │                           │
 writes │   search_papers          │ writes                    │ writes
        │   get_lab_notes           │                           │
        ▼   list_corpus_topics      ▼                           ▼
 state["research_answer"]    state["revised_answer"]      state["verdict"]
        │                                                       │  (structured JSON)
        └──── state["crucible:retrieved_ids"] ──────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────┐
        │  Eval harness:  dataset → runner → scored report     │
        │  + per-call metrics ledger  + cross-agent trace tree │
        └─────────────────────────────────────────────────────┘
                                    │
                                    ▼
              reports/report.md   +   reports/report.json
```

There are three layers stacked on top of each other:

1. **The agents** (`crucible/agents/`), researcher, reviser, judge, composed by `pipeline.py`.
2. **The capabilities they stand on**, retrieval (`retrieval.py`), tools (`tools.py`), the rubric/verdict schema (`rubric.py`), and the offline model (`offline.py`).
3. **The harness that observes and scores them**, metrics (`metrics.py`), tracing (`tracing.py`), and the eval package (`eval/dataset.py`, `eval/runner.py`, `eval/report.py`).

The single entry point `crucible/agent.py` exports `root_agent = build_pipeline()`, which is what `adk web` and `adk run` discover. The eval runner builds its own pipeline instance per case so it can attach metric callbacks without polluting the web UI's agent.

---

## 4. The pipeline: hierarchical delegation over an explicit state contract

`crucible/agents/pipeline.py` assembles the three agents into a single `SequentialAgent` named `crucible`. This is the smallest possible expression of **hierarchical delegation**: one parent agent owns a fixed `researcher → reviser → judge` order, and the children never talk to each other directly. They talk through **session state**.

```python
return SequentialAgent(
    name="crucible",
    description="Crucible: a multi-agent evaluation forge. A ReAct researcher answers "
                "from a knowledge base, a reviser self-reflects on the draft, and an "
                "LLM-as-judge scores the result on a multi-dimensional rubric.",
    sub_agents=[researcher, reviser, judge],
)
```

### The state contract

This is the heart of the architecture, and it is deliberately the most boring, most explicit thing in the codebase, because the knowledge base itself says the bugs in multi-agent systems are almost never in the prompts; they are *one agent overwriting a state key another agent depends on* (`NOTE` on multi-agent coordination, `KB-005`). So the contract is written down and enforced by `output_key`:

| Step | Agent | Reads | Writes (`output_key`) |
|------|-------|-------|-----------------------|
| 1 | `researcher` | the user question | `research_answer` |
| 2 | `reviser` | `research_answer` | `revised_answer` |
| 3 | `judge` | `revised_answer` (falls back to `research_answer`), `crucible:retrieved_ids` | `verdict` (structured `JudgeVerdict`) |

Two of those keys are the visible hand-off chain (`research_answer → revised_answer → verdict`). The fourth, `crucible:retrieved_ids`, is the **provenance side-channel**: the tools write into it every time they return a chunk, so by the time the judge runs it can compare the ids the answer *cited* against the ids that were *actually retrieved*. That comparison is the difference between "the answer looks grounded" and "the answer is grounded."

### Why `SequentialAgent` and not the graph API

ADK 2.x ships a newer lower-level graph API (`google.adk.workflow`) for dynamic, branching topologies. Crucible's pipeline is *strictly linear*, research, then revise, then judge, always in that order, so `SequentialAgent` is the right primitive: it is the canonical, well-documented composition tool, and choosing the more powerful API would buy flexibility the design explicitly does not want. The code comments this trade-off inline so the next reader knows it was a decision, not an oversight.

### Offline injection

`build_pipeline(use_offline=...)` is the one seam where the model gets swapped. When offline, each agent is built with its scripted model (`offline.researcher_model()` etc.); otherwise the agents take their configured Gemini model names. Nothing else about the topology changes, same agents, same state contract, same tools. (See [§8](#8-offline-mode-scriptedllm).)

---

## 5. The three agents

Each agent is an ADK `LlmAgent`. What makes them distinct is their *instruction surface*, their *tools*, their *output_key*, and, for the judge, a structured `output_schema`.

### 5.1 Researcher, ReAct over retrieval tools

`crucible/agents/researcher.py`

The researcher is the only agent with tools. Its instruction tells it to work in an explicit reason-then-act loop:

1. If unsure what the KB covers, call `list_corpus_topics`.
2. Call `search_papers` to find the most relevant entries.
3. Once it has a topic, call `get_lab_notes` for the candid operational detail.
4. *Only then* write the answer.

The rules encode the grounding contract directly into the prompt: ground every claim in retrieved evidence, do not use outside knowledge, cite source ids inline (`(KB-003)`, `(NOTE-001)`), aim for under 120 words, and, critically, *if the KB does not cover the question, say so plainly*. That last rule is what makes the out-of-scope test case (see [§10](#10-the-evaluation-harness)) meaningful: a well-behaved agent abstains rather than hallucinating.

It writes to `output_key="research_answer"`.

### 5.2 Reviser, one self-reflection pass

`crucible/agents/reviser.py`

The reviser has **no tools**. Its job is a single critique pass over the researcher's draft. Its instruction is *dynamic*: it is a function of `ReadonlyContext` that reads `research_answer` out of state and embeds it directly into the prompt, then asks the model to:

- Remove any claim not supported by a cited source id (but keep the citations).
- Cut padding, hedging, and repetition, counter verbosity.
- Preserve correct content; do not rewrite a good answer into a worse one.
- Stay under 120 words and end with a `Sources:` line listing the ids.
- Output *only* the revised answer, not the critique.

This is the project's nod to the lesson from `KB-002`: reflection helps most when the critique is grounded and bounded to one or two rounds; left unbounded it sycophantically agrees with the draft or over-edits a correct answer into a worse one. So the design uses *exactly one* pass, structurally, there is no reflection loop to run away.

It writes to `output_key="revised_answer"`.

### 5.3 Judge, LLM-as-judge with a structured verdict

`crucible/agents/judge.py`

The judge is where evaluation becomes rigorous. It also has a dynamic instruction: it reads the `revised_answer` (falling back to `research_answer` if revision did not run), and the list of `crucible:retrieved_ids`, and embeds both into the prompt along with the rubric. It is told to be calibrated and discriminating, *do not give everything a 4*, and to penalize groundedness if the answer cites ids that were never retrieved or makes uncited claims.

The defining feature: it is built with `output_schema=JudgeVerdict`, so ADK forces the model to return a structured object, not prose:

```python
LlmAgent(
    name="judge",
    model=...,
    instruction=judge_instruction,       # dynamic, reads state
    output_schema=JudgeVerdict,          # forces structured output
    output_key="verdict",
)
```

That `JudgeVerdict` (see [§7](#7-the-rubric-and-the-structured-verdict)) is the contract between the model and the report. The report never has to parse free text; it reads typed integer scores.

It writes to `output_key="verdict"`.

---

## 6. The retrieval and grounding layer

`crucible/retrieval.py`, `crucible/tools.py`, `crucible/corpus/`

### 6.1 The corpus

Two files, one "Agent Reliability KB":

- **`corpus/papers.json`**, eight curated entries (`KB-001` … `KB-008`), each with a title, topic, tags, a summary, key points, and failure modes. They cover ReAct, self-reflection, the tool-use failure taxonomy, hallucinated tool calls, multi-agent coordination, LLM-as-judge bias, RAG chunking, and long-horizon credit assignment. The corpus is explicitly described in its own header as *synthesized for demonstration, not verbatim paper abstracts.*
- **`corpus/lab_notes.md`**, six candid "what actually bit us in production" field notes, one per topic, parsed into `NOTE-001` … `NOTE-006`. These are deliberately more operational and opinionated than the KB entries.

There is a nice piece of self-reference here: the knowledge base the agents reason over *is the design rationale for the system reasoning over it*. The judge's multi-dimensional rubric exists because `KB-006` and the LLM-as-judge field note say single scores are useless; the explicit state contract exists because the multi-agent field note says state collisions are the real bug. The system practices what its corpus preaches.

### 6.2 The index: dependency-free TF-IDF + cosine

`RetrievalIndex` builds a classic TF-IDF vector space over papers + notes:

- `tokenize()` lowercases, splits on non-alphanumerics, drops a small stopword set and single characters, and does a light de-pluralization (`failures → failure`).
- `_build()` computes document frequencies, then IDF as `log((1+N)/(1+df)) + 1`, then a normalized TF-IDF vector per document.
- `search()` vectorizes the query, scores every document by cosine similarity, sorts by `(score, id)` for **deterministic ties**, and returns the top-k hits with score > 0.
- `notes_for_topic()` prefers an exact topic match and falls back to ranked search.
- `get_index()` is `@lru_cache(maxsize=1)`, the index is a process singleton, built once.

It is deliberately *not* a vector database. There are no embeddings, no external services, no dependencies beyond the standard library. The point is not retrieval sophistication; it is the *contract*: every `Document` carries a stable `id`, and that id rides along with every chunk the tools return. Determinism (sorting by id on ties) is what lets the offline mode and the tests be reproducible.

### 6.3 The tools and the provenance side-channel

`tools.py` exposes three plain Python functions, `search_papers`, `get_lab_notes`, `list_corpus_topics`, collected into `ALL_TOOLS`. ADK turns each function's **docstring and type hints into the tool schema the model sees**, which means the wording of those docstrings is part of the prompt surface and is written with that in mind (clear, mutually-exclusive descriptions, exactly the fix `KB-003` prescribes for selection errors).

The subtle, important piece is `_record()`. Whenever a tool returns chunks, it appends their ids to `crucible:retrieved_ids` in session state (deduped, order-preserving, and defensive, a missing `tool_context` is a no-op rather than a crash). This is how the system separates *what the answer claims* from *what was actually retrieved*:

```python
RETRIEVED_IDS_KEY = "crucible:retrieved_ids"

def _record(tool_context, ids):
    existing = list(tool_context.state.get(RETRIEVED_IDS_KEY, []))
    for i in ids:
        if i not in existing:
            existing.append(i)
    tool_context.state[RETRIEVED_IDS_KEY] = existing
```

The judge reads that list and the trace renders it. Grounding stops being a vibe and becomes a checkable set-membership test.

---

## 7. The rubric and the structured verdict

`crucible/rubric.py`

The rubric is intentionally multi-dimensional. The module docstring states the thesis bluntly: *a single quality score collapses to "4/5 everything" and tells you nothing.* So the rubric has three orthogonal dimensions:

| Dimension | What it asks | Why it is separate |
|-----------|--------------|--------------------|
| **Groundedness** | Is every claim supported by retrieved evidence, with source ids cited? | The core trust signal; fabrication scores low. |
| **Tool selection** | Did the agent call the right tools in a sensible order and use the observations? | Guessing without retrieval scores low. |
| **Conciseness** | Is the answer as short as it can be while complete? | Explicitly counters **verbosity bias**, the well-known failure where judges reward longer answers. |

`JudgeVerdict` is a Pydantic model with three `int` fields constrained to `1 ≤ score ≤ 5`, a free-text `rationale`, and a `cited_ids` list. It exposes an `overall` property (the mean of the three, rounded to two decimals) and an `as_row()` helper for reporting. Because it is the agent's `output_schema`, the model is *forced* to produce something that validates against it, the eval harness downstream never parses prose.

Scoring conciseness as its own dimension is the single most opinionated design choice in the rubric, and it is there precisely because the field notes say verbosity bias "showed up immediately" once they split the score apart. Crucible bakes the lesson into the schema.

---

## 8. Offline mode: ScriptedLLM

`crucible/offline.py`

This is the feature that makes Crucible runnable by anyone, instantly, and reproducibly green in CI, and it is built to be *honest* about what it is.

`ScriptedLLM` is a real `google.adk.models.BaseLlm` subclass whose `generate_content_async` produces responses from a **Python callable** instead of a network round-trip. Crucially, **only the model is swapped**. The ADK runner, the tool-calling machinery, the session-state hand-offs, and the structured-output path are all the real thing. The offline run exercises the actual orchestration; it does not replay a canned transcript.

The responders read context the way a real model would:

- **`researcher_responder`** implements the ReAct loop by *inspecting what it has already seen*. If it has not yet called `search_papers`, it emits that function call. If it has search hits but no field notes yet, it calls `get_lab_notes` on the top hit's topic. If it has evidence, it writes a cited answer. If it found nothing, it abstains, the same out-of-scope behavior the prompt asks of the real model.
- **`reviser_responder`** strips ADK's `"[agent] said:"` attribution prefix, tightens the draft to its core sentence, and guarantees a `Sources:` line built from the ids actually present in the text.
- **`judge_responder`** scores the answer it can read in context. It pulls the retrieved-id list from the judge's own system instruction (because tool observations from *other* agents are not replayed into the judge's contents, a real ADK detail the offline model has to respect), compares cited ids against retrieved ids, and assigns scores: groundedness 5 if cited ids were genuinely retrieved, tool-selection 5 if anything was retrieved, conciseness banded by word count. It returns a real `JudgeVerdict` serialized to JSON.

Two things are explicitly *not* real offline, and the code and README say so: **token counts are estimated** (`len(text) // 4`) and **latency is a small simulated sleep** (`0.12s` per call) purely so the trace and throughput fields are populated; **cost is `$0` by construction**. The honesty principle is load-bearing here, offline mode exists so the project runs with zero setup, *not* to fake results. A real `--online` run produces the same 5/5/5 verdict on Q1 at roughly 4.7K tokens and ~$0.002/question, with real per-agent latencies.

---

## 9. Observability: metrics and tracing

### 9.1 The metrics ledger

`crucible/metrics.py`

The ledger collects one `UsageRecord` per model call: which agent, which model, prompt tokens, output tokens, latency. From those records it derives exactly the operational numbers the system promises:

- **Per-call:** `total_tokens`, `cost` (from the price table in `config.py`), `tokens_per_sec`.
- **Per-agent:** call count, summed tokens, summed cost, summed latency, model name.
- **Per-run summary:** total tokens, total cost, total latency, **p50 and p95 latency** (via linear-interpolation percentiles), **request-level throughput** (output tokens ÷ summed call latency), and **cost-per-request**.

`cost_usd()` reads `config.PRICING_USD_PER_1M`, illustrative list prices for `gemini-2.5-flash` (\$0.30 / \$2.50 per 1M in/out), `gemini-2.5-pro` (\$1.25 / \$10.00), and `scripted-offline` (\$0 / \$0). The throughput field is documented as *request-level* output throughput, not raw decode tok/s, because call latency includes the prompt round-trip, a small honesty detail that keeps the number from being misread.

### 9.2 The trace

`crucible/tracing.py`

ADK emits its own OpenTelemetry spans; this is a small, *inspectable* trace built on top so a report can show, per run, which agent did what, which tools fired, and what evidence came back. A `Span` has a name, a kind (`agent` | `tool` | `model`), a start/end, and an attribute bag; `duration_ms` is derived. A `Trace` is an ordered list of spans with a `render()` method that prints the now-familiar tree:

```
trace[Q1]
  ├─ [tool] search_papers  → KB-003, NOTE-001, KB-004
  ├─ [tool] get_lab_notes  → NOTE-001
  ├─ [agent] researcher  367.6ms  814 tok/3 calls  (scripted-offline)
  ├─ [agent] reviser     122.6ms  986 tok          (scripted-offline)
  └─ [agent] judge       124.3ms  1164 tok         (scripted-offline)
```

A key subtlety: each agent span is annotated with that agent's **aggregate** tokens and latency, summed across *all* of its model calls. The researcher above made three (two tool-producing turns plus the final answer) but appears as one span with the right totals. This is what makes the numbers correct even when a single model response batches multiple parallel tool calls, the trace is built from the event stream but the *numbers* come from the metrics ledger, which counts calls directly.

---

## 10. The evaluation harness

`crucible/eval/`

### 10.1 The dataset

`eval/dataset.py` defines six `EvalCase`s. Five are in-scope questions spanning the KB (tool-use failures, self-reflection, LLM-as-judge, RAG chunking, long-horizon agents), each with a soft `expects_ids` hint used to sanity-check retrieval recall (a *signal*, not a hard assertion). The sixth is the one that matters most:

```python
EvalCase(
    id="Q6",
    question="What is the airspeed velocity of an unladen swallow?",
    topic="out-of-scope",
    expects_ids=[],
    in_scope=False,
)
```

This is the **abstention test**. A production-grounding signal you actually care about is not "can it answer questions in the corpus", it is "does it *refuse* to answer questions outside the corpus instead of confabulating." Q6 confirms the agent retrieves nothing and declines, and the report excludes it from the scored mean so a correct abstention does not get punished as a low score.

### 10.2 The runner

`eval/runner.py` is the orchestrator:

1. Builds a fresh pipeline per case (offline or online) so callbacks attach cleanly.
2. Attaches `CallMetrics.before` / `.after` as ADK `before_model_callback` / `after_model_callback` on each sub-agent. These bracket every model call to record agent, model, tokens (from `usage_metadata`), and wall latency into the ledger.
3. Runs the pipeline through an ADK `Runner` over an `InMemorySessionService`, collecting the full event stream.
4. **Catches `RESOURCE_EXHAUSTED` / 429** errors gracefully and renders them as a human-readable `rate limit hit (429, daily free-tier quota)` rather than a stack trace, because the free tier is 20 requests/day and online runs *will* hit it.
5. Reads the final session state for the verdict, the answer, and the retrieved ids; builds the structural trace from the event stream (`_build_trace`); and assembles a `RunResult`.

The CLI (`python -m crucible.eval.runner`) supports `--online`, `-q "single question"`, `--out DIR`, `--no-write`, and `--delay` (which paces requests, 0s offline, 4s online, to stay under per-minute quota). When `--online`, it loads `crucible/.env` the same way `adk web` does, so `GOOGLE_API_KEY` is picked up consistently.

### 10.3 The report

`eval/report.py` turns `RunResult`s into both markdown and JSON. `EvalReport` computes the aggregates: `mean_overall` (in-scope, non-errored only), per-dimension means, and **retrieval recall** (fraction of expected ids that retrieval surfaced). `totals()` rolls up questions, model calls, total tokens, total cost, cost-per-question, wall time, and request-level throughput. `to_markdown()` renders the scores table, the aggregates, the LLM-native metrics block, and an example trace; `to_json()` emits the full machine-readable structure (including every trace) for programmatic consumption. Both are written to `reports/`.

---

## 11. Request lifecycle: one question, end to end

Tracing Q1, *"What is the dominant failure mode in tool-use agents?"*, through the whole system:

1. **Entry.** The runner creates a session and sends the question as a user `Content` to the `crucible` `SequentialAgent`.
2. **Researcher, turn 1 (reason → act).** Reads the question. Decides it needs evidence and emits a `search_papers(query="...", top_k=3)` call. ADK runs the tool; `search_papers` queries the TF-IDF index, gets back `KB-003`, `NOTE-001`, `KB-004`, and `_record()`s those ids into `crucible:retrieved_ids`. A `tool` span is logged.
3. **Researcher, turn 2 (act again).** Sees it has hits but no field notes. Emits `get_lab_notes(topic="tool-use failures")`. The note `NOTE-001` is returned and recorded; second `tool` span logged.
4. **Researcher, turn 3 (answer).** Now has evidence. Writes a concise answer citing the ids and stores it under `research_answer`. The metrics ledger now holds three records for `researcher`; the trace will fold them into one span annotated `814 tok / 3 calls`.
5. **Reviser (reflect).** Its dynamic instruction embeds `research_answer`. It tightens the draft, drops anything uncited, and guarantees a `Sources:` line. Writes `revised_answer`. One model call → one span.
6. **Judge (score).** Its dynamic instruction embeds `revised_answer` and the `crucible:retrieved_ids` list. It produces a `JudgeVerdict`, here `groundedness=5, tool_selection=5, conciseness=5`, `overall=5.0`, with `cited_ids=[KB-003, NOTE-001, KB-004]`, forced to validate against the schema and stored under `verdict`.
7. **Harvest.** The runner reads `verdict`, `revised_answer`, and `retrieved_ids` from state; builds the trace tree from the event stream, annotated with ledger numbers; assembles a `RunResult`.
8. **Report.** Across all six cases, `EvalReport` computes means, recall, and totals, and writes `reports/report.md` and `reports/report.json`.

Every arrow in that chain is a documented state key or a logged span. There is no step where the system "just knows" something, which is the entire point.

---

## 12. Configuration and runtime modes

`crucible/config.py` centralizes everything tunable so nothing is hard-coded downstream:

- **Models per role**, each overridable by env var (`CRUCIBLE_RESEARCHER_MODEL`, `CRUCIBLE_REVISER_MODEL`, `CRUCIBLE_JUDGE_MODEL`), all defaulting to `gemini-2.5-flash` so the pipeline runs on the **free AI Studio tier** out of the box. (`gemini-2.5-pro` is not on the free tier; set the judge to pro on a paid tier for a stronger, separate judge, a clean way to avoid self-preference bias.)
- **Pricing table** for cost accounting.
- **`OFFLINE`** toggle, read from `CRUCIBLE_OFFLINE`.

Three runtime modes, one code path:

| Mode | Command | Needs key? | What it's for |
|------|---------|:----------:|---------------|
| **Offline** | `make eval` | No | Deterministic full-dataset run; the good first run; CI. |
| **Online** | `make eval-online` | Yes | Real Gemini over the dataset, paced to respect quota. |
| **Web UI** | `make web` → `localhost:8000` | Yes | ADK developer UI with live trace visualization. |
| **One-off** | `make run Q="..."` | No (offline default) | Single ad-hoc question. |

---

## 13. Testing strategy

`tests/` holds **26 tests, all fully offline** (no key, no network, no quota). They are organized by concern and verify the contracts, not just the happy path:

- `test_retrieval.py` (6), tokenization, TF-IDF/cosine ranking, deterministic ties, topic lookup.
- `test_tools.py` (5), tool return shapes and, importantly, that provenance ids land in session state.
- `test_rubric.py` (4), `JudgeVerdict` validation, score bounds, `overall` math.
- `test_metrics.py` (4), cost math, per-agent aggregation, percentile/throughput summary.
- `test_pipeline_offline.py` (4), the full pipeline runs end to end offline and produces a real verdict.
- `test_agent_discovery.py` (2), `root_agent` is discoverable by `adk` and wired correctly.
- `test_report.py` (1), report aggregation and rendering.

The fact that the *entire* pipeline is testable offline, including a real scored verdict, is a direct payoff of the ScriptedLLM design. CI never needs a secret.

---

## 14. Failure modes and how the design absorbs them

The corpus catalogs how agents fail; the architecture is built to catch those same failures. The symmetry is intentional.

| Failure mode (from the KB) | How Crucible's design responds |
|---|---|
| **Wrong-tool selection** (`KB-003`), the dominant production failure | Three tools with clear, mutually-exclusive docstrings (the prescribed fix); tool-selection is its own scored rubric dimension. |
| **Hallucinated / fabricated tool calls** (`KB-004`) | The agent is constrained to a registered tool schema; the judge cross-checks cited ids against the recorded `retrieved_ids` set. |
| **Sycophantic / runaway reflection** (`KB-002`) | Reflection is structurally bounded to exactly one pass, there is no loop to run away. |
| **Verbosity bias in scoring** (`KB-006`) | Conciseness is a first-class rubric dimension, scored separately so length cannot quietly inflate quality. |
| **Rubric collapse** ("everything is a 4") | Three orthogonal dimensions + an instruction to be discriminating; the report shows per-dimension means so collapse is visible. |
| **State-key collisions across hand-offs** (`KB-005`) | An explicit, documented `output_key` contract; each key written by exactly one agent. |
| **Final-answer-only eval hides where it broke** (`KB-008`) | Per-step tracing with stable ids and per-agent metrics localizes failures along the trajectory. |
| **Lost provenance** (`KB-007`) | Every chunk carries a source id end to end; an answer that cannot be traced to a source scores low on groundedness. |
| **Hallucinating out-of-scope answers** | The abstention test case (Q6) verifies the agent declines instead of confabulating. |
| **Quota exhaustion (429) mid-run** | The runner catches `RESOURCE_EXHAUSTED`, renders a clean message, and still emits a report for the cases that completed. |

---

## 15. The path to production

Crucible is shaped so the prototype *becomes* the production system rather than getting thrown away:

1. **Offline prototype**, `make eval`, deterministic, zero setup. Develop the orchestration and eval logic here.
2. **AI Studio key**, drop a `GOOGLE_API_KEY` into `crucible/.env`, flip off offline, and the *same agents* call real Gemini. Validate behavior and real cost.
3. **`adk web`**, the ADK developer UI for live, interactive trace visualization while iterating on prompts and tools.
4. **Vertex AI / Cloud Run**, ADK is built for this hand-off. The model config already routes through env vars, so the same package deploys against Vertex with no code change; the in-memory session service is swapped for a persistent one; and the eval harness becomes a CI/CD gate that scores every change before it ships.

**Scaling considerations** the design anticipates:

- **Throughput**, cases run sequentially today (with quota-respecting delays); the per-case design is embarrassingly parallel once quota allows, since each case gets its own session and pipeline instance.
- **Retrieval**, the TF-IDF singleton index is a stand-in for the grounding *contract*; swapping in a managed vector store (e.g. Vertex AI Search) is a `retrieval.py`-local change because every consumer only depends on the `id`-carrying `Document`/`Hit` interface.
- **Judge independence**, moving the judge to a different model family (pro, or a non-Gemini model) is a one-line env change and directly mitigates self-preference bias.
- **Observability**, the lightweight trace coexists with ADK's native OpenTelemetry spans, so production telemetry can flow to Cloud Trace while the human-readable report stays for eval gates.

---

## 16. Design decisions and trade-offs

Short architecture-decision records for the choices a reviewer would question.

**ADR-1, `SequentialAgent`, not the graph API.**
*Decision:* compose with the linear primitive. *Why:* the topology is genuinely fixed (research → revise → judge); the dynamic graph API would add branching power the design does not want and obscure the explicit state contract. *Trade-off:* no conditional routing; accepted because linearity is the feature.

**ADR-2, Dependency-free TF-IDF retriever.**
*Decision:* hand-rolled TF-IDF + cosine over a managed vector DB. *Why:* the goal is the grounding/provenance contract, runnable with zero infra, not retrieval SOTA. *Trade-off:* not semantically powerful; accepted and isolated behind a swappable interface.

**ADR-3, Multi-dimensional rubric, not a single score.**
*Decision:* score groundedness/tool-selection/conciseness separately. *Why:* a single number collapses to "4/5 everything" and surfaces nothing actionable; separating conciseness specifically exposes verbosity bias. *Trade-off:* slightly more judge-prompt complexity; clearly worth it.

**ADR-4, `output_schema` on the judge.**
*Decision:* force structured `JudgeVerdict` output. *Why:* the report should never parse prose; typed scores are robust and testable. *Trade-off:* the judge cannot freely format; that is the point.

**ADR-5, ScriptedLLM swaps the model, not the pipeline.**
*Decision:* offline mode runs the real ADK orchestration with a scripted `BaseLlm`. *Why:* zero-setup reproducibility and green CI *without* faking results; the trace and scores come from the actual pipeline. *Trade-off:* tokens are estimated and latency simulated offline, documented explicitly so the numbers are never over-claimed.

**ADR-6, Default all roles to `gemini-2.5-flash`.**
*Decision:* free-tier-friendly defaults, env-overridable. *Why:* anyone can run it online for ~\$0.002/question; the judge can be upgraded to pro on a paid tier for independence. *Trade-off:* a same-model judge has mild self-preference risk by default, mitigated by the one-line override.

**ADR-7, Provenance as a session-state side-channel.**
*Decision:* tools write `crucible:retrieved_ids` into state rather than threading ids through return values. *Why:* the judge needs ground truth about what was retrieved independent of what the answer claims. *Trade-off:* one more state key to manage, handled by the explicit contract and a dedup-safe writer.

---

## 17. Extensibility and future work

The seams are deliberately placed so the obvious next steps are local changes:

- **Swap the retriever**, replace `retrieval.py`'s index with embeddings or a managed vector store; consumers only touch `Document`/`Hit`.
- **Add a rubric dimension**, append to `RUBRIC` and a field to `JudgeVerdict`; the report's per-dimension means pick it up automatically.
- **Add eval cases**, extend `DATASET`; recall and aggregates flow through.
- **Bounded multi-round reflection**, turn the single reviser into a small loop with an explicit stop condition (exactly the safe pattern `KB-002` describes).
- **Pairwise / reference-based judging**, add held-out reference answers to calibrate the judge and detect drift (the field-note recommendation).
- **Parallel case execution**, run cases concurrently once quota allows; the per-case isolation already supports it.
- **Persistent sessions + Cloud Trace**, swap `InMemorySessionService` and wire the OTel spans to a backend for production observability.

---

## 18. Appendix

### 18.1 File map

```
crucible/
├── crucible/
│   ├── agent.py            # root_agent, the adk web / adk run entry point
│   ├── config.py           # models, illustrative pricing, offline toggle
│   ├── retrieval.py        # dependency-free TF-IDF + cosine index (RAG-like)
│   ├── tools.py            # search_papers / get_lab_notes / list_corpus_topics
│   ├── rubric.py           # rubric + JudgeVerdict (structured output schema)
│   ├── metrics.py          # tokens, latency, tokens/sec, cost-per-request
│   ├── tracing.py          # cross-agent span trace
│   ├── offline.py          # ScriptedLLM, zero-key deterministic model
│   ├── agents/
│   │   ├── researcher.py   # ReAct over retrieval tools     → research_answer
│   │   ├── reviser.py      # one self-reflection pass        → revised_answer
│   │   ├── judge.py        # LLM-as-judge, output_schema     → verdict
│   │   └── pipeline.py     # SequentialAgent composition
│   ├── corpus/
│   │   ├── papers.json      # 8 KB entries (KB-001 … KB-008)
│   │   └── lab_notes.md     # 6 field notes (NOTE-001 … NOTE-006)
│   └── eval/
│       ├── dataset.py       # 6 cases incl. one out-of-scope abstention test
│       ├── runner.py        # CLI runner + metric callbacks + trace builder
│       └── report.py        # aggregates + markdown/JSON rendering
└── tests/                   # 26 tests, fully offline
```

### 18.2 Session-state keys

| Key | Written by | Read by | Type |
|-----|-----------|---------|------|
| `research_answer` | researcher | reviser, judge (fallback) | str |
| `revised_answer` | reviser | judge | str |
| `verdict` | judge | runner / report | structured `JudgeVerdict` |
| `crucible:retrieved_ids` | tools (`_record`) | judge, runner / trace | list[str] |

### 18.3 Knowledge base index

| id | Title | Topic |
|----|-------|-------|
| KB-001 | ReAct: interleaving reasoning and acting | reasoning-action loops |
| KB-002 | Self-reflection and verbal reinforcement | self-reflection |
| KB-003 | Tool-use failure taxonomy | tool-use failures |
| KB-004 | Hallucinated tool calls | hallucinated tool calls |
| KB-005 | Multi-agent coordination and delegation | multi-agent coordination |
| KB-006 | LLM-as-judge: reliability and bias | llm-as-judge |
| KB-007 | RAG chunking and retrieval grounding | rag chunking |
| KB-008 | Long-horizon credit assignment | long-horizon agents |
| NOTE-001 … 006 | Field notes (one per topic) | the candid operational counterparts |

### 18.4 Glossary

- **ReAct**, an agent pattern that interleaves free-form *reasoning* with concrete tool *actions*, feeding each observation back into the next reasoning step.
- **Self-reflection**, a critique pass where the agent reviews and revises its own draft against the task and evidence, with no weight updates.
- **LLM-as-judge**, using a language model to score another model's output against a rubric; scales evaluation but inherits biases (position, verbosity, self-preference).
- **Hierarchical delegation**, composing specialist agents under a coordinator (here, a `SequentialAgent`) that owns the order and the shared-state contract.
- **Groundedness**, the property that every claim in an answer is supported by retrieved evidence with a citable source id.
- **Verbosity bias**, the tendency of LLM judges to reward longer answers; countered here by scoring conciseness explicitly.
- **Provenance**, the source id carried with every retrieved chunk, end to end, so any claim can be audited back to its evidence.

---

> *Built as a focused reference implementation. The knowledge base is synthesized for demonstration; the orchestration, the evaluation harness, the metrics, and the tracing are real.*
