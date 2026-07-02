# CRUCIBLE

**A multi-agent evaluation forge on Google ADK.** A ReAct researcher answers
questions grounded in a knowledge base, a reviser does one self-reflection pass,
and an LLM-as-judge scores the result on a multi-dimensional rubric, wrapped in
an eval harness that captures groundedness, retrieval recall, and LLM-native
metrics (tokens/sec, cost-per-request) with per-call tracing.

Built on [Google's Agent Development Kit](https://google.github.io/adk-docs/)
(`google-adk` 2.1). Runs three ways: a **zero-setup offline mode** with no API
key, real **Gemini** via an AI Studio key, or the **`adk web`** developer UI.

---

## Why this exists

Most "agent demos" stop at a chat loop. The harder, more valuable part, and the
part this builds, is the layer *around* the agent: did it call the right tools,
is every claim grounded in retrieved evidence, what did it cost, and where in the
trajectory did it break. Crucible is that layer, kept small enough to read in one
sitting.

The three agents deliberately exercise three named agentic patterns:

| Agent | Pattern | What it does |
|-------|---------|--------------|
| `researcher` | **ReAct** (reason ↔ act) | Loops over retrieval tools, then answers with inline source citations. |
| `reviser` | **Self-reflection** | One critique pass: strips unsupported claims, cuts verbosity, keeps citations. |
| `judge` | **LLM-as-judge** | Scores groundedness / tool-selection / conciseness (1–5) as a structured verdict. |

They are composed with `SequentialAgent` via **hierarchical delegation** over an
explicit shared-state contract (`research_answer` → `revised_answer` → `verdict`).

```
            user question
                  │
                  ▼
      ┌───────────────────────┐     tools: search_papers,
      │   researcher (ReAct)   │────▶ get_lab_notes, list_corpus_topics
      └───────────┬───────────┘     → writes state["research_answer"]
                  │
                  ▼
      ┌───────────────────────┐
      │  reviser (reflection)  │────▶ writes state["revised_answer"]
      └───────────┬───────────┘
                  │
                  ▼
      ┌───────────────────────┐
      │  judge (LLM-as-judge)  │────▶ writes state["verdict"]  (structured JSON)
      └───────────┬───────────┘
                  │
                  ▼
        scored eval report  (+ trace + tokens/cost/latency)
```

---

## Quickstart

```bash
make install            # python3.13 venv + editable install

# 1) Offline, no API key, no quota, deterministic. Good first run.
make eval               # runs the full dataset, writes reports/report.{md,json}

# 2) Real Gemini, copy .env.example to crucible/.env, add GOOGLE_API_KEY
make eval-online

# 3) ADK developer web UI with live trace visualization
make web                # http://localhost:8000  → pick "crucible"

# One-off question
make run Q="What is the dominant failure mode in tool-use agents?"

make test               # 26 tests, all offline (no key needed)
```

A free [AI Studio](https://aistudio.google.com/apikey) key is enough, the
defaults use `gemini-2.5-flash`, which is served on the free tier. (`gemini-2.5-pro`
is not on the free tier; set `CRUCIBLE_JUDGE_MODEL=gemini-2.5-pro` on a paid tier
for a stronger, separate judge.)

### Offline mode, honestly

`CRUCIBLE_OFFLINE=1` swaps Gemini for `ScriptedLLM`, a `BaseLlm` whose responses
come from a Python function instead of the network. It still runs the **real ADK
orchestration**, tool calls, state hand-offs, structured output, so the trace
and scores are produced by the actual pipeline; only the model is swapped. Token
counts are estimated and latency is a small simulated delay (cost is $0). It
exists so the project runs and tests green with zero setup, not to fake results.

---

## Sample output

Offline run (`make eval`, reproducible with no key), trimmed to the first row:

```
| Q  | Topic             | Ground | Tool | Concise | Overall | Retrieved                |
|----|-------------------|:------:|:----:|:-------:|:-------:|--------------------------|
| Q1 | tool-use failures |   5    |  5   |    5    |   5.0   | KB-003, NOTE-001, KB-004 |

LLM-native metrics
- Questions: 6 · model calls: 29 · total tokens: 16,513
- Throughput: 349 output tok/s (request-level) · wall time: 3.7s

trace[Q1]
  ├─ [tool] search_papers  → KB-003, NOTE-001, KB-004
  ├─ [tool] get_lab_notes  → NOTE-001
  ├─ [agent] researcher  367.6ms  814 tok/3 calls  (scripted-offline)
  ├─ [agent] reviser     122.6ms  986 tok          (scripted-offline)
  └─ [agent] judge       124.3ms  1164 tok         (scripted-offline)
```

The trace attributes each agent its *aggregate* tokens and latency across all of
its model calls (the researcher made 3: two tool-producing turns plus the answer).
Offline cost is $0 and latency is simulated; a real `--online` run of the same
question on `gemini-2.5-flash` returns the same 5/5/5 verdict at roughly 4.7K
tokens and ~$0.002 per question, with real per-agent latencies in the seconds.

The dataset also includes one deliberately **out-of-scope** question to confirm
the agent abstains (retrieves nothing, refuses to answer) instead of
hallucinating, the groundedness signal you actually care about in production.

---

## Limits, honestly

This repo was put through a code audit (adversarial probes, cross-checked math), and the honest findings belong next to the sample output above:

- **Offline, every in-scope question scores 5/5/5, and that proves plumbing, not intelligence.** The scripted researcher only cites what it retrieved and always appends a sources line, so the judge's checks always pass. The offline run verifies the harness end to end; the discriminating signal needs an online run or a deliberately harder dataset.
- **The groundedness check is coarse at the edges.** It only drops the score when *every* cited id is ungrounded, so an answer mixing one real citation with invented ones still scores 5.
- **Recall is only ever asserted at 1.0 in the tests.** A recall function stuck returning 1.0 would pass today's suite; a designed miss-case is the missing test.
- **The offline report's token and throughput numbers are estimates.** The code's docstrings say so; the generated report should carry the same label.

The audit also confirmed the load-bearing claims: the TF-IDF/cosine and percentile math check out against numpy, `ScriptedLLM` genuinely drives the real ADK Runner rather than replaying a transcript, and the judge does discriminate when given bad input (out-of-scope and hallucinated-citation probes score low). The list above is the worklist for making the eval discriminate, not a confession that it is hollow.

---

## Project layout

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
│   ├── agents/             # researcher, reviser, judge, pipeline
│   ├── corpus/             # Agent Reliability knowledge base + field notes
│   └── eval/               # dataset, runner (CLI), report (md/json)
└── tests/                  # 26 tests, fully offline
```

---

## What this demonstrates

| Capability | Where it shows up |
|---|---|
| Multi-agent systems (LangGraph / CrewAI / **ADK**) | `agents/`, three agents composed with `SequentialAgent` |
| Patterns: **ReAct, self-reflection, hierarchical delegation** | researcher / reviser / sequential composition |
| **High-performance evaluation pipelines** | `eval/`, dataset → runner → scored report |
| **Observability & granular tracing** | `tracing.py` + per-call model callbacks in `eval/runner.py` |
| **LLM-native metrics** (tokens/sec, cost-per-request) | `metrics.py`, surfaced in every report |
| **RAG-like retrieval / grounding** | `retrieval.py` + `tools.py`, every chunk carries a source id |
| State-management across hand-offs | explicit `output_key` contract in the pipeline |
| Prototype → production path | offline prototype → AI Studio key → `adk web` → Vertex/Cloud Run |

> Built as a focused reference implementation. The knowledge base is synthesized
> for demonstration; the orchestration, eval harness, and metrics are real.
