# Field Notes, Agent Reliability

Short operational notes kept alongside the knowledge base. These are the
"what actually bit us" observations, separate from the tidier KB entries.

## topic: tool-use failures
Most production incidents we traced back to tool selection, not bad arguments.
When two tools had overlapping descriptions, selection accuracy dropped sharply.
Rewriting tool descriptions to be mutually exclusive recovered most of the loss
with no model change. Measure tool-call accuracy per step before touching prompts.

## topic: self-reflection
A reflection pass only helped when the critique had the retrieved evidence in
context. Reflection on the answer alone, with no evidence, mostly produced
sycophantic "looks good" revisions. Cap reflection at one or two rounds and add
an explicit stop condition, or the loop runs forever on ambiguous tasks.

## topic: llm-as-judge
Single-number quality scores were useless for debugging, everything landed at 4/5.
Splitting the score into groundedness, tool-selection, and conciseness made the
judge actionable and surfaced verbosity bias immediately. Keep a small held-out
set with reference answers to detect judge drift over time.

## topic: rag chunking
Swapping the vector store changed almost nothing. Changing chunk size and overlap
changed everything. Carry the source id on every chunk; an answer you cannot trace
back to a source is an answer you cannot trust in an enterprise deployment.

## topic: long-horizon agents
Final-answer eval lied to us. A run could end correct while step 3 was wrong and
step 5 silently compensated. Per-step tracing with stable ids was the only way to
localize where reliability actually broke across a long trajectory.

## topic: multi-agent coordination
Shared session state is the real interface between agents. The bugs were never in
the prompts, they were one agent overwriting a state key another agent depended
on. Make the state contract explicit and log every write.
