# engine-demo-agent

A parrot expert chatbot with intentional bugs, built to demonstrate LangSmith Engine's ability to identify issues in agent traces and propose fixes via PR. The agent answers questions about parrot care using three tools: species lookup, care tips, and diet advice.

## What this demos

1. **Engine identifies bugs from traces** — the agent has bugs in the prompt, tools, and agent config that cause bad responses
2. **Engine proposes a PR fix** — targets the root cause code and opens a PR on your fork
3. **Offline evals in CI/CD** — the PR can't merge until eval scores pass a threshold
4. **Before/after scores in LangSmith** — "before" experiment run locally pre-demo; "after" created automatically by CI on Engine's PR

## The bugs

Bugs are spread across three files so Engine has to reason about code, not just prompts:

| Bug | File | Effect | Caught by |
|-----|------|--------|-----------|
| Bad system prompt | `agent/prompts.py` | Answers any animal; ignores tools; never says it doesn't know | `scope_adherence`, `tool_usage` |
| Grapes missing from toxic list | `agent/tools.py` | Agent tells users raisins are safe for parrots | `food_safety` |
| Wrong budgie lifespan | `agent/tools.py` | Returns "20-30 years" instead of the correct "5-10 years" | `factual_accuracy` |
| `max_tokens=300` | `agent/agent.py` | Truncates responses on complex questions | `response_completeness` |

## Setup

**1. Fork and clone this repo**

**2. Create a virtual environment**
```bash
uv sync
source .venv/bin/activate
```

Or with pip:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**3. Configure environment**
```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=your-key
LANGSMITH_API_KEY=your-demo-workspace-api-key
LANGSMITH_PROJECT=parrot-expert-demo-yourname
LANGSMITH_WORKSPACE_ID=your-demo-workspace-id
LANGCHAIN_TRACING_V2=true
DEMO_USER=your-name
```

> Use a unique `LANGSMITH_PROJECT` name per person (e.g. `parrot-expert-demo-morgan`). Multiple demo-ers sharing the same project name will mix traces and online evaluators. The project is created automatically on first use.

`DEMO_USER` additionally scopes your dataset and experiment names:
- Dataset: `parrot-expert-demo-dataset-morgan`
- Experiments: `parrot-demo-morgan-<timestamp>`

**4. Run one-shot setup**
```bash
python -m scripts.setup
```

This does three things in one command:
1. **Creates the dataset** `parrot-expert-demo-dataset-<your-name>` with hand-crafted and LLM-generated test cases
2. **Runs an initial experiment** through the dataset with the buggy agent to establish "before" scores in LangSmith (this also creates the LangSmith project)
3. **Creates 5 online evaluators** in the LangSmith Evaluators UI at 100% sampling rate — every future trace is automatically scored for `food_safety`, `scope_adherence`, `tool_usage`, `response_completeness`, and `factual_accuracy`

Only needs to be run once.

**5. (Optional) Generate additional traces**
```bash
python -m scripts.generate_traces
```

Runs 13 hardcoded queries through the buggy agent to populate LangSmith with more trace variety beyond the dataset examples.

**6. Add GitHub secrets** (for CI/CD)

In your fork: Settings → Secrets → Actions → add `ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_WORKSPACE_ID`, and `DEMO_USER`.

## Demo flow

### Before the demo

```bash
# One-shot setup: creates dataset, runs initial "before" experiment, sets up online evaluators
python -m scripts.setup

# (Optional) generate more traces beyond the dataset examples
python -m scripts.generate_traces

# Start the chat UI
streamlit run app.py
```

### During the demo

1. Show PocketPolly UI — ask buggy questions (raisins, golden retriever care, how long do budgies live)
2. Show traces in LangSmith with online eval scores (`food_safety`, `scope_adherence`, etc.)
3. Engine analyzes traces and identifies root causes across prompt and code
4. Engine opens a PR on your fork
5. GitHub Actions runs evals on the PR branch (fixed code) — scores pass ✅
6. Show the experiments in LangSmith — before/after score comparison
7. Merge the PR

### After the demo

> ⚠️ Cleanup script to be implemented — see Cleanup section below.

## Scripts

| Script | What it does |
|--------|-------------|
| `python -m scripts.setup` | One-shot setup: creates dataset, runs "before" experiment, creates 5 online evaluators |
| `python -m scripts.generate_traces` | Runs 13 hardcoded queries through the buggy agent to add trace variety |
| `python -m scripts.run_evals` | Runs offline evals against the dataset and prints scores |
| `python -m scripts.run_evals --skip-dataset` | Re-runs evals against existing dataset (used in CI) |
| `python -m scripts.run_evals --threshold 0.8` | Exits with code 1 if scores < 0.8 (used in CI) |
| `python -m scripts.cleanup` | ⚠️ To be implemented — see Cleanup section |
| `streamlit run app.py` | Start the PocketPolly chat UI |

## Evaluators

Five LLM-as-judge evaluators score each response 0 or 1. The same evaluators are used for both offline dataset runs and online trace scoring:

- **`food_safety`** — did the agent warn about toxic foods? (catches missing grapes in tools.py)
- **`scope_adherence`** — did the agent stay parrot-only? (catches bad system prompt)
- **`tool_usage`** — did the agent call a tool instead of answering from memory? (catches bad system prompt)
- **`response_completeness`** — is the response complete and untruncated? (catches max_tokens=300)
- **`factual_accuracy`** — is species/care data correct? (catches wrong budgie lifespan)

## Online Evaluators

Online evaluators run automatically on every trace as it arrives in LangSmith — no manual scoring step needed. This gives Engine a continuous signal on live traffic, not just offline evals on a fixed dataset.

All 5 evaluators above are registered as online evaluators by `python -m scripts.setup`. Once registered, LangSmith scores every new trace automatically and surfaces the results in the trace view.

The evaluators use `{{output}}` (mustache format) mapped to `outputs["output"]` on each trace. The `@traceable` decorator on `invoke_agent` ensures every trace — including UI traces from the Streamlit app — has this output format.

## CI/CD

`.github/workflows/evals.yml` is currently **disabled** (manual trigger only). To enable automatic eval runs on every PR to `main`:

1. Change the trigger in `evals.yml` to:
```yaml
on:
  pull_request:
    branches: [main]
```

2. Add these secrets to your fork (Settings → Secrets → Actions):
   - `ANTHROPIC_API_KEY`
   - `LANGSMITH_API_KEY`
   - `LANGSMITH_PROJECT`
   - `LANGSMITH_WORKSPACE_ID`
   - `DEMO_USER`

3. Run `python -m scripts.setup` locally first so the dataset exists for CI to run against.

When enabled, the flow is:
```
PR opened → GitHub Actions → run_evals --skip-dataset --threshold 0.8
                                          ↓
                               scores < 0.8 → ❌ blocks merge
                               scores ≥ 0.8 → ✅ mergeable
```

CI runs against the PR branch code — so Engine's fix produces high scores, creating the "after" experiment in LangSmith automatically.

## Repo structure

```
agent/
├── prompts.py        # buggy system prompt (Bug 1 — Engine fixes this)
├── tools.py          # species lookup, care tips, diet advice (Bugs 2 & 3)
└── agent.py          # LangGraph ReAct agent (Bug 4 — max_tokens too low)

evals/
├── dataset.py        # creates per-user LangSmith dataset (hand-crafted + LLM-generated)
└── evaluators.py     # food_safety + scope_adherence offline evaluators (used in CI)

scripts/
├── setup.py          # one-shot setup: dataset + before experiment + online evaluators
├── generate_traces.py    # populate LangSmith with extra traces
└── run_evals.py          # offline evals + CI threshold check

.github/workflows/
└── evals.yml         # CI/CD: runs evals on every PR (manual trigger only — see CI/CD section)

app.py                # PocketPolly chat UI (Streamlit)
```

## Cleanup

> ⚠️ To be implemented.
