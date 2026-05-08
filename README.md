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
| Bad system prompt | `agent/prompts.py` | Answers any animal; ignores tools; answers from memory instead of calling tools | `tool_called`, `correct_tool_selected` |
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
LANGSMITH_PROJECT=pocket-polly-demo-yourname
LANGSMITH_WORKSPACE_ID=your-demo-workspace-id
LANGCHAIN_TRACING_V2=true
DEMO_USER=your-name
```

> Use a unique `LANGSMITH_PROJECT` name per person (e.g. `pocket-polly-demo-morgan`). Multiple demo-ers sharing the same project name will mix traces and online evaluators. The project is created automatically on first use.

`DEMO_USER` additionally scopes your dataset and experiment names:
- Dataset: `pocket-polly-demo-dataset-morgan`
- Experiments: `pocket-polly-demo-morgan-<timestamp>`

**4. Run one-shot setup**
```bash
python -m scripts.setup
```

This does four things in one command:
1. **Creates the LangSmith project** by sending one trace (required before online evaluators can be registered)
2. **Creates the dataset** `pocket-polly-demo-dataset-<your-name>` with 10 curated test cases
3. **Runs an initial experiment** through the dataset with the buggy agent to establish "before" scores in LangSmith
4. **Creates 5 online evaluators** in the LangSmith Evaluators UI at 100% sampling rate — every future trace is automatically scored for `food_safety`, `scope_adherence`, `tool_usage`, `response_completeness`, and `factual_accuracy`

Only needs to be run once.

**5. (Optional) Generate additional traces**
```bash
python -m scripts.generate_traces
```

Runs 13 single-turn queries and 3 multi-turn threaded conversations through the buggy agent to populate LangSmith with trace and thread variety beyond the dataset examples.

**6. Add GitHub secrets** (for CI/CD)

In your fork: Settings → Secrets → Actions → add `ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_WORKSPACE_ID`, and `DEMO_USER`.

**7. Connect Engine**

In LangSmith Engine, connect your LangSmith project (`LANGSMITH_PROJECT`) and your GitHub fork so Engine can read traces and open PRs against your repo.

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

```bash
python -m scripts.cleanup
```

## Scripts

| Script | What it does |
|--------|-------------|
| `python -m scripts.setup` | One-shot setup: creates dataset, runs "before" experiment, creates 5 online evaluators |
| `python -m scripts.generate_traces` | Runs 13 single-turn queries + 3 multi-turn threads through the buggy agent |
| `python -m scripts.run_evals` | Runs offline evals against the dataset and prints scores |
| `python -m scripts.run_evals --skip-dataset` | Re-runs evals against existing dataset (used in CI) |
| `python -m scripts.run_evals --threshold 0.8` | Exits with code 1 if scores < 0.8 (used in CI) |
| `python -m scripts.cleanup` | Resets demo to clean state — see Cleanup section |
| `streamlit run app.py` | Start the PocketPolly chat UI |

## Evaluators

Four evaluators run in CI (offline). A mix of deterministic code evals and LLM-as-judge:

**Code evaluators** (deterministic, fast):
- **`tool_called`** — did the agent call at least one tool? Skips scope/decline examples. Goes 0→1 when the bad system prompt is fixed.
- **`correct_tool_selected`** — did the agent call the right tool for the question type? (`get_diet_advice` for food questions, `get_care_tips` for care, `lookup_species` for species info, no tool for out-of-scope). Goes 0→1 when fixed.
- **`response_not_empty`** — did the agent return a non-empty response?

**LLM-as-judge evaluators** (Claude Haiku scores 0 or 1):
- **`food_safety`** — did the agent warn about toxic foods and avoid recommending them? (catches missing grapes in tools.py)
- **`scope_adherence`** — did the agent stay parrot-only and decline non-parrot questions? (catches bad system prompt) — online evaluator only, not used in CI

## Online Evaluators

Online evaluators run automatically on every trace as it arrives in LangSmith — no manual scoring step needed. This gives Engine a continuous signal on live traffic, not just offline evals on a fixed dataset.

All 5 evaluators above are registered as online evaluators by `python -m scripts.setup`. Once registered, LangSmith scores every new trace automatically and surfaces the results in the trace view.

The evaluators use `{{output}}` (mustache format) mapped to `outputs["output"]` on each trace. The `@traceable` decorator on `invoke_agent` ensures every trace — including UI traces from the Streamlit app — has this output format.

## CI/CD

`.github/workflows/evals.yml` runs automatically on every PR to `main`.

Add these secrets to your repo (Settings → Secrets → Actions):
- `ANTHROPIC_API_KEY`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGSMITH_WORKSPACE_ID`
- `DEMO_USER`

Run `python -m scripts.setup` locally first so the dataset exists for CI to run against. `DEMO_USER` and `LANGSMITH_PROJECT` must match what you used locally — that's how CI finds the right dataset.

```
PR opened → GitHub Actions → run_evals --skip-dataset --threshold 0.8
                                          ↓
                               scores < 0.8 → ❌ blocks merge
                               scores ≥ 0.8 → ✅ mergeable
```

CI runs against the PR branch code — so Engine's fix produces high scores, creating the "after" experiment in LangSmith automatically. Because `--skip-dataset` fetches the existing dataset from LangSmith by name, any examples Engine adds to the dataset are included in the eval run automatically.

## Repo structure

```
agent/
├── prompts.py        # buggy system prompt (Bug 1 — Engine fixes this)
├── tools.py          # species lookup, care tips, diet advice (Bugs 2 & 3)
└── agent.py          # LangGraph ReAct agent (Bug 4 — max_tokens too low)

evals/
├── dataset.py        # creates per-user LangSmith dataset (10 curated examples)
└── evaluators.py     # 3 code evals + 2 LLM-as-judge offline evaluators (used in CI)

scripts/
├── setup.py          # one-shot setup: dataset + before experiment + online evaluators
├── generate_traces.py    # populate LangSmith with extra traces and threads
├── run_evals.py          # offline evals + CI threshold check
└── cleanup.py            # resets demo to clean state after presentation

.github/workflows/
└── evals.yml         # CI/CD: runs evals on every PR to main

app.py                # PocketPolly chat UI (Streamlit)
```

## Cleanup

Run after the demo to reset everything for the next presenter:

```bash
python -m scripts.cleanup
```

This does three things:
1. **Resets the dataset** — deletes Engine-added assertion examples and restores the original 10 curated test cases
2. **Deletes experiments** — removes all before/after experiment runs visible in the LangSmith dataset view
3. **Deletes online evaluators** — removes all run rules and platform evaluators, including any Engine added during the demo

After cleanup, run `python -m scripts.setup` to start fresh.
