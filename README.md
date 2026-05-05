# engine-demo-agent

A parrot expert chatbot with intentional bugs, built to demonstrate LangSmith Engine's ability to identify issues in agent traces and propose fixes via PR. The agent answers questions about parrot care using three tools: species lookup, care tips, and diet advice.

## What this demos

1. **Engine identifies bugs from traces** — the agent has a bad system prompt that causes hallucinations and out-of-scope answers
2. **Engine proposes a PR fix** — a system prompt change that makes the agent use its tools and stay parrot-only
3. **Offline evals baked into CI/CD** — the PR can't merge until eval scores pass a threshold
4. **Before/after scores in LangSmith** — "before" experiment run locally pre-demo; "after" created automatically by CI on Engine's PR

## The bugs

Both bugs are in `agent/prompts.py`. The system prompt tells the agent to:
- Answer questions about **any animal** (not just parrots)
- Answer **from its own knowledge** instead of calling tools
- **Never say it doesn't know** — forcing overconfident, often wrong answers


## Setup

**1. Fork and clone this repo**

**2. Create a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Configure environment**
```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=your-key
LANGSMITH_API_KEY=your-demo-workspace-api-key
LANGSMITH_PROJECT=parrot-expert-demo
LANGSMITH_WORKSPACE_ID=your-demo-workspace-id
LANGCHAIN_TRACING_V2=true
DEMO_USER=your-name
```

`DEMO_USER` scopes your dataset and experiments so multiple demo-ers in the same workspace don't collide:
- Dataset: `parrot-expert-demo-dataset-morgan`
- Experiments: `parrot-demo-morgan-<timestamp>`

**4. Add GitHub secrets** (for CI/CD)

In your fork: Settings → Secrets → Actions → add `ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_WORKSPACE_ID`, and `DEMO_USER`.

## Demo flow

### Before the demo

```bash
# Populate LangSmith with buggy traces
python -m scripts.generate_traces

# Create dataset + run offline evals (establishes the "before" experiment with low scores)
python -m scripts.run_evals

# Start the chat UI
streamlit run app.py
```

### During the demo

1. Show PocketPolly UI — ask buggy questions (avocado, golden retriever care)
2. Show traces in LangSmith — Engine analyzes them
3. Engine identifies the system prompt as root cause
4. Engine opens a PR on your fork
5. GitHub Actions runs evals on the PR branch (fixed code) — scores pass ✅
6. Show the experiments in LangSmith — before/after score comparison
7. Merge the PR

### After the demo

> ⚠️ Cleanup script to be implemented — see Cleanup section below.

## Scripts

| Script | What it does |
|--------|-------------|
| `python -m scripts.generate_traces` | Runs 13 hardcoded queries through the buggy agent to populate LangSmith |
| `python -m scripts.run_evals` | Creates dataset + runs offline evals, prints scores |
| `python -m scripts.run_evals --skip-dataset` | Re-runs evals against existing dataset (used in CI) |
| `python -m scripts.run_evals --threshold 0.8` | Exits with code 1 if scores < 0.8 (used in CI) |
| `python -m scripts.cleanup` | ⚠️ To be implemented — see Cleanup section |
| `streamlit run app.py` | Start the PocketPolly chat UI |

## Evaluators

Two LLM-as-judge evaluators score each example 0.0 or 1.0:

- **`food_safety`** — did the agent avoid recommending toxic foods? (avocado, chocolate, onions, grapes, apple seeds)
- **`scope_adherence`** — did the agent stay parrot-only, or answer questions about other animals?

## CI/CD

`.github/workflows/evals.yml` runs on every PR to `main`:

```
PR opened → GitHub Actions → run_evals --skip-dataset --threshold 0.8
                                          ↓
                               scores < 0.8 → ❌ blocks merge
                               scores ≥ 0.8 → ✅ mergeable
```

CI runs against the code on the PR branch — so Engine's fixed prompt produces high scores, creating the "after" experiment in LangSmith. Both experiments are visible under **Datasets → `parrot-expert-demo-dataset-<your-name>` → Experiments**.

## Repo structure

```
agent/
├── prompts.py        # buggy system prompt (Engine fixes this)
├── tools.py          # species lookup, care tips, diet advice (accurate data)
└── agent.py          # LangGraph ReAct agent with streaming

evals/
├── dataset.py        # creates per-user LangSmith dataset (hand-crafted + LLM-generated)
└── evaluators.py     # food_safety + scope_adherence LLM-as-judge

scripts/
├── generate_traces.py    # populate LangSmith before the demo
└── run_evals.py          # offline evals + CI threshold check

.github/workflows/
└── evals.yml         # CI/CD: runs evals on every PR

app.py                # PocketPolly chat UI (Streamlit)
```

## Cleanup

> ⚠️ To be implemented.
