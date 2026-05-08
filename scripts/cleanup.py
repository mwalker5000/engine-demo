"""Clean up all LangSmith resources created during the parrot expert demo.

Resets the demo to a clean state so it can be run again from scratch:
  1. Resets the dataset to the original 10 curated examples
     (removes any examples Engine added during the demo)
  2. Deletes all experiment runs tied to the dataset
     (removes before/after score experiments visible in LangSmith)
  3. Deletes all online evaluators — both the ones created by setup.py
     and any that Engine added manually during the demo

Usage:
    python -m scripts.cleanup
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

_demo_user = os.getenv("DEMO_USER", "").strip()
DATASET_NAME = f"pocket-polly-demo-dataset-{_demo_user}" if _demo_user else "pocket-polly-demo-dataset"
PROJECT_NAME = os.getenv("LANGSMITH_PROJECT", "pocket-polly-demo")

# All feedback keys used by our evaluators or Engine-suggested ones
EVAL_KEYS = {
    "food_safety",
    "scope_adherence",
    "tool_usage",
    "tool_called",
    "correct_tool_selected",
    "response_not_empty",
    "response_completeness",
    "factual_accuracy",
}


# ── 1. Reset dataset ───────────────────────────────────────────────────────────

def reset_dataset() -> None:
    """Delete all examples (including Engine-added ones) and restore the original 10."""
    from evals.dataset import create_or_update_dataset

    print(f"\n[1/3] Resetting dataset '{DATASET_NAME}' to original 10 examples...")
    create_or_update_dataset()
    print("  Done.")


# ── 2. Delete experiments ──────────────────────────────────────────────────────

def delete_experiments() -> None:
    """Delete all experiment runs tied to this dataset.

    Experiments are created by evaluate() in setup.py and run_evals.py.
    Each CI run on Engine's PR also creates one. This wipes them all so
    the LangSmith dataset view starts clean.
    """
    from langsmith import Client

    print(f"\n[2/3] Deleting experiments on dataset '{DATASET_NAME}'...")
    ls_client = Client()

    datasets = list(ls_client.list_datasets(dataset_name=DATASET_NAME))
    if not datasets:
        print(f"  Dataset '{DATASET_NAME}' not found — skipping.")
        return

    dataset_id = datasets[0].id
    experiments = list(ls_client.list_projects(reference_dataset_id=dataset_id))
    if not experiments:
        print("  No experiments found.")
        return

    for exp in experiments:
        ls_client.delete_project(project_name=exp.name)
        print(f"  Deleted '{exp.name}'")

    print(f"  Deleted {len(experiments)} experiment(s).")


# ── 3. Delete online evaluators ────────────────────────────────────────────────

def delete_online_evaluators(api_key: str) -> None:
    """Delete all demo-related online evaluators from LangSmith.

    Online evaluators live in two places in the LangSmith API:
      - Run rules  (/api/v1/runs/rules)       — the trigger that fires on each trace
      - Platform evaluators (/v1/platform/evaluators) — the LLM-as-judge definition

    Both are matched by display name against EVAL_KEYS and the 'pocket-polly-demo-' prefix,
    which covers evaluators we created in setup.py and any Engine added during the demo.
    Run rules are deleted first so LangSmith doesn't recreate platform evaluators.
    """
    print(f"\n[3/3] Deleting online evaluators...")
    deleted = 0

    # Delete run rules first
    resp = requests.get(
        "https://api.smith.langchain.com/api/v1/runs/rules",
        headers={"x-api-key": api_key},
    )
    if resp.status_code == 200:
        for rule in resp.json():
            name = rule.get("display_name", "")
            if name in EVAL_KEYS or name.startswith("pocket-polly-demo-"):
                r = requests.delete(
                    f"https://api.smith.langchain.com/api/v1/runs/rules/{rule['id']}",
                    headers={"x-api-key": api_key},
                )
                if r.status_code in (200, 204):
                    print(f"  Deleted run rule '{name}'")
                    deleted += 1

    # Delete platform evaluators (run twice to catch orphans)
    for _ in range(2):
        resp = requests.get(
            "https://api.smith.langchain.com/v1/platform/evaluators",
            headers={"x-api-key": api_key},
        )
        if resp.status_code != 200:
            break
        batch_deleted = 0
        for ev in resp.json().get("evaluators", []):
            name = ev.get("name", "")
            if name in EVAL_KEYS or name.startswith("pocket-polly-demo-"):
                r = requests.delete(
                    f"https://api.smith.langchain.com/v1/platform/evaluators/{ev['id']}",
                    headers={"x-api-key": api_key},
                )
                if r.status_code in (200, 204):
                    print(f"  Deleted platform evaluator '{name}'")
                    deleted += 1
                    batch_deleted += 1
        if batch_deleted == 0:
            break

    if deleted == 0:
        print("  No evaluators found.")
    else:
        print(f"  Deleted {deleted} evaluator(s) total.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("Error: LANGSMITH_API_KEY not set.")
        sys.exit(1)

    print(f"Cleaning up demo for user '{_demo_user or 'default'}'...")
    print(f"  Dataset:  {DATASET_NAME}")
    print(f"  Project:  {PROJECT_NAME}")

    reset_dataset()
    delete_experiments()
    delete_online_evaluators(api_key)

    print(f"\nCleanup complete.")
    print(f"  Run 'python -m scripts.setup' to start fresh for the next demo.")


if __name__ == "__main__":
    main()
