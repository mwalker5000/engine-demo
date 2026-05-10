"""Clean up LangSmith resources after the pocket-polly demo.

Resets the demo to a clean state so it can be run again without re-running setup:
  1. Removes Engine-added examples from the dataset (restores to 'baseline' tag)
  2. Removes 'after' experiments — keeps the oldest 'before' experiment
  3. Removes Engine-added online evaluators (keeps the 5 registered by setup.py)

Usage:
    python -m scripts.cleanup
"""

import json
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

_demo_user = os.getenv("DEMO_USER", "").strip()
DATASET_NAME = f"pocket-polly-demo-dataset-{_demo_user}" if _demo_user else "pocket-polly-demo-dataset"
PROJECT_NAME = os.getenv("LANGSMITH_PROJECT", "pocket-polly-demo")


# ── 1. Reset dataset ───────────────────────────────────────────────────────────

def reset_dataset() -> None:
    """Delete only Engine-added examples, keeping the original 3.

    setup.py tags the post-upload state as 'baseline'. This function reads
    example IDs from that tag and deletes anything added after it.
    """
    from langsmith import Client

    print(f"\n[1/3] Removing Engine-added examples from dataset '{DATASET_NAME}'...")
    ls_client = Client()

    try:
        baseline_ids = {e.id for e in ls_client.list_examples(dataset_name=DATASET_NAME, as_of="baseline")}
    except Exception as e:
        print(f"  Warning: could not read 'baseline' tag ({e}). Skipping dataset reset.")
        return

    all_examples = list(ls_client.list_examples(dataset_name=DATASET_NAME))
    to_delete = [e.id for e in all_examples if e.id not in baseline_ids]

    if to_delete:
        ls_client.delete_examples(to_delete)
        print(f"  Removed {len(to_delete)} Engine-added example(s). Original {len(baseline_ids)} kept.")
    else:
        print(f"  No Engine-added examples found. Dataset unchanged.")


# ── 2. Delete 'after' experiments ─────────────────────────────────────────────

def delete_ci_experiments() -> None:
    """Delete all CI-generated experiments, keeping only the one setup.py created.

    setup.py saves the experiment name it created to .demo_state.json. Every
    other experiment linked to the dataset (before- and after- from CI) is removed.
    """
    from langsmith import Client

    print(f"\n[2/3] Removing CI-generated experiments from dataset '{DATASET_NAME}'...")

    try:
        with open(".demo_state.json") as f:
            state = json.load(f)
        setup_experiment_name = state.get("setup_experiment_name")
    except FileNotFoundError:
        print("  Warning: .demo_state.json not found — run 'python -m scripts.setup' first.")
        print("  Skipping experiment cleanup.")
        return

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

    to_delete = [e for e in experiments if e.name != setup_experiment_name]

    if not to_delete:
        print("  No CI-generated experiments found.")
        return

    for exp in to_delete:
        ls_client.delete_project(project_name=exp.name)
        print(f"  Deleted '{exp.name}'")

    print(f"  Deleted {len(to_delete)} CI-generated experiment(s). Kept '{setup_experiment_name}'.")


# ── 3. Delete Engine-added online evaluators ───────────────────────────────────

def delete_engine_evaluators(api_key: str) -> None:
    """Delete only the online evaluators Engine added — leave our setup.py ones intact.

    setup.py saves the run rule IDs it created to .demo_state.json.
    Any run rule scoped to our project that is NOT in that list was added by
    Engine and is safe to remove. Rules belonging to other projects are ignored.
    """
    from langsmith import Client

    print(f"\n[3/3] Removing Engine-added online evaluators...")

    try:
        with open(".demo_state.json") as f:
            state = json.load(f)
        our_rule_ids = set(state.get("run_rule_ids", []))
    except FileNotFoundError:
        print("  Warning: .demo_state.json not found — run 'python -m scripts.setup' first.")
        print("  Skipping online evaluator cleanup.")
        return

    # Get our project ID so we only touch rules scoped to our project
    ls_client = Client()
    projects = list(ls_client.list_projects())
    project = next((p for p in projects if p.name == PROJECT_NAME), None)
    if not project:
        print(f"  Warning: project '{PROJECT_NAME}' not found. Skipping.")
        return
    project_id = str(project.id)

    resp = requests.get(
        "https://api.smith.langchain.com/api/v1/runs/rules",
        headers={"x-api-key": api_key},
    )
    if resp.status_code != 200:
        print(f"  Could not list run rules ({resp.status_code}). Skipping.")
        return

    deleted = 0
    for rule in resp.json():
        # Only consider rules tied to our project
        if rule.get("session_id") != project_id:
            continue
        if rule["id"] not in our_rule_ids:
            r = requests.delete(
                f"https://api.smith.langchain.com/api/v1/runs/rules/{rule['id']}",
                headers={"x-api-key": api_key},
            )
            if r.status_code in (200, 204):
                print(f"  Deleted Engine run rule '{rule.get('display_name', rule['id'])}'")
                deleted += 1

    if deleted == 0:
        print("  No Engine-added evaluators found.")
    else:
        print(f"  Deleted {deleted} Engine-added evaluator(s).")


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
    delete_ci_experiments()
    delete_engine_evaluators(api_key)

    print(f"\nCleanup complete. Ready for the next demo.")


if __name__ == "__main__":
    main()
