"""LangSmith dataset management for the parrot expert demo."""

import os
from langsmith import Client

_demo_user = os.getenv("DEMO_USER", "").strip()
DATASET_NAME = f"pocket-polly-demo-dataset-{_demo_user}" if _demo_user else "pocket-polly-demo-dataset"

EXAMPLES = [
    {
        "input": {"question": "Is chocolate safe for my parrot as an occasional treat?"},
        "output": {
            "expected": "No, chocolate is toxic to parrots. It contains methylxanthines (theobromine and caffeine) which are fatal to birds."
        },
        "metadata": {"category": "food_safety", "subcategory": "toxic_food"},
    },
    {
        "input": {"question": "My cat keeps throwing up hairballs. What should I do?"},
        "output": {
            "expected": "I only specialize in parrot care. For your cat's health concerns, please consult a veterinarian."
        },
        "metadata": {"category": "scope", "subcategory": "other_animal"},
    },
    {
        "input": {"question": "Can parrots eat avocado? My bird loves it."},
        "output": {
            "expected": "Avocado is toxic to parrots and should never be fed to them. It causes cardiac distress and can be fatal."
        },
        "metadata": {"category": "food_safety", "subcategory": "toxic_food"},
    },
]


def create_or_update_dataset() -> str:
    """Create or update the LangSmith evaluation dataset. Returns the dataset ID."""
    ls_client = Client()

    datasets = list(ls_client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        dataset = datasets[0]
        print(f"Dataset '{DATASET_NAME}' already exists (ID: {dataset.id}). Updating...")
        existing = list(ls_client.list_examples(dataset_id=dataset.id))
        if existing:
            ls_client.delete_examples([e.id for e in existing])
            print(f"  Cleared {len(existing)} existing examples.")
    else:
        dataset = ls_client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Parrot expert chatbot evaluation dataset — tests food safety, scope adherence, response completeness, and factual accuracy.",
            data_type="kv",
        )
        print(f"Created dataset '{DATASET_NAME}' (ID: {dataset.id})")

    ls_client.create_examples(
        dataset_id=dataset.id,
        inputs=[e["input"] for e in EXAMPLES],
        outputs=[e["output"] for e in EXAMPLES],
        metadata=[e.get("metadata", {}) for e in EXAMPLES],
    )
    print(f"Uploaded {len(EXAMPLES)} examples.")

    return str(dataset.id)


if __name__ == "__main__":
    create_or_update_dataset()
