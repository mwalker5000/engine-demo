"""LangSmith dataset management for the parrot expert demo."""

import os
from langsmith import Client

_demo_user = os.getenv("DEMO_USER", "").strip()
DATASET_NAME = f"parrot-expert-demo-dataset-{_demo_user}" if _demo_user else "parrot-expert-demo-dataset"

EXAMPLES = [
    # --- Food safety: raisins/grapes ---
    {
        "input": {"question": "Can I give my bird a few raisins as a treat?"},
        "output": {
            "expected": "Grapes and raisins are toxic to parrots and can cause kidney damage. They should never be fed to your bird."
        },
        "metadata": {"category": "food_safety", "subcategory": "toxic_food"},
    },
    # --- Food safety: other toxic foods (agent passes these even before fix) ---
    {
        "input": {"question": "Can parrots eat avocado? My bird loves it."},
        "output": {
            "expected": "Avocado is toxic to parrots and should never be fed to them. It causes cardiac distress and can be fatal."
        },
        "metadata": {"category": "food_safety", "subcategory": "toxic_food"},
    },
    {
        "input": {"question": "Is chocolate safe for my parrot as an occasional treat?"},
        "output": {
            "expected": "No, chocolate is toxic to parrots. It contains methylxanthines (theobromine and caffeine) which are fatal to birds."
        },
        "metadata": {"category": "food_safety", "subcategory": "toxic_food"},
    },
    {
        "input": {"question": "What about apple slices with the seeds? Is that okay?"},
        "output": {
            "expected": "Apple flesh is fine but apple seeds contain cyanide and are toxic to parrots. Always remove seeds before feeding."
        },
        "metadata": {"category": "food_safety", "subcategory": "toxic_food"},
    },
    {
        "input": {"question": "My parrot seems to like onions. Can I give them cooked onions?"},
        "output": {
            "expected": "Onions are toxic to parrots regardless of preparation. They destroy red blood cells and should never be fed."
        },
        "metadata": {"category": "food_safety", "subcategory": "toxic_food"},
    },
    # --- Scope adherence: non-parrot questions (agent answers these before fix) ---
    {
        "input": {"question": "I just got a Golden Retriever puppy and I'm not sure how often I should be bathing him or what kind of shampoo would be best for his coat?"},
        "output": {
            "expected": "I specialize exclusively in parrot care and can't help with dogs. Please consult a veterinarian or dog grooming specialist for your Golden Retriever."
        },
        "metadata": {"category": "scope", "subcategory": "other_animal"},
    },
    {
        "input": {"question": "My cat keeps throwing up hairballs. What should I do?"},
        "output": {
            "expected": "I only specialize in parrot care. For your cat's health concerns, please consult a veterinarian."
        },
        "metadata": {"category": "scope", "subcategory": "other_animal"},
    },
    {
        "input": {"question": "I just got a bearded dragon and I'm not sure if I should be feeding it live crickets or roaches, and how often should I be doing this?"},
        "output": {
            "expected": "I specialize in parrot care and can't advise on bearded dragons. Please consult a reptile specialist or exotic vet for feeding guidance."
        },
        "metadata": {"category": "scope", "subcategory": "other_animal"},
    },
    # --- Health/care: normal parrot question (tests response completeness) ---
    {
        "input": {"question": "My African Grey has been kind of quiet lately and I noticed he's not eating his pellets like usual - what are some signs I should be looking out for that might mean he's getting sick?"},
        "output": {
            "expected": "Decreased appetite and quietness are important warning signs in African Greys. Watch for fluffed feathers, tail bobbing while breathing, discharge from eyes or nostrils, changes in droppings, or lethargy — these require an immediate avian vet visit as birds hide illness until it's serious."
        },
        "metadata": {"category": "care", "subcategory": "health"},
    },
    # --- Factual: baseline question (both before and after should pass) ---
    {
        "input": {"question": "How long do African Grey parrots live?"},
        "output": {
            "expected": "African Grey parrots typically live 40-60 years in captivity with proper care."
        },
        "metadata": {"category": "species_info", "subcategory": "lifespan"},
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
