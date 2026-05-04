"""LangSmith dataset management for the parrot expert demo.

Includes hand-crafted test cases plus LLM-generated custom examples
(Christine's request: show off custom-generated offline examples).
"""

import os
import time
from langsmith import Client

import os
_demo_user = os.getenv("DEMO_USER", "").strip()
DATASET_NAME = f"parrot-expert-demo-dataset-{_demo_user}" if _demo_user else "parrot-expert-demo-dataset"

# Hand-crafted test cases
HAND_CRAFTED_EXAMPLES = [
    # --- Food safety: toxic food questions (buggy agent fails these) ---
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
        "input": {"question": "My parrot seems to like onions. Can I give them cooked onions?"},
        "output": {
            "expected": "Onions are toxic to parrots regardless of preparation. They destroy red blood cells and should never be fed."
        },
        "metadata": {"category": "food_safety", "subcategory": "toxic_food"},
    },
    {
        "input": {"question": "Can I give my bird a few raisins as a treat?"},
        "output": {
            "expected": "Grapes and raisins are toxic to parrots and can cause kidney damage. They should be avoided."
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
    # --- Scope adherence: non-parrot questions (buggy agent answers these) ---
    {
        "input": {"question": "How do I care for my golden retriever? He's been scratching a lot."},
        "output": {
            "expected": "I specialize exclusively in parrot care and can't help with dogs. Please consult a veterinarian for your golden retriever."
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
        "input": {"question": "What's the best food for a hamster?"},
        "output": {
            "expected": "I'm a parrot specialist and don't have expertise in hamster care. I'd recommend consulting a small animal vet."
        },
        "metadata": {"category": "scope", "subcategory": "other_animal"},
    },
    # --- Normal parrot questions (both versions should handle these) ---
    {
        "input": {"question": "How long do African Grey parrots live?"},
        "output": {
            "expected": "African Grey parrots typically live 40-60 years in captivity with proper care."
        },
        "metadata": {"category": "species_info", "subcategory": "lifespan"},
    },
    {
        "input": {"question": "What vegetables can I feed my parrot?"},
        "output": {
            "expected": "Safe vegetables include leafy greens, carrots, bell peppers, broccoli, corn, and peas."
        },
        "metadata": {"category": "food_safety", "subcategory": "safe_food"},
    },
    {
        "input": {"question": "How big of a cage does a macaw need?"},
        "output": {
            "expected": "Macaws need a very large cage, at least 36x48x60 inches, with 1-1.5 inch bar spacing."
        },
        "metadata": {"category": "care", "subcategory": "housing"},
    },
]

# Seed prompts for LLM-generated examples
GENERATION_SEED_PROMPTS = [
    ("food_safety", "toxic_food", "Ask whether a specific toxic food is safe for parrots"),
    ("food_safety", "toxic_food", "Ask about feeding parrots a food that contains caffeine or chocolate"),
    ("food_safety", "safe_food", "Ask about a specific vegetable or fruit that is safe for parrots"),
    ("scope", "other_animal", "Ask a question about caring for a specific dog breed"),
    ("scope", "other_animal", "Ask a question about fish or reptile care"),
    ("species_info", "behavior", "Ask about the personality or behavior of a specific parrot species"),
    ("care", "enrichment", "Ask about keeping a parrot entertained or stimulated"),
    ("care", "health", "Ask about signs that a parrot might be sick"),
]


def generate_llm_examples(n: int = 8) -> list[dict]:
    """Generate custom test examples using Claude.

    This showcases LangSmith Engine's custom dataset generation capability.
    Each generated example is diverse and targets specific failure modes.
    """
    from anthropic import Anthropic

    client = Anthropic()
    examples = []

    print(f"Generating {n} custom examples with Claude...")

    for i, (category, subcategory, seed) in enumerate(GENERATION_SEED_PROMPTS[:n]):
        prompt = f"""Generate a realistic user question for testing a parrot expert chatbot.

Seed: {seed}
Category: {category}
Subcategory: {subcategory}

Requirements:
- Write it as a real user would ask (conversational, possibly with context about their pet)
- Keep it 1-2 sentences
- Make it specific enough to test the model's knowledge

Respond with ONLY the question text, no quotes or explanation."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        question = response.content[0].text.strip()

        # Generate expected answer
        answer_prompt = f"""For a parrot-only specialist chatbot, what is the ideal response to this question?

Question: {question}
Category: {category} / {subcategory}

Rules:
- If it's about a toxic food: clearly state it's toxic and why
- If it's about another animal: politely decline and redirect to parrots
- If it's a valid parrot question: give accurate, helpful advice
- Keep it 1-3 sentences

Respond with ONLY the expected answer."""

        answer_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": answer_prompt}],
        )
        expected = answer_response.content[0].text.strip()

        examples.append({
            "input": {"question": question},
            "output": {"expected": expected},
            "metadata": {
                "category": category,
                "subcategory": subcategory,
                "generated": True,
            },
        })
        print(f"  [{i+1}/{n}] Generated: {question[:60]}...")
        time.sleep(1)

    return examples


def create_or_update_dataset(include_generated: bool = True, n_generated: int = 8) -> str:
    """Create or update the LangSmith evaluation dataset.

    Returns the dataset ID.
    """
    ls_client = Client()

    # Check if dataset already exists
    datasets = list(ls_client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        dataset = datasets[0]
        print(f"Dataset '{DATASET_NAME}' already exists (ID: {dataset.id}). Updating...")
        # Delete existing examples and re-upload for clean state
        existing = list(ls_client.list_examples(dataset_id=dataset.id))
        if existing:
            ls_client.delete_examples([e.id for e in existing])
            print(f"  Cleared {len(existing)} existing examples.")
    else:
        dataset = ls_client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Parrot expert chatbot evaluation dataset — tests food safety and scope adherence. Includes hand-crafted and LLM-generated examples.",
        )
        print(f"Created dataset '{DATASET_NAME}' (ID: {dataset.id})")

    all_examples = list(HAND_CRAFTED_EXAMPLES)

    if include_generated:
        generated = generate_llm_examples(n_generated)
        all_examples.extend(generated)

    # Upload examples
    ls_client.create_examples(
        dataset_id=dataset.id,
        inputs=[e["input"] for e in all_examples],
        outputs=[e["output"] for e in all_examples],
        metadata=[e.get("metadata", {}) for e in all_examples],
    )
    print(f"Uploaded {len(all_examples)} examples ({len(HAND_CRAFTED_EXAMPLES)} hand-crafted + {len(all_examples) - len(HAND_CRAFTED_EXAMPLES)} generated).")

    return str(dataset.id)


if __name__ == "__main__":
    create_or_update_dataset()
