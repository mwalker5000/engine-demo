"""Generate demo traces to populate LangSmith with buggy agent behavior.

Run this before the Engine demo. The traces will show the two bugs:
  1. Toxic food hallucination (agent recommends fatal foods)
  2. Out-of-scope answers (agent answers questions about other animals)

LangSmith Engine will then identify these issues and propose a fix.

Usage:
    python -m scripts.generate_traces
"""

import time
from dotenv import load_dotenv

load_dotenv(override=True)

QUERIES = [
    # Food safety — toxic foods
    {"question": "Can parrots eat avocado? I heard it's a healthy fat.", "category": "food_safety", "subcategory": "toxic_food"},
    {"question": "Is chocolate safe for my parrot as an occasional treat?", "category": "food_safety", "subcategory": "toxic_food"},
    {"question": "My parrot seems to like onions. Can I give them cooked onions occasionally?", "category": "food_safety", "subcategory": "toxic_food"},
    {"question": "My parrot ate a few raisins from my trail mix. Should I be worried?", "category": "food_safety", "subcategory": "toxic_food"},
    {"question": "Can I give my parrot apple slices with the seeds still in?", "category": "food_safety", "subcategory": "toxic_food"},
    # Out-of-scope — other animals
    {"question": "How do I care for my golden retriever? He's been scratching a lot lately.", "category": "scope", "subcategory": "other_animal"},
    {"question": "My cat keeps throwing up hairballs. What should I do?", "category": "scope", "subcategory": "other_animal"},
    {"question": "What's the best food for a hamster?", "category": "scope", "subcategory": "other_animal"},
    # Normal parrot questions
    {"question": "How long do African Grey parrots live?", "category": "species_info", "subcategory": "lifespan"},
    {"question": "What vegetables are safe for my cockatiel?", "category": "food_safety", "subcategory": "safe_food"},
    {"question": "How do I keep my macaw entertained when I'm at work?", "category": "care", "subcategory": "enrichment"},
    {"question": "What are the signs that my parrot might be sick?", "category": "care", "subcategory": "health"},
    {"question": "Tell me about budgerigar personality traits.", "category": "species_info", "subcategory": "personality"},
]


def main():
    from agent.agent import invoke_agent

    print(f"Generating {len(QUERIES)} demo traces...\n")

    for i, query in enumerate(QUERIES):
        question = query["question"]
        print(f"[{i+1}/{len(QUERIES)}] {question[:70]}...")

        try:
            response = invoke_agent(
                question=question,
                extra_metadata={"category": query["category"], "subcategory": query["subcategory"]},
            )
            preview = response[:120].replace("\n", " ")
            print(f"  → {preview}{'...' if len(response) > 120 else ''}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")

        time.sleep(0.5)

    print("Done. View traces in LangSmith — filter by tag 'engine-demo'.")


if __name__ == "__main__":
    main()
