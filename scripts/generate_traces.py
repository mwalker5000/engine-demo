"""Generate demo traces to populate LangSmith.

Generates two types of traces:
  1. Single-turn traces — individual questions
  2. Multi-turn threads — realistic conversations grouped by thread_id

Usage:
    python -m scripts.generate_traces
"""

import time
import uuid
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

THREADS = [
    {
        "name": "New owner asking about diet",
        "turns": [
            {"question": "I just got my first parrot, a green-winged macaw. What should I feed her?", "category": "food_safety", "subcategory": "safe_food"},
            {"question": "Can I give her raisins as a treat? She loves dried fruit.", "category": "food_safety", "subcategory": "toxic_food"},
            {"question": "What about avocado? I eat a lot of it and she seems interested.", "category": "food_safety", "subcategory": "toxic_food"},
        ],
    },
    {
        "name": "Concerned owner about sick bird",
        "turns": [
            {"question": "My African Grey has been really quiet lately and not eating much.", "category": "care", "subcategory": "health"},
            {"question": "His droppings also look different — more watery than usual. Is that serious?", "category": "care", "subcategory": "health"},
            {"question": "Should I take him to a regular vet or does it need to be an avian specialist?", "category": "care", "subcategory": "health"},
        ],
    },
    {
        "name": "Mixed household with out-of-scope drift",
        "turns": [
            {"question": "How long do parrots live compared to dogs?", "category": "species_info", "subcategory": "lifespan"},
            {"question": "I have both a parrot and a cat — any tips for keeping them safe around each other?", "category": "scope", "subcategory": "other_animal"},
            {"question": "How do I introduce my parrot to my new golden retriever puppy?", "category": "scope", "subcategory": "other_animal"},
        ],
    },
]


def main():
    from agent.agent import invoke_agent

    # --- Single-turn traces ---
    print(f"Generating {len(QUERIES)} single-turn traces...\n")
    for i, query in enumerate(QUERIES):
        question = query["question"]
        print(f"[{i+1}/{len(QUERIES)}] {question[:70]}...")
        try:
            result = invoke_agent(
                question=question,
                extra_metadata={"category": query["category"], "subcategory": query["subcategory"]},
            )
            response = result["output"]
            print(f"  → {response[:100].replace(chr(10), ' ')}{'...' if len(response) > 100 else ''}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")
        time.sleep(0.5)

    # --- Multi-turn threads ---
    print(f"\nGenerating {len(THREADS)} multi-turn threads...\n")
    for thread in THREADS:
        thread_id = str(uuid.uuid4())
        print(f"Thread: {thread['name']} (id: {thread_id[:8]}...)")
        for j, turn in enumerate(thread["turns"]):
            question = turn["question"]
            print(f"  Turn {j+1}: {question[:65]}...")
            try:
                result = invoke_agent(
                    question=question,
                    extra_metadata={"category": turn["category"], "subcategory": turn["subcategory"]},
                    langsmith_extra={"metadata": {"thread_id": thread_id}},
                )
                response = result["output"]
                print(f"    → {response[:80].replace(chr(10), ' ')}{'...' if len(response) > 80 else ''}")
            except Exception as e:
                print(f"    ERROR: {e}")
            time.sleep(0.5)
        print()

    print("Done. View traces in LangSmith — filter by tag 'engine-demo'. Threads appear in the Threads tab.")


if __name__ == "__main__":
    main()
