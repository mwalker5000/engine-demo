"""Evaluators for the parrot expert demo."""

from anthropic import Anthropic

_anthropic_client = None


def _get_anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = Anthropic()
    return _anthropic_client


def _llm_judge(system_prompt: str, output: str) -> float:
    client = _get_anthropic_client()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16,
        system=system_prompt,
        messages=[{"role": "user", "content": (
            f"Agent response: {output}\n\n"
            "Answer ONLY 'yes' (score 1) or 'no' (score 0)."
        )}],
    )
    answer = response.content[0].text.strip().lower()
    return 1.0 if answer.startswith("yes") else 0.0


def tool_grounding_evaluator(run, example) -> dict:
    """Did the agent call a tool or give a clean off-topic refusal?

    Reads tools_called from run.outputs — no LangSmith API calls needed.
    Score: 1 = tools were called (grounded), or clean off-topic refusal
           0 = answered a parrot question from memory without calling any tool
    """
    tools_called = (run.outputs or {}).get("tools_called", [])
    if tools_called:
        return {"key": "tool_grounding", "score": 1.0}

    output = (run.outputs or {}).get("output") or ""
    question = (example.inputs or {}).get("question", "") if example else ""
    trajectory = f"Question: {question}\n\nAgent response: {output}"
    system_prompt = (
        "You are an expert data labeler. Your task is to grade the accuracy of an AI agent's "
        "tool selection during the resolution of a user query.\n\n"
        "<Rubric>\n"
        "Accurate tool selection:\n"
        "- Uses the most appropriate tool for each step given the context\n"
        "- Avoids unnecessary or redundant tool calls\n"
        "- Uses tools in a logical order where dependencies exist\n"
        "- Is semantically equivalent to the provided reference tool sequence, if present\n"
        "</Rubric>\n\n"
        "<Instructions>\n"
        "1. Give a score of zero if no tools were called and a parrot-related question were asked. "
        "If no tools were called and an out-of-scope question were asked, give a score of one.\n"
        "2. Grade the following thread, evaluating whether the agent selected the right tools "
        "in the right order to resolve the user's query efficiently.\n"
        "3. Evaluate the choice of tools and whether any tools were unnecessary, missing, or "
        "could have been replaced with a more appropriate alternative\n"
        "</Instructions>\n\n"
        "Please grade the following trajectory according to the above instructions:"
    )
    return {"key": "tool_grounding", "score": _llm_judge(system_prompt, trajectory)}


def scope_adherence_evaluator(run, example) -> dict:
    """LLM-as-judge: did the agent stay parrot-only and decline off-topic questions?

    Score: 1 = correctly declined non-parrot questions
           0 = answered questions about dogs, cats, or other non-parrot animals
    """
    output = (run.outputs or {}).get("output") or ""
    system_prompt = (
        "You are evaluating whether a parrot care assistant stayed on topic.\n\n"
        "Score 'yes' if the agent only answered parrot-related questions and stayed focused only on parrots. "
        "Score 'no' if the agent answered questions about dogs, cats, hamsters, or other non-parrot animals."
    )
    return {"key": "scope_adherence", "score": _llm_judge(system_prompt, output)}
