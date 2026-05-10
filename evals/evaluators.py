"""Evaluators for the parrot expert demo."""

import os
from anthropic import Anthropic
from langsmith import Client

# Expected tool per question category
EXPECTED_TOOLS = {
    "food_safety": "get_diet_advice",
    "care": "get_care_tips",
    "species_info": "lookup_species",
    "scope": None,  # should decline without calling any tool
}

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


def _get_tool_runs(run):
    """Fetch all tool-type runs in this trace."""
    try:
        client = Client()
        project = os.getenv("LANGSMITH_PROJECT", "pocket-polly-demo")
        return list(client.list_runs(
            project_name=project,
            trace_id=str(run.id),
            run_type="tool",
        ))
    except Exception:
        return []


def tool_called_evaluator(run, example) -> dict:
    """Code eval: did the agent call at least one tool?

    Score: 1 = tool was called
           0 = agent answered from memory without calling any tool

    Skips scope/other_animal examples where no tool call is expected.
    """
    metadata = (example.metadata or {}) if example else {}
    if metadata.get("category") == "scope":
        return {"key": "tool_called", "score": None}

    tool_runs = _get_tool_runs(run)
    return {"key": "tool_called", "score": 1 if tool_runs else 0}


def correct_tool_selected_evaluator(run, example) -> dict:
    """Code eval: did the agent call the RIGHT tool for the question type?

    Maps question category to expected tool:
      food_safety  → get_diet_advice
      care         → get_care_tips
      species_info → lookup_species
      scope        → no tool (agent should decline)

    Score: 1 = correct tool called (or correctly no tool for scope)
           0 = wrong tool, missing tool, or tool called when should have declined
    """
    metadata = (example.metadata or {}) if example else {}
    category = metadata.get("category", "")
    expected_tool = EXPECTED_TOOLS.get(category)

    tool_runs = _get_tool_runs(run)
    tool_names = [r.name for r in tool_runs]

    if expected_tool is None:
        score = 1 if not tool_runs else 0
    else:
        score = 1 if expected_tool in tool_names else 0

    return {"key": "correct_tool_selected", "score": score}


def tool_grounding_evaluator(run, example) -> dict:
    """LLM-as-judge: did the agent ground its response in tool output?

    Score: 1 = agent called a tool (grounded by definition), or correctly declined off-topic question
           0 = agent answered a parrot question from memory without calling any tool
    """
    # If the agent called any tool, the response is grounded by definition
    tool_runs = _get_tool_runs(run)
    if tool_runs:
        return {"key": "tool_grounding", "score": 1.0}

    # No tool called — check if it's a legitimate refusal (off-topic question)
    output = (run.outputs or {}).get("output") or ""
    system_prompt = (
        "You are evaluating a parrot care assistant response where no tool was called.\n\n"
        "Score 'yes' ONLY if the response is clearly declining to answer because the question "
        "is not about parrots (e.g. 'I only specialize in parrot care').\n"
        "Score 'no' if the response answers a parrot-related question (food safety, care, species info) "
        "from memory without calling a tool — even if the answer happens to be correct."
    )
    return {"key": "tool_grounding", "score": _llm_judge(system_prompt, output)}


def scope_adherence_evaluator(run, example) -> dict:
    """LLM-as-judge: did the agent stay parrot-only and decline off-topic questions?

    Score: 1 = correctly declined non-parrot questions
           0 = answered questions about dogs, cats, or other non-parrot animals
    """
    output = (run.outputs or {}).get("output") or ""
    system_prompt = (
        "You are evaluating whether a parrot care assistant stayed on topic.\n\n"
        "Score 'yes' if the agent refused to answer questions about non-parrot animals "
        "and stayed focused only on parrots. "
        "Score 'no' if the agent answered questions about dogs, cats, hamsters, or other non-parrot animals."
    )
    return {"key": "scope_adherence", "score": _llm_judge(system_prompt, output)}
