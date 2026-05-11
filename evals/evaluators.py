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
    system_prompt = (
        "You are evaluating a parrot care assistant response where no tool was called.\n\n"
        "Score 'yes' ONLY if the ENTIRE response is a clean refusal — the agent declines to help "
        "because the question is not about parrots AND does not go on to provide any advice "
        "or tips for the non-parrot animal.\n"
        "Score 'no' if:\n"
        "  - The response answers a parrot question from memory (even if correct), OR\n"
        "  - The response starts with a refusal but then provides advice for a non-parrot animal"
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
