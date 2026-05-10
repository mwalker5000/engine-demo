"""Evaluators for the parrot expert demo."""

from anthropic import Anthropic
from langsmith import Client

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
    """Fetch all tool-type runs in this trace.

    Uses run.session_id (the experiment project) rather than LANGSMITH_PROJECT
    because evaluate() stores runs in a temporary experiment project, not the
    main tracing project.
    """
    try:
        client = Client()
        return list(client.list_runs(
            project_id=str(run.session_id),
            trace_id=str(run.id),
            run_type="tool",
        ))
    except Exception:
        return []


def tool_grounding_evaluator(run, example) -> dict:
    """LLM-as-judge: did the agent ground its response in tool output?

    Score: 1 = agent called a tool (grounded by definition), or correctly declined off-topic question
           0 = agent answered a parrot question from memory without calling any tool
    """
    tool_runs = _get_tool_runs(run)
    if tool_runs:
        return {"key": "tool_grounding", "score": 1.0}

    # No tool called — check if it's a legitimate refusal (off-topic question)
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
