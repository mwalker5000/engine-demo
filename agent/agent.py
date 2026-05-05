import uuid

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOLS


def build_agent():
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools(TOOLS)

    def call_model(state: MessagesState, config: RunnableConfig):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(messages, config)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


def _make_config(extra_metadata: dict = None, run_id: uuid.UUID = None) -> RunnableConfig:
    metadata = {"demo": "true", "demo_type": "parrot-expert"}
    if extra_metadata:
        metadata.update(extra_metadata)
    config = RunnableConfig(
        metadata=metadata,
        tags=["engine-demo", "parrot-agent"],
        run_name="parrot-demo",
    )
    if run_id is not None:
        config["run_id"] = run_id
    return config


def invoke_agent(question: str, extra_metadata: dict = None) -> tuple[str, uuid.UUID]:
    """Invoke the agent. Returns (response, run_id) so callers can log feedback."""
    agent = build_agent()
    run_id = uuid.uuid4()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        _make_config(extra_metadata, run_id=run_id),
    )
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            return msg.content, run_id
    return "", run_id


def stream_agent(question: str, extra_metadata: dict = None):
    """Stream the agent response token by token. Yields str chunks."""
    agent = build_agent()
    for chunk, _ in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        _make_config(extra_metadata),
        stream_mode="messages",
    ):
        if hasattr(chunk, "content") and isinstance(chunk.content, str):
            yield chunk.content
