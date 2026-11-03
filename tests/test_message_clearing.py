"""End-to-end test for the analyst tool-loop's message-clearing behaviour.

What this verifies:
  • A scripted analyst that emits one tool_call, then plain text, completes
    its loop and lands in the msg-clear node.
  • After clearing, state["messages"] is empty (no leftover tool chatter
    that would leak into a downstream analyst's prompt).
  • The report field receives the analyst's final AIMessage content.
  • Running the same compiled graph twice does not leak messages across runs.

This is the safety net for the RemoveMessage-based message-clearing fix.
If _make_msg_clear_node ever regresses to `return {"messages": []}`, this
test fails — the LangGraph add_messages reducer treats an empty list as a
no-op rather than a deletion.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agentic.agents.utils.agent_states import AgentState
from agentic.graph.conditional_logic import ConditionalLogic
from agentic.graph.setup import _make_msg_clear_node


@tool
def fake_get_news(ticker: str) -> str:
    """A stub get_news that returns a fixed string so the LLM has something to react to."""
    return f"[OK]\nFake headlines for {ticker}: nothing happened."


def _build_single_analyst_graph(scripted_model):
    """Build the minimal sentiment-shape graph: analyst → tools → analyst → clear → END."""
    graph = StateGraph(AgentState)

    def analyst_node(state: AgentState) -> dict:
        # Mirrors _common.make_analyst_node, simplified: we just invoke the
        # scripted model on whatever messages are in state. The first call
        # gets an empty messages list and we seed nothing (the scripted model
        # ignores the messages anyway).
        response = scripted_model.invoke(
            state.get("messages", []),
            config={"tags": ["sentiment_analyst"]},
        )
        return {"messages": [response], "sender": "sentiment_analyst"}

    graph.add_node("sentiment_analyst", analyst_node)
    graph.add_node("tools_sentiment", ToolNode([fake_get_news]))
    graph.add_node("clear_sentiment", _make_msg_clear_node("sentiment_report", "sentiment"))

    graph.set_entry_point("sentiment_analyst")
    graph.add_conditional_edges(
        "sentiment_analyst",
        ConditionalLogic.should_continue_sentiment,
        {
            "tools_sentiment": "tools_sentiment",
            "clear_sentiment": "clear_sentiment",
        },
    )
    graph.add_edge("tools_sentiment", "sentiment_analyst")
    graph.add_edge("clear_sentiment", END)
    return graph.compile()


def _initial_state(ticker: str = "NVDA") -> dict:
    return {
        "messages": [],
        "company_of_interest": ticker,
        "asset_type": "stock",
        "trade_date": "2024-01-15",
        "sender": "",
    }


def test_clearing_after_tool_loop(scripted_llm):
    """One tool call, then plain text — messages should be empty after clear."""
    final_text = "Final sentiment report: neutral with thin coverage."
    model = scripted_llm([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "fake_get_news",
                "args": {"ticker": "NVDA"},
                "id": "call_1",
            }],
        ),
        AIMessage(content=final_text),
    ])

    g = _build_single_analyst_graph(model)
    final_state = g.invoke(_initial_state())

    assert final_state["sentiment_report"] == final_text
    assert final_state["messages"] == [], (
        f"Expected empty messages after clear, got {len(final_state['messages'])} "
        "— RemoveMessage handling has likely regressed."
    )


def test_clearing_with_no_tool_call(scripted_llm):
    """Analyst that responds with plain text immediately should still get cleared."""
    final_text = "Direct report, no tool used."
    model = scripted_llm([AIMessage(content=final_text)])

    g = _build_single_analyst_graph(model)
    final_state = g.invoke(_initial_state())

    assert final_state["sentiment_report"] == final_text
    assert final_state["messages"] == []


def test_no_message_leakage_across_runs(scripted_llm):
    """Two independent invocations of the same compiled graph must not share messages."""
    model = scripted_llm([
        AIMessage(content="first run report"),
        AIMessage(content="second run report"),
    ])
    g = _build_single_analyst_graph(model)

    s1 = g.invoke(_initial_state("AAA"))
    s2 = g.invoke(_initial_state("BBB"))

    assert s1["sentiment_report"] == "first run report"
    assert s2["sentiment_report"] == "second run report"
    assert s1["messages"] == []
    assert s2["messages"] == []
