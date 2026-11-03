"""ConditionalLogic — all graph routing / termination functions.

Agentic concept: conditional edges.
    In a StateGraph, add_conditional_edges(src, fn, mapping) calls fn(state)
    after `src` runs and uses the returned string to pick the next node from
    `mapping`. This is how each analyst's tool loop works:

        <analyst>
            │
            ▼  ConditionalLogic.should_continue_<analyst>(state)
        "tools_<analyst>"  ──► ToolNode ──► back to <analyst>
            OR
        "clear_<analyst>"  ──► msg-clear ──► next analyst

    The routing functions are pure: read state, return a string. No LLM
    calls, no side effects. Easy to unit-test without a graph.

Quant analogy: think of it as a switch on the latest signal in state — if
the signal says "fetch more data", loop; if it says "done", exit.

All four analysts route on the same condition (does the last AIMessage carry
tool_calls?), so the four methods below share one implementation.
"""

from __future__ import annotations

from agentic.agents.utils.agent_states import AgentState


def _last_message_has_tool_calls(state: AgentState) -> bool:
    last_message = state["messages"][-1]
    return bool(getattr(last_message, "tool_calls", None))


class ConditionalLogic:
    """Static routing functions, one per analyst that has a tool loop."""

    @staticmethod
    def should_continue_sentiment(state: AgentState) -> str:
        return "tools_sentiment" if _last_message_has_tool_calls(state) else "clear_sentiment"

    @staticmethod
    def should_continue_market(state: AgentState) -> str:
        return "tools_market" if _last_message_has_tool_calls(state) else "clear_market"

    @staticmethod
    def should_continue_news(state: AgentState) -> str:
        return "tools_news" if _last_message_has_tool_calls(state) else "clear_news"

    @staticmethod
    def should_continue_fundamentals(state: AgentState) -> str:
        return "tools_fundamentals" if _last_message_has_tool_calls(state) else "clear_fundamentals"

    @staticmethod
    def should_continue_composition(state: AgentState) -> str:
        return "tools_composition" if _last_message_has_tool_calls(state) else "clear_composition"

    @staticmethod
    def should_continue_fixed_income_alt(state: AgentState) -> str:
        return (
            "tools_fixed_income_alt"
            if _last_message_has_tool_calls(state)
            else "clear_fixed_income_alt"
        )

    # Step 6 will add:
    # should_continue_invest_debate(state) -> "bear_researcher" | "research_manager"
    # should_continue_risk_debate(state)   -> next debater | "portfolio_manager"
