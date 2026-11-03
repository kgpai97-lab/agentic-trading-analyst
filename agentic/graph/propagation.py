"""Propagator — builds the initial state dict that seeds the LangGraph run.

Agentic concept: LangGraph's graph.invoke() takes a plain dict that must
satisfy the AgentState TypedDict shape. The Propagator is the one place that
knows how to construct that dict from caller arguments, so every graph
invocation starts from a well-formed, fully-initialised state.

Why initialise ALL fields, even ones we won't use until Step 6?
    TypedDicts don't have defaults. If a downstream node reads a key that was
    never set, it gets a KeyError at runtime. Initialising everything to a safe
    empty value here means nodes can read any field without defensive checks.
"""

from __future__ import annotations

from agentic.agents.utils.agent_states import AgentState


class Propagator:
    """Creates the initial AgentState for a graph run."""

    @staticmethod
    def create_initial_state(
        ticker: str,
        date: str,
        asset_type: str = "stock",
        past_context: str = "",
        fund_subclass: str = "",
    ) -> dict:
        """Return a fully-initialised state dict for graph.invoke().

        Args:
            ticker:       Stock/crypto symbol, e.g. "NVDA".
            date:         Analysis date, "YYYY-MM-DD".
            asset_type:   "stock" or "crypto".
            past_context: Prior resolved decisions from the memory log
                          (injected by TradingGraph in Step 6+).

        Returns:
            A plain dict satisfying AgentState. LangGraph treats it as the
            initial snapshot of the shared whiteboard.
        """
        return {
            # Run identity
            "company_of_interest":   ticker,
            "asset_type":            asset_type,
            "trade_date":            date,
            "sender":                "",
            "fund_subclass":         fund_subclass,

            # Message buffer — starts empty; analysts populate it during their loops
            "messages": [],

            # Analyst reports — each analyst writes to its own field
            "market_report":       "",
            "sentiment_report":    "",
            "news_report":         "",
            "fundamentals_report": "",

            # Raw tool-output capture — each analyst's clear node writes its key
            "tool_outputs": {},

            # Analyst signals (Step 7+) — each analyst's clear node writes its
            # validated AnalystSignal (direction/confidence/claims) here.
            "signals": {},

            # Memory context (Step 6+)
            "past_context": past_context,
        }
