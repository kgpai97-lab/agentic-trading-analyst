"""Shared state definitions — the "whiteboard" every agent reads and writes.

Agentic concept: in LangGraph, every node function receives the full state dict
and returns a *partial* update (only the keys it touched). LangGraph merges the
partial update back into the state automatically. This is like a shared
whiteboard: each analyst writes its report into a dedicated field, and later
nodes can read all prior reports without any explicit message passing.

Two special merge rules apply:
  - The `messages` key (inherited from MessagesState) uses LangGraph's
    add_messages reducer: new messages are *appended*, not overwritten.
    This supports the tool-loop pattern where a node may run many times.
  - All other keys are last-write-wins: returning {"sentiment_report": text}
    replaces the previous value of that key.

Quant analogy: AgentState is the shared DataFrame that every function can
read and append a new column to — but only MessagesState has special
concat semantics; the rest are simple assignments.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import MessagesState

from agentic.agents.utils.signals import AnalystSignal, _merge_signals


def _merge_tool_outputs(
    left: dict[str, str] | None,
    right: dict[str, str] | None,
) -> dict[str, str]:
    """Reducer for AgentState.tool_outputs: shallow-merge right into left.

    Each analyst's clear node returns {its_name: raw_output}. Without a merge
    reducer LangGraph would last-write-wins the whole dict, so only the final
    analyst's key would survive. Merging preserves every analyst's capture.
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged


# ---------------------------------------------------------------------------
# Primary shared state
# ---------------------------------------------------------------------------

class AgentState(MessagesState):
    """The whiteboard shared by every node in the LangGraph StateGraph.

    MessagesState contributes:
        messages: Annotated[list[BaseMessage], add_messages]
    which is the tool-loop message buffer. Each analyst appends tool calls
    and tool results here; a "Msg Clear" node wipes it before the next analyst.

    All other fields below are plain last-write-wins str or sub-TypedDict.
    """

    # --- Run identity --------------------------------------------------------
    company_of_interest: str   # ticker symbol, e.g. "NVDA" or "XLK"
    asset_type: str            # "stock" | "etf" | "income_alt" | "crypto"
    trade_date: str            # "YYYY-MM-DD" — the date we're analysing as of
    sender: str                # name of the last node that wrote to state
    fund_subclass: str         # income_alt only: "bond_fund" | "alternative_fund"

    # --- Analyst reports (one per analyst, written by Steps 3-5) -------------
    # Each report is prose markdown produced by the corresponding analyst node.
    # Later nodes read these to construct their prompts.
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str

    # --- Raw tool output capture (Step 6+) -----------------------------------
    # Each analyst's tool loop produces ToolMessages (raw OHLCV, indicators,
    # news, ETF holdings) that are normally wiped by the paired msg-clear node.
    # Before the wipe, the clear node copies the raw tool-result text here keyed
    # by analyst name, so the portfolio layer can forward the underlying data
    # (not just the analyst's prose) to the Head Analyst, Researcher, and PM.
    # Each analyst writes its own key exactly once; the _merge_tool_outputs
    # reducer keeps all keys (last-write-wins would drop earlier analysts').
    tool_outputs: Annotated[dict[str, str], _merge_tool_outputs]

    # --- Structured analyst signals (Step 7, Phase 1a) -----------------------
    # Each analyst's clear node writes a validated AnalystSignal (numeric
    # direction/confidence + falsifiable Claims) here, keyed by analyst name.
    # Produced by a cheap second extraction call over the prose report (see
    # agents/analysts/_common.py). The _merge_signals reducer keeps all keys
    # (last-write-wins would drop earlier analysts', exactly like tool_outputs).
    # Consumed by the Phase-5 claim checker, Phase-6 ensemble, and PM input.
    signals: Annotated[dict[str, AnalystSignal], _merge_signals]

    # --- Memory context (Step 6+) --------------------------------------------
    # Prior resolved decisions for this ticker, injected by the memory log.
    past_context: str


# ---------------------------------------------------------------------------
# Portfolio-aware state (Step 6)
#
# PortfolioGraph owns this. It is *not* a superset of AgentState — it lives
# in its own LangGraph run. Per-ticker AgentState instances are produced by
# TradingGraph.propagate() during the orchestration loop and folded into
# `per_ticker_reports` here as plain dicts.
# ---------------------------------------------------------------------------

class PortfolioState(TypedDict):
    # Inputs
    objective: str
    constraints: object            # PortfolioConstraints — numeric hard limits (Decision B)
    current_weights: dict[str, float]
    ticker_types: dict[str, str]   # ticker -> equity|equity_fund|bond_fund|alternative_fund|cash
    main_ticker: str               # "" when no candidate (pure rebalance)
    trade_date: str
    asset_type: str
    priority_threshold: float

    # Junior PA output
    priority_scores: dict[str, float]
    priority_rationale: str
    selected_tickers: list[str]

    # Per-ticker layer-1 + layer-2 output, keyed by ticker.
    # Shape: {"briefing": str, "head_assessment": str, "researcher_report": str}
    per_ticker_reports: dict[str, dict]

    # Portfolio Manager output (initial pass + uncertainty pass)
    pm_initial: dict
    uncertainty_reports: dict[str, str]
    final_decision: dict
