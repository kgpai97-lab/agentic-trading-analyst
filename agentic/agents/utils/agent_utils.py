"""Tool registration — the LangChain @tool wrappers agents actually call.

Agentic concept: LangChain's @tool decorator does two things:
    1. Wraps the function so LangGraph's ToolNode can execute it when the LLM
       emits a tool_call JSON blob naming it.
    2. Generates a JSON schema from the function's type hints and docstring,
       which gets sent to the LLM in the system prompt so it knows *how* to
       call the tool (argument names, types, and what it returns).

The docstring is not just documentation — it is the tool description the LLM
reads. Write it from the LLM's perspective: what does calling this tool give me?

The actual data-fetch logic lives in dataflows/ (dumb, testable, LLM-agnostic).
These wrappers are the thin adapter between the LangChain tool protocol and
the vendor routing layer.

TOOLS_* lists collect the tools for each analyst. They are passed to:
    - llm.bind_tools(TOOLS_<NAME>)  in the analyst factory (for generation)
    - ToolNode(TOOLS_<NAME>)        in setup.py            (for execution)
Both must reference the same list so the schema and the executor stay in sync.

Run-scoped context (as_of_date, lookback, etc.):
    Tool wrappers can't read graph state directly. For run metadata the LLM
    shouldn't have to pass, analyst nodes publish it into a run-scoped
    DataContext (dataflows/data_context.py) before invoking the LLM, and the
    wrappers read it back. As of Phase 2 that context is a `contextvars.
    ContextVar` (isolated per task — fixes the old concurrency-clobber
    limitation) and carries the `as_of_date` no-look-ahead upper bound, which
    the price tools pass through to the vendor layer.
"""

from __future__ import annotations

from typing import Union

from langchain_core.tools import tool

# Re-export the run-context API from its new home so existing imports
# (`from agentic.agents.utils.agent_utils import set_run_context`,
#  `current_fund_subclass`) keep working unchanged.
from agentic.dataflows.data_context import (  # noqa: F401
    current_as_of_date as _current_as_of_date,
    current_fund_subclass,
    current_lookback_days as _current_lookback_days,
    current_ticker as _current_ticker,
    current_universe as _current_universe,
    set_run_context,
)
from agentic.dataflows.interface import route_to_vendor


def _normalise_ticker(ticker: Union[str, dict]) -> str:
    """Coerce ticker to a plain string and snap near-misses to the known ticker.

    Two corrections applied in order:
    1. Dict unwrap — small models sometimes pass {"ticker": "NVDA"} instead of
       the bare string "NVDA".
    2. Typo snap — if the cleaned string differs from a KNOWN ticker by exactly
       1 edit (insertion/deletion/substitution), snap to it and warn. Catches
       gpt-oss hallucinations like "TQQ3" for "TQQQ". The known set is the
       current analyst's context ticker PLUS the run universe (the portfolio
       holdings), so a hallucinated cross-holding typo is caught even when the
       current analyst is studying a DIFFERENT name. Only fires when at least
       one known ticker is set.
    """
    if isinstance(ticker, dict):
        raw = str(ticker.get("ticker") or ticker.get("symbol") or next(iter(ticker.values()), ""))
    else:
        raw = str(ticker)

    # Build the known set: context ticker first (highest priority), then the
    # run universe. An exact match short-circuits (no correction).
    raw_u = raw.upper()
    known_set: list[str] = []
    ctx_ticker = _current_ticker()
    if ctx_ticker:
        known_set.append(ctx_ticker)
    known_set.extend(t for t in _current_universe() if t not in known_set)

    if raw_u in (k.upper() for k in known_set):
        return raw  # already a known ticker — nothing to snap
    for known in known_set:
        if _edit_distance(raw_u, known.upper()) <= 1:
            print(f"[tool] ticker typo corrected: {raw!r} -> {known!r}")
            return known
    return raw


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two short strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ch_a in a:
        curr = [prev[0] + 1]
        for j, ch_b in enumerate(b):
            curr.append(min(prev[j] + (0 if ch_a == ch_b else 1),
                            curr[-1] + 1,
                            prev[j + 1] + 1))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------

@tool
def get_news(ticker: Union[str, dict]) -> str:
    """Fetch recent news headlines for a stock, crypto, or fund ticker.

    Use this tool to retrieve current news before writing your report.
    Returns a numbered list of recent headlines with publication dates and
    sources. Call this once at the start of your analysis.

    For a FUND (ETF) ticker, the result also includes an "Underlying
    look-through" section with news for the index/asset the fund tracks
    (e.g. SPY → S&P 500, GLD → gold price, BND → US bond market) — because
    the wrapper itself has thin news. Treat that section as the news for what
    the fund actually holds.

    Args:
        ticker: The stock, crypto, or fund symbol, e.g. "NVDA", "AAPL",
            "BTC-USD", "SPY", "GLD".
    """
    return route_to_vendor(
        "get_news", _normalise_ticker(ticker), as_of_date=_current_as_of_date()
    ).to_llm_string()


@tool
def get_global_news(ticker: Union[str, dict]) -> str:
    """Fetch broad macro / market-wide news headlines, lightly contextualised by ticker.

    Use this in addition to get_news to capture market-wide drivers
    (Fed policy, inflation prints, geopolitics, sector flows) that affect
    the ticker even when no ticker-specific story is in the headlines.

    Args:
        ticker: The stock or crypto symbol. The sector context is derived from it.
    """
    return route_to_vendor(
        "get_global_news", _normalise_ticker(ticker), as_of_date=_current_as_of_date()
    ).to_llm_string()


@tool
def get_stock_data(ticker: Union[str, dict]) -> str:
    """Fetch recent OHLCV price history for a ticker.

    Returns a text table of the last ~60 trading days with Open, High, Low,
    Close, Volume and the percent change vs the previous close. Use this to
    describe price action, identify trends/reversals, and ground claims about
    "the stock rallied" or "broke down" in actual prices.

    Args:
        ticker: The stock or crypto symbol, e.g. "NVDA", "AAPL".
    """
    return route_to_vendor("get_stock_data", _normalise_ticker(ticker),
                           lookback_days=_current_lookback_days(),
                           as_of_date=_current_as_of_date()).to_llm_string()


@tool
def get_indicators(ticker: Union[str, dict]) -> str:
    """Fetch a set of technical indicators (SMA20/50/200, RSI14, MACD) for a ticker.

    Returns the most recent values of each indicator along with a one-line
    interpretation hint (e.g. "above 200-day SMA — long-term uptrend").
    Use this alongside get_stock_data: the OHLCV gives you the picture,
    the indicators give you the standard signals.

    Args:
        ticker: The stock or crypto symbol, e.g. "NVDA".
    """
    return route_to_vendor("get_indicators", _normalise_ticker(ticker),
                           lookback_days=_current_lookback_days(),
                           as_of_date=_current_as_of_date()).to_llm_string()


@tool
def get_fundamentals(ticker: Union[str, dict]) -> str:
    """Fetch key fundamental metrics for a ticker.

    Returns market cap, trailing/forward P/E, EPS, profit margins, revenue
    growth, return on equity, debt/equity, dividend yield, and 52-week range.
    Use this to characterise the company's valuation and financial health
    before writing the fundamentals report.

    Args:
        ticker: The stock symbol, e.g. "NVDA". Crypto generally returns sparse data.
    """
    return route_to_vendor("get_fundamentals", _normalise_ticker(ticker)).to_llm_string()


@tool
def get_income_statement(ticker: Union[str, dict]) -> str:
    """Fetch the most recent annual income-statement line items for a ticker.

    Returns revenue, gross profit, operating income, net income, and EPS for
    the latest available fiscal years. Use this when the fundamentals snapshot
    is not enough and you need to discuss revenue/earnings trajectory.

    Args:
        ticker: The stock symbol, e.g. "NVDA".
    """
    return route_to_vendor("get_income_statement", _normalise_ticker(ticker)).to_llm_string()


@tool
def get_etf_holdings(ticker: Union[str, dict]) -> str:
    """Fetch top holdings and sector breakdown for an ETF.

    Returns the fund's category and family, asset-class mix (stocks/bonds/cash),
    top 10 holdings with their weights, and non-zero sector weightings.
    Use this once at the start of an ETF composition analysis.

    Args:
        ticker: The ETF symbol, e.g. "XLK", "SPY", "QQQ".
    """
    return route_to_vendor("get_etf_holdings", _normalise_ticker(ticker)).to_llm_string()


# ---------------------------------------------------------------------------
# Tool sets — one list per analyst node.
# Both bind_tools() and ToolNode() must use the same list reference.
# ---------------------------------------------------------------------------

TOOLS_SENTIMENT:    list = [get_news]
TOOLS_MARKET:       list = [get_stock_data, get_indicators]
TOOLS_NEWS:         list = [get_news, get_global_news]
TOOLS_FUNDAMENTALS: list = [get_fundamentals, get_income_statement]

# Composition analyst — replaces fundamentals when asset_type == "etf".
# TODO: add `get_etf_profile` (AUM, expense ratio, inception date, avg volume)
# as a second tool here when the Profile Analyst lands.
TOOLS_COMPOSITION:  list = [get_etf_holdings]

# Fixed-income / alternatives analyst — fourth slot when asset_type ==
# "income_alt" (bond_fund or alternative_fund). Same single tool as
# composition; the report differs via prompt + fund_subclass run-context.
TOOLS_INCOME_ALT:   list = [get_etf_holdings]
