"""Tool-result validation: bad tickers should produce ToolResult.ok == False.

These tests hit yfinance live for two reasons:
  1. They double as a smoke check that the network and yfinance package work.
  2. The validation layer's correctness depends on yfinance's actual error
     shape for unknown symbols, which has historically drifted.

If you need to run pytest fully offline, mark these with -k "not yfinance".
"""

from __future__ import annotations

import pytest

from agentic.dataflows.result import ToolResult
from agentic.dataflows.y_finance import (
    _verify_ticker,
    get_etf_holdings_yfinance,
    get_fundamentals_yfinance,
    get_income_statement_yfinance,
    get_indicators_yfinance,
    get_news_yfinance,
    get_stock_data_yfinance,
)


BAD = "ZZZZZZZZZZ"   # almost certainly not a real ticker
GOOD = "AAPL"        # large-cap, stable, present on yfinance


# ----------------------------------------------------------------------
# ToolResult unit tests — no network required
# ----------------------------------------------------------------------

def test_toolresult_success_tagging():
    r = ToolResult.success("hello world")
    assert r.ok is True
    assert r.to_llm_string().startswith("[OK]")
    assert "hello world" in r.to_llm_string()


def test_toolresult_empty_tagging():
    r = ToolResult.empty("nothing here")
    assert r.ok is False
    assert r.meta.get("empty") is True
    assert r.to_llm_string().startswith("[EMPTY]")
    assert "nothing here" in r.to_llm_string()


def test_toolresult_failure_tagging():
    r = ToolResult.failure("network broke")
    assert r.ok is False
    assert r.meta.get("empty") is not True
    assert r.to_llm_string().startswith("[ERROR]")
    assert "network broke" in r.to_llm_string()


def test_empty_ticker_string_rejected():
    bad = _verify_ticker("")
    assert isinstance(bad, ToolResult)
    assert bad.ok is False
    assert bad.meta.get("empty") is True


# ----------------------------------------------------------------------
# Live yfinance — marked so they can be skipped
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "fn",
    [
        get_news_yfinance,
        get_stock_data_yfinance,
        get_indicators_yfinance,
        get_fundamentals_yfinance,
        get_income_statement_yfinance,
        get_etf_holdings_yfinance,
    ],
    ids=["news", "stock_data", "indicators", "fundamentals", "income_stmt", "etf_holdings"],
)
def test_bad_ticker_returns_empty(fn):
    r = fn(BAD)
    assert isinstance(r, ToolResult)
    assert r.ok is False, f"{fn.__name__}({BAD!r}) unexpectedly returned ok=True"
    # Either [EMPTY] (caught by validator) or [ERROR] (caught downstream) is fine.
    assert "[" in r.to_llm_string()


def test_good_ticker_market_data_returns_ok():
    r = get_stock_data_yfinance(GOOD)
    assert isinstance(r, ToolResult)
    assert r.ok is True
    assert r.to_llm_string().startswith("[OK]")
