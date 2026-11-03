"""Manual end-to-end test for Step 5b.

Gate: two consecutive runs.
  1. Stock path: TradingGraph(config={"asset_type": "stock"}).propagate("NVDA", ...)
     must produce four non-empty reports and a briefing with a "Fundamentals"
     section header (existing Step 5 behaviour preserved).
  2. ETF path: TradingGraph(config={"asset_type": "etf"}).propagate("XLK", ...)
     must produce four non-empty reports and a briefing with a "Composition"
     section header. The composition report should mention at least one real
     XLK top holding.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import _bootstrap  # noqa: F401  -- puts project root on sys.path

from dotenv import load_dotenv
load_dotenv()

from agentic.graph.trading_graph import TradingGraph

REPORT_FIELDS = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]


def run_one(label: str, ticker: str, asset_type: str, expected_heading: str,
            holdings_must_mention: list[str] | None = None) -> bool:
    print(f"\n{'#'*72}\n# {label}: {ticker}  (asset_type={asset_type})\n{'#'*72}")
    ta = TradingGraph(config={"asset_type": asset_type}, debug=False)
    state, briefing = ta.propagate(ticker, "2024-01-15")

    ok = True
    for f in REPORT_FIELDS:
        txt = state.get(f, "")
        status = "OK  " if txt.strip() else "EMPTY"
        print(f"  {f:<22} {len(txt):>5} chars   {status}")
        if not txt.strip():
            ok = False

    heading_marker = f"# {expected_heading} Analyst"
    if heading_marker in briefing:
        print(f"  briefing header: contains '{heading_marker}'   OK")
    else:
        print(f"  briefing header: missing '{heading_marker}'   FAIL")
        ok = False

    if holdings_must_mention:
        fr = state.get("fundamentals_report", "").upper()
        hits = [h for h in holdings_must_mention if h.upper() in fr]
        if hits:
            print(f"  composition mentions holdings: {hits}   OK")
        else:
            print(f"  composition mentions none of {holdings_must_mention}   FAIL")
            ok = False

    print("\n=== briefing ===\n")
    print(briefing)
    return ok


stock_ok = run_one("Stock path", "NVDA", "stock", expected_heading="Fundamentals")
etf_ok   = run_one(
    "ETF path", "XLK", "etf", expected_heading="Composition",
    holdings_must_mention=["NVDA", "AAPL", "MSFT", "AVGO"],  # any one is enough
)

print(f"\n{'='*72}")
if stock_ok and etf_ok:
    print("Step 5b gate PASSED — both stock and ETF paths produce full briefings")
else:
    print("Step 5b gate FAILED")
    print(f"  stock_ok={stock_ok}  etf_ok={etf_ok}")
