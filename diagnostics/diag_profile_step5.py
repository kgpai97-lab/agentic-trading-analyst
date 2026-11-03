r"""Profile a single propagate() call — wall clock + per-analyst usage.

Used to answer ask #6 (How long? How many tokens? How many tool calls?) and
to extrapolate Step 7 sizing (e.g. 100 tickers × profile).

Usage:
    .\venv\Scripts\python.exe diagnostics\diag_profile_step5.py
    .\venv\Scripts\python.exe diagnostics\diag_profile_step5.py NVDA 2024-01-15

Prints:
  • Wall-clock elapsed time for propagate()
  • Per-analyst bucket table (in_tokens, out_tokens, llm_calls, tool_calls, seconds)
  • Sanity reminders if Ollama did not populate token counts.
"""

from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import _bootstrap  # noqa: F401  -- puts project root on sys.path

from dotenv import load_dotenv
load_dotenv()

from agentic.graph.trading_graph import TradingGraph

DEFAULT_TICKER = "NVDA"
DEFAULT_DATE   = "2024-01-15"


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKER
    date   = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATE

    print(f"Profiling propagate({ticker!r}, {date!r}) ...")

    # debug=False keeps the LangGraph stream chatter out of the way.
    # We print self.usage.report() ourselves below.
    ta = TradingGraph(debug=False)

    t0 = time.perf_counter()
    state, briefing = ta.propagate(ticker, date)
    elapsed = time.perf_counter() - t0

    print(f"\nWall clock: {elapsed:.2f}s")
    print(ta.usage.report())

    report_fields = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
    print("\n=== report field lengths ===")
    for f in report_fields:
        txt = state.get(f, "")
        print(f"  {f:<22} {len(txt):>5} chars   {'OK' if txt.strip() else 'EMPTY'}")


if __name__ == "__main__":
    main()
