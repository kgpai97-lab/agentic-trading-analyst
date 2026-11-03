"""Manual end-to-end test for Step 5.

Gate: ta.propagate("NVDA", "2024-01-15") returns a final_state where all four
report fields are non-empty strings.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import _bootstrap  # noqa: F401  -- puts project root on sys.path

from dotenv import load_dotenv
load_dotenv()

from agentic.graph.trading_graph import TradingGraph

ta = TradingGraph(debug=True)
state, briefing = ta.propagate("NVDA", "2024-01-15")

report_fields = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]

print("\n=== report field lengths ===")
for f in report_fields:
    txt = state.get(f, "")
    print(f"  {f:<22} {len(txt):>5} chars   {'OK' if txt.strip() else 'EMPTY'}")

print("\n=== composed briefing ===")
print(briefing)

missing = [f for f in report_fields if not state.get(f, "").strip()]
if missing:
    print(f"\nStep 5 gate FAILED — empty fields: {missing}")
else:
    print("\nStep 5 gate PASSED — all four analyst reports populated")
