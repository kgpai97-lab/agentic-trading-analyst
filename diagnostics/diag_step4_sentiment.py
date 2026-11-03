import _bootstrap  # noqa: F401  -- puts project root on sys.path

from dotenv import load_dotenv
load_dotenv()

from agentic.graph.trading_graph import TradingGraph

ta = TradingGraph(debug=True)
state, report = ta.propagate("NVDA", "2024-01-15")

print("\n=== messages trace ===")
for m in state["messages"]:
    kind = type(m).__name__
    tc = getattr(m, "tool_calls", None)
    if tc:
        names = [t["name"] for t in tc]
        print(f"  {kind}: tool_calls={names}")
    else:
        print(f"  {kind}: {repr(m.content[:80])}")

print("\n=== sentiment_report ===")
print(report)
print()
print("Step 4 gate PASSED" if report.strip() else "Step 4 gate FAILED — report is empty")
