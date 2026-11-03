"""Portfolio-aware agents for the Step 6 PortfolioGraph.

Unlike the Step 5 analysts, these agents do NOT call tools. Each one is a
single llm.invoke() that consumes prior reports and emits a structured
output. They are wired up directly inside PortfolioGraph rather than via
make_analyst_node() (which is specifically for tool-loop nodes).
"""
