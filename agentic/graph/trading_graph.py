"""TradingGraph — the public entry point for the whole pipeline.

Usage:
    from agentic.graph.trading_graph import TradingGraph
    from agentic.default_config import get_config

    ta = TradingGraph(config=get_config())
    state, report = ta.propagate("NVDA", "2024-01-15")
    print(report)

The class owns three responsibilities:
  1. Build-time setup  — create LLMs, compile the graph (done in __init__)
  2. Run-time invoke   — create initial state, call graph.invoke() (propagate)
  3. Post-run extract  — pull the final signal out of state (Step 6: Buy/Hold/Sell)

Keeping setup out of propagate() means the graph is compiled once and reused
across multiple propagate() calls — important when running backtests.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agentic.agents.utils.agent_utils import set_run_context
from agentic.default_config import get_config
from agentic.graph.propagation import Propagator
from agentic.graph.setup import GraphSetup
from agentic.llm_clients.factory import create_llm_pair
from agentic.llm_clients.usage_callback import UsageCallback


class TradingGraph:
    """Orchestrates the multi-agent trading analysis pipeline.

    Args:
        config:  Dict of config overrides applied on top of DEFAULT_CONFIG.
                 Pass None to use defaults as-is.
        debug:   If True, LangGraph streams node-level events to stdout
                 so you can watch the graph execute step by step.
    """

    def __init__(
        self,
        config: dict | None = None,
        debug: bool = False,
    ) -> None:
        self.config = get_config(config)
        self.debug  = debug

        # Per-instance usage accounting. The same callback instance is shared
        # by both LLM tiers and is reset at the start of every propagate() so
        # buckets reflect a single run.
        self.usage = UsageCallback()

        # Create both LLM tiers once at startup.
        # deep_llm  → synthesis nodes (Step 6+: Research Manager, Portfolio Manager)
        # quick_llm → all analyst and debate nodes
        self.deep_llm, self.quick_llm = create_llm_pair(
            self.config, callbacks=[self.usage]
        )

        # Compile the StateGraph. Compilation validates edge/node wiring and
        # returns a Pregel-based runnable. Expensive once; free to invoke many times.
        self._graph = GraphSetup(
            quick_llm=self.quick_llm,
            deep_llm=self.deep_llm,
            config=self.config,
        ).build()

    def propagate(self, ticker: str, date: str, fund_subclass: str = "") -> tuple[dict, str]:
        """Run the full agent pipeline for a ticker on a given date.

        Args:
            ticker: Stock/crypto symbol, e.g. "NVDA".
            date:   Analysis date string, "YYYY-MM-DD".
            fund_subclass: For the income_alt fourth-slot analyst, the
                           four-way class ("bond_fund" | "alternative_fund").
                           Empty for stock/etf. Threaded into state so the
                           analyst's task instruction can branch.

        Returns:
            (final_state, signal) where:
              final_state — the complete AgentState dict after all nodes ran.
              signal      — the key output for the current build step:
                            Step 3:  sentiment_report only
                            Step 5:  all four analyst reports concatenated  ← current
                            Step 6+: consumed per-ticker by PortfolioGraph
        """
        asset_type   = self.config.get("asset_type", "stock")
        lookback_days = self.config.get("lookback_days", 30)
        initial_state = Propagator.create_initial_state(
            ticker, date, asset_type=asset_type, fund_subclass=fund_subclass
        )

        # Publish run metadata so tool wrappers can read it without threading
        # state all the way down into @tool functions. snapshot_dir rides the
        # context (the @tool wrappers call route_to_vendor without a config), so
        # the inner-graph fetches actually write provenance snapshots.
        set_run_context(
            trade_date=date,
            lookback_days=lookback_days,
            fund_subclass=fund_subclass,
            snapshot_dir=self.config.get("snapshot_dir", ""),
            universe=tuple(self.config.get("universe", ())),
        )

        # Fresh usage accounting per run.
        self.usage.reset()

        # graph.invoke() runs the graph to completion and returns the final state.
        # For debugging, graph.stream() yields one dict per node instead.
        if self.debug:
            final_state = self._run_with_stream(initial_state)
        else:
            final_state = self._graph.invoke(initial_state)

        signal = self._compose_briefing(final_state, ticker, date, asset_type)

        if self.debug or self.config.get("print_usage", False):
            print(self.usage.report())

        self._write_transcript(ticker, date, asset_type)

        return final_state, signal

    def _write_transcript(self, ticker: str, date: str, asset_type: str) -> None:
        """Persist the per-agent transcript for this run, if configured."""
        tdir = self.config.get("transcript_dir")
        if not tdir:
            return
        fmt = self.config.get("transcript_format", "jsonl")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(tdir) / f"{ticker}_{date}_{stamp}.{fmt}"
        written = self.usage.write_transcript(
            path,
            meta={"ticker": ticker, "date": date, "asset_type": asset_type,
                  "mode": "single_ticker"},
        )
        if self.debug or self.config.get("print_usage", False):
            print(f"[transcript] wrote {written}")

    @staticmethod
    def _compose_briefing(state: dict, ticker: str, date: str, asset_type: str = "stock") -> str:
        """Concatenate the four analyst reports into a single briefing string.

        The fourth section's heading depends on asset_type: "Fundamentals" for
        stocks, "Composition" for ETFs. The underlying state field is the
        same (`fundamentals_report`) regardless.
        """
        fourth_heading = {
            "etf":        "Composition",
            "income_alt": "Fixed Income / Alternatives",
        }.get(asset_type, "Fundamentals")
        sections = [
            ("Market",        state.get("market_report", "")),
            ("Sentiment",     state.get("sentiment_report", "")),
            ("News",          state.get("news_report", "")),
            (fourth_heading,  state.get("fundamentals_report", "")),
        ]
        header = f"# Investment Briefing — {ticker} ({date})\n"
        body   = "\n\n".join(
            f"---\n\n# {name} Analyst\n\n{content.strip() or '_(no report)_'}"
            for name, content in sections
        )
        return f"{header}\n{body}\n"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_with_stream(self, initial_state: dict) -> dict:
        """Stream node events to stdout, then return the final accumulated state."""
        final_state: dict = {}
        for chunk in self._graph.stream(initial_state, stream_mode="values"):
            node_name = list(chunk.keys())[0] if chunk else "?"
            print(f"[graph] node={node_name}")
            final_state = chunk  # stream_mode="values" yields full state each step
        return final_state
