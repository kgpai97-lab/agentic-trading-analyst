"""PortfolioGraph — the Step 6 orchestrator.

Unlike TradingGraph (which compiles a LangGraph StateGraph for the
per-ticker analyst pipeline), this class is plain Python. The Step 6 flow
is a sequential loop over selected tickers with one conditional follow-up
pass — no cycles, no shared message buffer between roles. Wrapping it in a
StateGraph would add scaffolding without benefit.

Pipeline:
    1. Load portfolio file -> PortfolioState
    2. Composition pass for ALL non-cash holdings (+ main ticker if any): run
       ONLY the fourth-slot analyst (composition / fundamentals /
       fixed_income_alt) -> per ticker a composition report + raw holdings/
       sector tool output. (The expensive market/news/sentiment analysts are
       deferred — they only matter for the focus list and the PM.)
    3. Junior PA -> composition assessment. Reads every holding's composition
       data and scores how much it needs PM focus while rebalancing (poor
       objective fit / cross-holding overlap / concentration), emits
       priority_scores + a composition rationale, and selects the focus list
       (selected_tickers) for the deep roles.
    4. For each FOCUS ticker (cost control): market + news + sentiment analysts
       -> assemble the four-section briefing (MNS + composition) -> Head Analyst
       (consolidates + composition raw tool output) -> Researcher (bull + bear).
    5. Portfolio Manager (initial) — independent rebalance from the Junior PA
       composition view + Head/Researcher + MNS reports + composition raw output.
    6. If the PM flags uncertain_holdings (held names outside the focus list it
       wants deep research on): they already have a composition report from
       Phase 2, so run MNS -> Head -> Researcher on each, then PM (final pass).
    7. Compose final markdown report.

The same UsageCallback is shared with the inner TradingGraph so token
accounting aggregates across both layers.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agentic.agents.portfolio.head_analyst import create_head_analyst
from agentic.agents.portfolio.junior_pa import create_junior_pa
from agentic.agents.portfolio.portfolio_manager import create_portfolio_manager
from agentic.agents.portfolio.researcher import create_researcher
from agentic.dataflows.constraints import (
    PortfolioConstraints,
    enforce_constraints,
    scan_objective_for_numeric_constraints,
)
from agentic.dataflows.manifest import write_manifest
from agentic.default_config import get_config
from agentic.graph.trading_graph import TradingGraph
from agentic.llm_clients.factory import create_llm_pair
from agentic.llm_clients.usage_callback import UsageCallback

_WEIGHT_TOLERANCE = 0.01

# Map the four JSON-facing asset classes to the internal asset_type the
# inner TradingGraph's build step switches on. bond_fund and alternative_fund
# share one "income_alt" graph; they differ only by the fund_subclass hint
# passed at invoke time.
_CLASS_TO_ASSET_TYPE: dict[str, str] = {
    "equity":           "stock",
    "equity_fund":      "etf",
    "bond_fund":        "income_alt",
    "alternative_fund": "income_alt",
}

# Classes that need a fund_subclass hint threaded into the analyst prompt.
_INCOME_ALT_CLASSES = {"bond_fund", "alternative_fund"}

_VALID_CLASSES = set(_CLASS_TO_ASSET_TYPE) | {"cash"}


class PortfolioGraph:
    """Orchestrates the portfolio-aware Step 6 pipeline.

    Args:
        config:  Overrides on top of DEFAULT_CONFIG. Notable keys:
                 - asset_type — DEFAULT fourth analyst for tickers without an
                   explicit class (e.g. the main ticker when no type given)
                 - priority_threshold (float in [0, 1])
                 - max_pm_uncertainty_rounds (int, 0 or 1)
                 - lookback_days, deep/quick_think_llm, etc.
        debug:   Pass through to TradingGraph for streaming per-node logs.

    Per-ticker asset-class routing: each holding declares a `type` in the
    portfolio JSON (equity / equity_fund / bond_fund / alternative_fund). The
    orchestrator keeps a cache of inner TradingGraphs keyed by internal
    asset_type and routes each ticker to the matching one, so a mixed-asset
    portfolio is analysed correctly without per-ticker graph rebuilds.
    """

    def __init__(self, config: dict | None = None, debug: bool = False) -> None:
        self.config = get_config(config)
        self.debug  = debug

        # Inner per-ticker graphs, built lazily and cached by internal
        # asset_type ("stock" | "etf" | "income_alt"). print_usage is forced
        # off so the inner graphs don't dump a report on every call — we
        # aggregate and print once at the end of propagate().
        self._inner_cache: dict[tuple[str, str], TradingGraph] = {}

        # Layer-3 (portfolio roles) LLM with its own UsageCallback so we can
        # report layer-1+2 vs layer-3 spend separately.
        self.usage = UsageCallback()
        self.deep_llm, self.quick_llm = create_llm_pair(
            self.config,
            callbacks=[self.usage],
            deep_num_predict=self.config.get("deep_num_predict"),
            deep_num_ctx=self.config.get("deep_num_ctx"),
        )

        # Both tiers are gpt-oss:20b with reasoning=True (reasoning=False → empty
        # content on gpt-oss). Junior PA uses deep_llm; the tiers differ only by
        # num_predict headroom, which is fine for the urgency-scoring task.
        self._junior_pa  = create_junior_pa(self.deep_llm)
        self._head       = create_head_analyst(self.deep_llm)
        self._researcher = create_researcher(self.deep_llm)
        self._pm_initial, self._pm_uncertainty = create_portfolio_manager(
            self.deep_llm,
            cadence_factor=float(self.config.get("cadence_factor", 0.1)),
        )

    # ------------------------------------------------------------------ run

    def propagate(
        self,
        main_ticker: str | None,
        portfolio_path: str | Path,
        date: str,
        main_ticker_type: str | None = None,
    ) -> tuple[dict, str]:
        """Run the full Step 6 pipeline.

        main_ticker is OPTIONAL: pass None/"" for a pure rebalance of the
        existing holdings (no candidate add). When supplied, main_ticker_type
        (or a `main_ticker_type` field in the JSON) gives its asset class;
        defaults to "equity".

        Returns (final_state, rendered_report_md).
        """
        self.usage.reset()
        self._inner_usage_totals: dict[str, dict] = {}

        state = self._load_initial_state(
            main_ticker, portfolio_path, date, main_ticker_type=main_ticker_type
        )
        # Known tradeable tickers for this run — every holding plus the main
        # ticker. Published into each inner graph's context so _normalise_ticker
        # can snap a hallucinated cross-holding typo (e.g. "TQQ3"→"TQQQ") even
        # when the active analyst is studying a different name. CASH excluded.
        self._universe: tuple[str, ...] = tuple(
            t for t in (
                [state["main_ticker"]] if state.get("main_ticker") else []
            ) + list(state["current_weights"])
            if t and t.upper() != "CASH"
        )

        self._log(f"[portfolio] objective: {state['objective'][:80]}...")
        self._log(f"[portfolio] holdings: {state['current_weights']}")
        self._log(f"[portfolio] ticker_types: {state['ticker_types']}")
        self._log(f"[portfolio] main_ticker: {state['main_ticker'] or '(none — pure rebalance)'}")

        # 1. Composition pass for ALL holdings ----------------------------
        # Only the fourth-slot analyst (composition / fundamentals /
        # fixed_income_alt) runs here — it produces the holdings/sector data the
        # Junior PA needs to judge whole-book overlap/concentration/fit. The
        # expensive market/news/sentiment analysts are deferred to Phase 3 and
        # run ONLY for the focus list (cost control).
        per_ticker: dict[str, dict] = {}
        for ticker in self._tickers_to_analyse(state):
            ticker_class = state["ticker_types"].get(ticker, "equity")
            self._log(f"[portfolio] --- composition pass: {ticker} (class={ticker_class}) ---")
            comp_report, comp_tool_out, comp_signals = self._run_composition(ticker, date, ticker_class)
            per_ticker[ticker] = {
                "composition_report": comp_report,
                "tool_outputs":       comp_tool_out,   # composition raw only
                "signals":            dict(comp_signals),  # AnalystSignal per analyst
            }
        state["per_ticker_reports"] = per_ticker

        # 2. Junior PA — composition assessment + focus selection ----------
        # Reads each holding's composition data (Phase 1) and scores how much it
        # needs PM focus while rebalancing (poor objective fit, cross-holding
        # overlap, single-name/sector concentration), then selects the focus
        # list for the deep roles.
        self._log("[portfolio] running Junior Portfolio Analyst (composition assessment)")
        junior_out = self._junior_pa(state)
        state.update(junior_out)
        self._log(f"[portfolio] selected_tickers (PM focus): {state['selected_tickers']}")
        self._log(f"[portfolio] priority_scores: {state['priority_scores']}")
        self._log(f"[portfolio] priority_rationale:\n{state['priority_rationale']}")

        # 3. Focus list: market/news/sentiment -> Head -> Researcher -------
        for ticker in state["selected_tickers"]:
            entry = per_ticker.get(ticker)
            if entry is None:   # focus ticker not analysed (shouldn't happen)
                self._log(f"[portfolio] focus ticker {ticker} missing composition; skipping deep chain")
                continue
            ticker_class = state["ticker_types"].get(ticker, "equity")
            self._deepen_focus_ticker(ticker, date, ticker_class, entry)
        state["per_ticker_reports"] = per_ticker

        # 4. Portfolio Manager (initial) ----------------------------------
        self._log("[portfolio] running Portfolio Manager (initial)")
        pm_initial = self._pm_initial(state)
        state["pm_initial"] = pm_initial
        self._log(f"[portfolio] PM target_weights: {pm_initial.get('target_weights')}")
        self._log(f"[portfolio] PM narrative:\n{pm_initial.get('narrative')}")
        self._log(f"[portfolio] PM uncertain_holdings: {pm_initial.get('uncertain_holdings')}")

        uncertain = pm_initial.get("uncertain_holdings", []) or []
        max_rounds = int(self.config.get("max_pm_uncertainty_rounds", 1))

        if uncertain and max_rounds > 0:
            # 5. Uncertainty pass. The PM may flag holdings outside the focus
            # list it wants deep research on. Those already have a composition
            # report + tool output from Phase 1 but NO market/news/sentiment, so
            # they get the same deepening as a focus ticker (MNS -> briefing ->
            # Head -> Researcher), then the PM is re-invoked once.
            self._log(f"[portfolio] PM flagged uncertain: {uncertain}; running deep chain")
            uncertainty_reports: dict[str, str] = {}
            for ticker in uncertain:
                entry = per_ticker.get(ticker)
                if entry is None:
                    # Not analysed in Phase 1 (e.g. CASH or an invented name).
                    self._log(f"[portfolio] skipping {ticker}: no Phase-1 composition")
                    continue
                ticker_class = state["ticker_types"].get(ticker, "equity")
                self._deepen_focus_ticker(ticker, date, ticker_class, entry)
                uncertainty_reports[ticker] = entry.get("researcher_report", "")
            state["uncertainty_reports"] = uncertainty_reports
            state["per_ticker_reports"] = per_ticker

            self._log("[portfolio] running Portfolio Manager (uncertainty pass)")
            final = self._pm_uncertainty(state, uncertainty_reports)
            self._log(f"[portfolio] PM (final) target_weights: {final.get('target_weights')}")
            self._log(f"[portfolio] PM (final) narrative:\n{final.get('narrative')}")
            state["final_decision"] = final
        else:
            state["uncertainty_reports"] = {}
            state["final_decision"] = pm_initial

        # Phase 1d (Decision B): the SINGLE hard-constraint layer. Applied here,
        # after the PM (and, later, after any risk overlay), to the final target
        # weights. This is the one place numeric limits are enforced — never in a
        # prompt. The same enforce_constraints() with the same PortfolioConstraints
        # is applied to the GARCH benchmark (Phase 4a) so comparisons are fair.
        self._enforce_hard_constraints(state)

        self._validate_final_weights(state["final_decision"])

        report_md = self._compose_report(state)

        if self.debug or self.config.get("print_usage", False):
            print("=== Inner (per-ticker analyst) usage ===")
            self._print_inner_totals()
            print("=== Outer (portfolio layer) usage ===")
            print(self.usage.report())

        self._write_transcript(main_ticker, date)
        self._write_report(main_ticker, date, report_md, state["final_decision"])

        return state, report_md

    def _write_transcript(self, main_ticker: str | None, date: str) -> None:
        """Persist the merged per-agent transcript (analysts + portfolio roles)."""
        tdir = self.config.get("transcript_dir")
        if not tdir:
            return
        # Sort folded entries by their wall-clock timestamp so the file reads
        # in execution order (inner entries are appended out-of-band on merge).
        self.usage.transcript.sort(key=lambda e: e.get("timestamp", ""))
        fmt = self.config.get("transcript_format", "jsonl")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        label = (main_ticker or "rebalance").upper()
        path = Path(tdir) / f"portfolio_{label}_{date}_{stamp}.{fmt}"
        written = self.usage.write_transcript(
            path,
            meta={"main_ticker": main_ticker or "(pure rebalance)", "date": date,
                  "mode": "portfolio"},
        )
        if self.debug or self.config.get("print_usage", False):
            print(f"[transcript] wrote {written}")

    def _write_report(
        self,
        main_ticker: str | None,
        date: str,
        report_md: str,
        final_decision: dict,
    ) -> None:
        """Persist the rendered markdown report + structured final_decision.

        Writes a timestamped pair (``.md`` + ``.json``) under ``report_dir``.
        Mirrors _write_transcript's naming so a run's report, decision, and
        transcript share the same ``<label>_<date>_<stamp>`` stem. Disabled
        when ``report_dir`` is empty (report is still returned to the caller).
        """
        rdir = self.config.get("report_dir")
        if not rdir:
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        label = (main_ticker or "rebalance").upper()
        stem = f"portfolio_{label}_{date}_{stamp}"
        out_dir = Path(rdir)
        out_dir.mkdir(parents=True, exist_ok=True)

        md_path = out_dir / f"{stem}.md"
        md_path.write_text(report_md, encoding="utf-8")

        json_path = out_dir / f"{stem}.json"
        payload = {
            "main_ticker": main_ticker or "(pure rebalance)",
            "date": date,
            "final_decision": final_decision,
        }
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Phase 2d: reproducibility manifest sharing this run's stem (model ids,
        # frozen-prompt hashes, config, snapshot pointer, dep versions).
        prompts_dir = Path(__file__).resolve().parent.parent.parent / "prompts"
        manifest_path = write_manifest(
            out_dir, stem, self.config, prompts_dir, date, label=label,
            extra={"main_ticker": main_ticker or "", "report_stem": stem},
        )

        if self.debug or self.config.get("print_usage", False):
            print(f"[report] wrote {md_path}")
            print(f"[report] wrote {json_path}")
            if manifest_path:
                print(f"[report] wrote {manifest_path}")

    # ------------------------------------------------------------- helpers

    def _inner_for(self, asset_type: str, subset: str) -> TradingGraph:
        """Return (building lazily) the cached inner TradingGraph for an
        (asset_type, analyst_subset) pair.

        The portfolio layer runs two passes — `subset="composition"` (fourth
        analyst, all holdings) and `subset="mns"` (market/news/sentiment, focus
        list) — so we cache one graph per (asset_type, subset).
        """
        key = (asset_type, subset)
        graph = self._inner_cache.get(key)
        if graph is None:
            inner_cfg = dict(self.config)
            inner_cfg["print_usage"] = False
            # Inner graphs don't each write their own transcript file — their
            # entries are folded into the single outer portfolio transcript
            # (see _merge_inner_usage). Disable per-inner-run file writes.
            inner_cfg["transcript_dir"] = ""
            inner_cfg["asset_type"]     = asset_type
            inner_cfg["analyst_subset"] = subset
            graph = TradingGraph(config=inner_cfg, debug=self.debug)
            self._inner_cache[key] = graph
            self._log(f"[portfolio] built inner graph for {key!r}")
        # Publish THIS run's universe onto the (possibly cached) inner graph so
        # set_run_context picks it up for the typo-snap. Set every call — the
        # universe is per-run, the inner graph is cached across runs.
        graph.config["universe"] = getattr(self, "_universe", ())
        return graph

    def _tickers_to_analyse(self, state: dict) -> list[str]:
        """The set of tickers to run the analyst layer on: every non-cash
        holding, plus the main ticker (candidate add) if it isn't already held.

        Main ticker first (so its analysis leads the report), then holdings in
        portfolio order.
        """
        main = state.get("main_ticker", "")
        ordered: list[str] = []
        if main:
            ordered.append(main)
        for ticker in state["current_weights"]:
            if ticker.upper() == "CASH" or ticker == main:
                continue
            ordered.append(ticker)
        return ordered

    def _run_composition(
        self, ticker: str, date: str, ticker_class: str
    ) -> tuple[str, str, dict]:
        """Phase 1: run ONLY the fourth-slot analyst (composition / fundamentals
        / fixed_income_alt) for one ticker. Returns
        (composition_report, composition_raw_tool_output, signals).

        The report is the fourth-section text (`fundamentals_report` field); the
        tool output is the captured holdings/sector block — the look-through data
        the Junior PA and PM need; `signals` is the inner AnalystSignal dict
        (Step 7 Phase 1b) so the forward test can persist claims for Phase 5.
        """
        asset_type = _CLASS_TO_ASSET_TYPE.get(ticker_class, "stock")
        fund_subclass = ticker_class if ticker_class in _INCOME_ALT_CLASSES else ""
        inner = self._inner_for(asset_type, "composition")
        final_state, _ = inner.propagate(ticker, date, fund_subclass=fund_subclass)
        self._merge_inner_usage(ticker, inner)
        report = final_state.get("fundamentals_report", "")
        tool_outputs = self._format_tool_outputs(final_state.get("tool_outputs", {}))
        return report, tool_outputs, final_state.get("signals", {})

    def _run_mns(
        self, ticker: str, date: str, ticker_class: str
    ) -> tuple[dict[str, str], dict]:
        """Phase 3: run market + news + sentiment for one (focus) ticker.
        Returns ({"market":…, "sentiment":…, "news":…} prose, signals).

        MNS tool output is captured by the inner graph but intentionally NOT
        returned — the PM reads the MNS report prose, not the raw OHLCV/news
        dumps (the composition raw from Phase 1 is the only raw data forwarded).
        `signals` is the inner AnalystSignal dict (market/news/sentiment) so the
        forward test can persist their claims for Phase 5.
        """
        asset_type = _CLASS_TO_ASSET_TYPE.get(ticker_class, "stock")
        fund_subclass = ticker_class if ticker_class in _INCOME_ALT_CLASSES else ""
        inner = self._inner_for(asset_type, "mns")
        final_state, _ = inner.propagate(ticker, date, fund_subclass=fund_subclass)
        self._merge_inner_usage(ticker, inner)
        reports = {
            "market":    final_state.get("market_report", ""),
            "sentiment": final_state.get("sentiment_report", ""),
            "news":      final_state.get("news_report", ""),
        }
        return reports, final_state.get("signals", {})

    def _deepen_focus_ticker(
        self, ticker: str, date: str, ticker_class: str, entry: dict
    ) -> None:
        """Run the deep chain for one focus (or PM-flagged uncertain) ticker,
        mutating its `per_ticker_reports` entry in place.

        The entry already holds `composition_report` + `tool_outputs` from
        Phase 1. Here we add the market/news/sentiment reports, the assembled
        four-section briefing, and the Head -> Researcher outputs.
        """
        mns, mns_signals = self._run_mns(ticker, date, ticker_class)
        asset_type = _CLASS_TO_ASSET_TYPE.get(ticker_class, "stock")
        briefing = self._assemble_briefing(
            ticker, date, asset_type, mns, entry.get("composition_report", "")
        )
        tool_out = entry.get("tool_outputs", "")   # composition raw
        head_md  = self._head(ticker, briefing, tool_out)
        self._log(f"[portfolio] {ticker} head assessment ({len(head_md)} chars):\n{head_md}")
        res_md   = self._researcher(ticker, head_md, tool_out)
        self._log(f"[portfolio] {ticker} researcher report ({len(res_md)} chars):\n{res_md}")
        entry["mns_reports"]       = mns
        entry["briefing"]          = briefing
        entry["head_assessment"]   = head_md
        entry["researcher_report"] = res_md
        # Merge MNS analyst signals into the composition signals captured in
        # Phase 1, so the entry carries the full set for Phase-5 claim eval.
        merged = dict(entry.get("signals", {}))
        merged.update(mns_signals)
        entry["signals"] = merged

    @staticmethod
    def _assemble_briefing(
        ticker: str, date: str, asset_type: str,
        mns: dict[str, str], composition_report: str,
    ) -> str:
        """Assemble the four-section briefing for the Head Analyst from the
        Phase-3 MNS reports and the Phase-1 composition report. Mirrors
        TradingGraph._compose_briefing's section layout.
        """
        fourth_heading = {
            "etf":        "Composition",
            "income_alt": "Fixed Income / Alternatives",
        }.get(asset_type, "Fundamentals")
        sections = [
            ("Market",       mns.get("market", "")),
            ("Sentiment",    mns.get("sentiment", "")),
            ("News",         mns.get("news", "")),
            (fourth_heading, composition_report),
        ]
        header = f"# Investment Briefing — {ticker} ({date})\n"
        body   = "\n\n".join(
            f"---\n\n# {name} Analyst\n\n{content.strip() or '_(no report)_'}"
            for name, content in sections
        )
        return f"{header}\n{body}\n"

    @staticmethod
    def _format_tool_outputs(tool_outputs: dict[str, str]) -> str:
        """Render the per-analyst raw tool captures into one labelled block."""
        if not tool_outputs:
            return ""
        blocks = [
            f"--- {analyst} tools ---\n{text}"
            for analyst, text in tool_outputs.items()
            if str(text).strip()
        ]
        return "\n\n".join(blocks)

    def _merge_inner_usage(self, ticker: str, inner: TradingGraph) -> None:
        """Fold the inner graph's buckets into self._inner_usage_totals.

        Keys are prefixed with the ticker so per-ticker breakdown survives
        the inner reset on the next call.
        """
        for bucket_name, vals in inner.usage.buckets.items():
            key = f"{ticker}/{bucket_name}"
            tgt = self._inner_usage_totals.setdefault(
                key,
                {"in_tokens": 0, "out_tokens": 0, "llm_calls": 0,
                 "tool_calls": 0, "llm_seconds": 0.0, "tool_seconds": 0.0},
            )
            for k, v in vals.items():
                tgt[k] = tgt.get(k, 0) + v

        # Fold the inner analysts' transcript entries into the outer callback so
        # the single portfolio transcript file covers analysts AND portfolio
        # roles. Prefix the bucket with the ticker for per-ticker attribution.
        for entry in inner.usage.transcript:
            merged = dict(entry)
            merged["bucket"] = f"{ticker}/{entry['bucket']}"
            self.usage.transcript.append(merged)

    def _load_initial_state(
        self,
        main_ticker: str | None,
        portfolio_path: str | Path,
        date: str,
        main_ticker_type: str | None = None,
    ) -> dict:
        path = Path(portfolio_path)
        data = json.loads(path.read_text(encoding="utf-8"))

        if "objective" not in data or "holdings" not in data:
            raise ValueError(
                f"portfolio file {path} must contain 'objective' (str) and 'holdings' (dict)"
            )

        # Parse the typed holdings shape: {ticker: {"weight": float, "type": str}}.
        # Backward-compat: a bare float value defaults type to "equity".
        holdings: dict[str, float] = {}
        ticker_types: dict[str, str] = {}
        for raw_ticker, val in data["holdings"].items():
            ticker = raw_ticker.upper()
            if isinstance(val, dict):
                weight = float(val["weight"])
                ttype  = str(val.get("type", "equity")).lower()
            else:
                weight = float(val)
                ttype  = "cash" if ticker == "CASH" else "equity"
                print(f"[portfolio] WARNING: holding {ticker} has no type; defaulting to {ttype!r}")
            if ttype not in _VALID_CLASSES:
                raise ValueError(
                    f"holding {ticker} has invalid type {ttype!r}; "
                    f"expected one of {sorted(_VALID_CLASSES)}"
                )
            holdings[ticker] = weight
            ticker_types[ticker] = ttype

        total = sum(holdings.values())
        if abs(total - 1.0) > _WEIGHT_TOLERANCE:
            raise ValueError(
                f"portfolio weights sum to {total:.4f}; expected 1.0 ± {_WEIGHT_TOLERANCE}"
            )

        # Optional main ticker. Resolve its class from CLI/JSON, default equity.
        main = main_ticker.upper() if main_ticker else ""
        if main:
            main_type = (
                main_ticker_type
                or data.get("main_ticker_type")
                or ticker_types.get(main)   # already held? reuse its type
                or "equity"
            ).lower()
            if main_type not in _CLASS_TO_ASSET_TYPE:
                raise ValueError(
                    f"main ticker type {main_type!r} invalid; "
                    f"expected one of {sorted(_CLASS_TO_ASSET_TYPE)}"
                )
            ticker_types[main] = main_type

        threshold = float(self.config.get("priority_threshold", 0.5))

        # Phase 1c: parse the optional `constraints` block (numeric hard limits,
        # Decision B) and warn if numeric limits are stranded in the objective
        # prose where they would be unenforced. Missing block -> permissive
        # defaults (no caps), matching prior behavior.
        constraints = PortfolioConstraints.from_dict(data.get("constraints"))
        for warning in scan_objective_for_numeric_constraints(str(data["objective"])):
            print(f"[portfolio] WARNING: {warning}")

        return {
            "objective":          str(data["objective"]),
            "constraints":        constraints,
            "current_weights":    holdings,
            "ticker_types":       ticker_types,
            "main_ticker":        main,
            "trade_date":         date,
            "asset_type":         self.config.get("asset_type", "stock"),
            "priority_threshold": threshold,

            "priority_scores":    {},
            "priority_rationale": "",
            "selected_tickers":   [],

            "per_ticker_reports": {},

            "pm_initial":          {},
            "uncertainty_reports": {},
            "final_decision":      {},
        }

    def _enforce_hard_constraints(self, state: dict) -> None:
        """Apply the numeric hard limits to final_decision['target_weights'].

        Decision B's single enforcement layer. Mutates the decision in place:
        replaces target_weights with the constrained, renormalised version and
        records which limits were active under `constraints_applied` (two-line
        attribution, mirroring the PM/overlay rationale logging in Decision A).
        A permissive (all-None) PortfolioConstraints leaves weights unchanged
        apart from renormalisation.
        """
        constraints = state.get("constraints") or PortfolioConstraints()
        decision = state["final_decision"]
        before = decision.get("target_weights", {})
        if not before:
            return
        after = enforce_constraints(before, state.get("ticker_types", {}), constraints)
        decision["target_weights"] = {t: round(w, 6) for t, w in after.items()}

        changed = {
            t: (round(before.get(t, 0.0), 4), round(after.get(t, 0.0), 4))
            for t in set(before) | set(after)
            if abs(before.get(t, 0.0) - after.get(t, 0.0)) > 1e-4
        }
        decision["constraints_applied"] = {
            "limits": constraints.model_dump(exclude_none=True),
            "adjusted": changed,
        }
        if changed:
            self._log(
                f"[portfolio] hard constraints adjusted {len(changed)} position(s): "
                f"{changed}"
            )

    @staticmethod
    def _validate_final_weights(decision: dict) -> None:
        # apply_changes() in portfolio_manager.py already conserves mass and
        # renormalises target_weights to sum to 1.0, so this is a safety net:
        # only correct residual floating-point drift, and never silently move
        # weight into CASH (that decision now lives with the PM's change list).
        weights = decision.get("target_weights", {})
        if not weights:
            raise ValueError("Portfolio Manager returned empty target_weights")
        total = sum(float(w) for w in weights.values())
        if abs(total - 1.0) > _WEIGHT_TOLERANCE:
            print(
                f"[portfolio] WARNING: target_weights summed to {total:.4f}; "
                "renormalising (should not happen post apply_changes)"
            )
            if total > 0:
                decision["target_weights"] = {
                    t: round(float(w) / total, 6) for t, w in weights.items()
                }

    @staticmethod
    def _compose_report(state: dict) -> str:
        decision = state["final_decision"]
        weights  = decision.get("target_weights", {})
        rationales = decision.get("per_position_rationale", {})

        lines: list[str] = []
        title_subject = state["main_ticker"] or "Rebalance"
        lines.append(f"# Portfolio Rebalance — {title_subject} ({state['trade_date']})")
        lines.append("")
        lines.append(f"**Objective:** {state['objective']}")
        lines.append("")

        # Junior PA composition assessment (focus scores: higher = needs more
        # PM attention this rebalance — poor fit / overlap / concentration).
        lines.append("## Junior PA Composition Assessment")
        lines.append("")
        lines.append("_Focus score per holding (higher = needs more PM attention "
                     "while rebalancing):_")
        lines.append("")
        scores = state.get("priority_scores", {})
        if scores:
            for t, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
                marker = " ★" if t in state.get("selected_tickers", []) else ""
                lines.append(f"- **{t}**: {s:.3f}{marker}")
        else:
            lines.append("_(no scores)_")
        lines.append("")
        if state.get("priority_rationale"):
            lines.append(f"_Assessment:_ {state['priority_rationale']}")
            lines.append("")
        lines.append(f"_PM focus list (deep-analysed):_ {', '.join(state.get('selected_tickers', []))}")
        lines.append("")

        # Per-ticker research
        lines.append("## Per-Ticker Analysis")
        lines.append("")
        for t, payload in state.get("per_ticker_reports", {}).items():
            lines.append(f"### {t}")
            lines.append("")

            # Focus tickers carry a full assembled briefing (market/news/
            # sentiment + composition); non-focus holdings only ran the
            # composition pass, so show that report alone.
            briefing = payload.get("briefing", "").strip()
            if briefing:
                lines.append("#### Analyst Reports")
                lines.append("")
                lines.append(briefing)
                lines.append("")
            else:
                comp = payload.get("composition_report", "").strip()
                if comp:
                    lines.append("#### Composition Report")
                    lines.append("")
                    lines.append(comp)
                    lines.append("")

            head = payload.get("head_assessment", "").strip()
            if head:
                lines.append("#### Head Analyst Assessment")
                lines.append("")
                lines.append(head)
                lines.append("")

            # Head/Researcher only run on the focus list; non-focus holdings
            # show their composition report but no deep chain.
            research = payload.get("researcher_report", "").strip()
            if research:
                lines.append("#### Bull + Bear Research")
                lines.append("")
                lines.append(research)
                lines.append("")
            elif t not in state.get("selected_tickers", []):
                lines.append("_(not on the PM focus list — composition only, no market/news/sentiment or deep research this cycle)_")
                lines.append("")

        # Final allocation
        lines.append("## Final Target Allocation")
        lines.append("")
        lines.append("| Ticker | Current | Target | Δ |")
        lines.append("|---|---:|---:|---:|")
        current = state.get("current_weights", {})
        all_tickers = sorted(set(current) | set(weights))
        for t in all_tickers:
            cur = current.get(t, 0.0)
            tgt = float(weights.get(t, 0.0))
            delta = tgt - cur
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {t} | {cur:.4f} | {tgt:.4f} | {sign}{delta:.4f} |")
        lines.append("")

        # Rationale for each change (held positions are omitted by design)
        lines.append("## Rationale for Changes")
        lines.append("")
        if rationales:
            for t, txt in rationales.items():
                lines.append(f"- **{t}**: {txt}")
        else:
            lines.append("_No changes — the portfolio was held as-is._")
        lines.append("")

        # Narrative
        lines.append("## Narrative")
        lines.append("")
        lines.append(decision.get("narrative", "_(no narrative)_"))
        lines.append("")

        if state.get("uncertainty_reports"):
            lines.append("## Uncertainty Pass")
            lines.append("")
            lines.append(
                "Portfolio Manager flagged the following holdings for follow-up "
                "research: " + ", ".join(state["uncertainty_reports"].keys())
            )
            lines.append("")

        return "\n".join(lines)

    def _print_inner_totals(self) -> None:
        if not self._inner_usage_totals:
            print("(no inner usage recorded)")
            return
        total_in = total_out = total_calls = 0
        for key, v in sorted(self._inner_usage_totals.items()):
            print(
                f"  {key:<40s} in={v['in_tokens']:>6d} out={v['out_tokens']:>6d} "
                f"calls={v['llm_calls']:>3d} tools={v['tool_calls']:>3d}"
            )
            total_in += v["in_tokens"]
            total_out += v["out_tokens"]
            total_calls += v["llm_calls"]
        print(f"  TOTAL  in={total_in} out={total_out} llm_calls={total_calls}")

    def _log(self, msg: str) -> None:
        if self.debug:
            print(msg)
