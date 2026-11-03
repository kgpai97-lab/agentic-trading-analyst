"""Forward-test runner — five independent weekday paper-portfolio tracks.

Why this exists (Phase 3): a live, frozen-prompt track record is the
hardest-to-fake evidence the project can produce — its value is pure elapsed
time, so it must start accumulating now and run behind every later phase. We run
FIVE independent portfolios, one per weekday (Mon–Fri): the Monday track holds
its own state and rebalances every Monday, etc. Five tracks accumulate picks
~5× faster AND give a robustness check (if it only works on one weekday, a
single series would hide that). Caveat: the five tracks are highly correlated
(adjacent weekdays see nearly the same market) — fine for accumulating
picks/claims, but a significance test must treat them as a correlated series,
NOT 5× the independent sample.

Each track lives under <forward_dir>/<weekday>/:
    portfolio.json   — the track's CURRENT holdings (read each run, rewritten
                       with the new target weights after the rebalance).
    history.jsonl    — append-only: one record per run (date, target weights,
                       per-ticker analyst signals + claims, report/manifest stem).
    reports/, transcripts/, snapshots/ — the per-run artefacts (PortfolioGraph
                       writes here via config; snapshots are the input provenance).

The flow is a pure rebalance (no candidate ticker): fetch live → decide → store.
Prompts are frozen (their hashes are in each run's manifest). On the FIRST run
for a weekday the track is seeded from `seed_path`.
"""

from __future__ import annotations

import json
from datetime import date as _date, datetime
from pathlib import Path

from agentic.default_config import get_config
from agentic.graph.portfolio_graph import PortfolioGraph

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]


def weekday_name(d: _date) -> str:
    """Lowercase weekday name; weekends map to Friday (markets closed)."""
    idx = d.weekday()  # Mon=0 .. Sun=6
    if idx >= 5:       # Sat/Sun → run the Friday track
        idx = 4
    return _WEEKDAYS[idx]


def _signal_to_dict(sig) -> dict:
    """Serialise an AnalystSignal (pydantic) to a plain JSON-able dict.

    Robust to a plain-dict signal too (defensive). Claims keep their full
    fields so the Phase-5 checker can score them later.
    """
    if hasattr(sig, "model_dump"):
        return sig.model_dump(mode="json")
    return sig if isinstance(sig, dict) else {}


def _collect_signals(per_ticker_reports: dict) -> dict:
    """{ticker: {analyst: signal_dict}} from the run's per_ticker_reports."""
    out: dict[str, dict] = {}
    for ticker, entry in (per_ticker_reports or {}).items():
        sigs = entry.get("signals") or {}
        if sigs:
            out[ticker] = {name: _signal_to_dict(s) for name, s in sigs.items()}
    return out


# Default forward-test universe: 3 distinct portfolios (each a different code
# path / risk profile), each run on all 5 weekdays → 15 independent tracks.
DEFAULT_PORTFOLIOS: dict[str, str] = {
    "diversified":  "portfolio_test.json",
    "conservative": "portfolio_test_cash.json",
    "concentrated": "portfolio_test_concentrated.json",
}


class ForwardTest:
    """Drives N portfolios × 5 weekday tracks.

    Layout: outputs/forward/<portfolio>/<weekday>/. One run_all() per invocation
    runs EVERY portfolio for today's weekday (so a single nightly scheduled task
    advances all of that weekday's tracks). Each (portfolio, weekday) pair is an
    independent track with its own portfolio.json / history.jsonl / reports /
    transcripts / snapshots, its own GARCH benchmark and scoreboard downstream.
    """

    def __init__(
        self,
        portfolios: dict[str, str] | None = None,
        forward_dir: str | Path = "outputs/forward",
        config: dict | None = None,
        debug: bool = False,
        seed_hold_built: bool = False,
    ) -> None:
        self.portfolios = dict(portfolios or DEFAULT_PORTFOLIOS)
        self.forward_dir = Path(forward_dir)
        self.debug = debug
        # Whether to seed the `hold_built` NAV book (the full-conviction build,
        # then held). By design it is derived from the FULL-COVERAGE (1.0) run
        # only — the focused (0.5) run does NOT carry it. forward_run.py sets this
        # True for the `full` variant only.
        self.seed_hold_built = seed_hold_built
        # Caller config overrides win; we layer per-track output dirs on top per run.
        self._base_overrides = dict(config or {})

    # ---- track state ------------------------------------------------------

    def _track_dir(self, portfolio: str, weekday: str) -> Path:
        return self.forward_dir / portfolio / weekday

    def _portfolio_path(self, portfolio: str, weekday: str) -> Path:
        return self._track_dir(portfolio, weekday) / "portfolio.json"

    def _ensure_seeded(self, portfolio: str, weekday: str) -> Path:
        """Create the track dir on first use, writing BOTH the working
        `portfolio.json` (mutable, advances each run) AND an immutable
        `seed_portfolio.json` (the client's original inception allocation, never
        touched again — the permanent record of where the track started)."""
        tdir = self._track_dir(portfolio, weekday)
        tdir.mkdir(parents=True, exist_ok=True)
        pf = self._portfolio_path(portfolio, weekday)
        if not pf.exists():
            seed_path = Path(self.portfolios[portfolio])
            seed = json.loads(seed_path.read_text(encoding="utf-8"))
            pf.write_text(json.dumps(seed, indent=2), encoding="utf-8")
            # Immutable inception record (written ONCE).
            seed_copy = tdir / "seed_portfolio.json"
            if not seed_copy.exists():
                seed_copy.write_text(json.dumps(seed, indent=2), encoding="utf-8")
            self._log(f"[forward/{portfolio}/{weekday}] seeded from {seed_path}")
        return pf

    def _current_weights(self, portfolio: str, weekday: str) -> dict:
        """The track's CURRENT holdings (weights the upcoming run starts from)."""
        pf = self._portfolio_path(portfolio, weekday)
        if not pf.exists():
            return {}
        data = json.loads(pf.read_text(encoding="utf-8"))
        return {t: (v["weight"] if isinstance(v, dict) else float(v))
                for t, v in data.get("holdings", {}).items()}

    def _write_back_weights(self, portfolio: str, weekday: str, date_str: str,
                            target_weights: dict) -> None:
        """Persist the new target weights as the track's CURRENT holdings AND a
        DATED, immutable snapshot — so every portfolio state over time is kept,
        never destructively overwritten.

        - `portfolio.json`            : the mutable "current holdings" pointer
                                        (advances each run; what the next run reads).
        - `portfolios/portfolio_<date>.json` : an immutable per-run snapshot of
                                        the post-rebalance book, dated.
        Preserves each holding's `type`; objective + constraints carry over so the
        track keeps its identity.
        """
        pf = self._portfolio_path(portfolio, weekday)
        data = json.loads(pf.read_text(encoding="utf-8"))
        old_holdings = data.get("holdings", {})

        def _type_for(ticker: str) -> str:
            prev = old_holdings.get(ticker)
            if isinstance(prev, dict) and prev.get("type"):
                return prev["type"]
            return "cash" if ticker.upper() == "CASH" else "equity"

        data["holdings"] = {
            t: {"weight": round(float(w), 6), "type": _type_for(t)}
            for t, w in target_weights.items()
        }
        data["as_of"] = date_str
        # Dated immutable snapshot (full state history of the book).
        snap_dir = self._track_dir(portfolio, weekday) / "portfolios"
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / f"portfolio_{date_str}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8")
        # Update the mutable current pointer.
        pf.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ---- run --------------------------------------------------------------

    def run_all(self, run_date: str | None = None) -> list[dict]:
        """Run EVERY portfolio for today's (or `run_date`'s) weekday.

        Returns one summary record per portfolio. A failure in one portfolio is
        isolated (logged, recorded) so the others still run — important for an
        unattended nightly job.
        """
        records = []
        for name in self.portfolios:
            try:
                records.append(self.run_one(name, run_date))
            except Exception as exc:  # noqa: BLE001 — isolate per-portfolio failure
                d = (datetime.strptime(run_date, "%Y-%m-%d").date()
                     if run_date else _date.today())
                self._log(f"[forward/{name}/{weekday_name(d)}] ERROR: {exc}")
                records.append({"portfolio": name, "date": d.isoformat(),
                                "error": str(exc)})
        return records

    def run_one(self, portfolio: str, run_date: str | None = None) -> dict:
        """Run one portfolio's weekday track once."""
        d = datetime.strptime(run_date, "%Y-%m-%d").date() if run_date else _date.today()
        date_str = d.isoformat()
        weekday = weekday_name(d)
        tag = f"{portfolio}/{weekday}"
        tdir = self._track_dir(portfolio, weekday)
        pf = self._ensure_seeded(portfolio, weekday)

        # Per-track output dirs + frozen-prompt provenance (snapshots ON).
        overrides = dict(self._base_overrides)
        overrides["snapshot_dir"]   = str(tdir / "snapshots")
        overrides["report_dir"]     = str(tdir / "reports")
        overrides["transcript_dir"] = str(tdir / "transcripts")

        # Two-stage cadence: the FIRST rebalance of a track is the decisive
        # initial repositioning (client's raw seed → a sensible book that meets
        # the objective), so it moves a LARGER fraction (`cadence_factor_initial`,
        # default 0.5). Subsequent weekly rebalances damp to `cadence_factor`
        # (0.1) to avoid noise-chasing. "First" = the track has no history yet.
        is_first = not (tdir / "history.jsonl").exists()
        if is_first:
            overrides["cadence_factor"] = float(
                self._base_overrides.get("cadence_factor_initial", 0.5))
        cfg = get_config(overrides)

        # Buy-and-hold drift: before the PM runs, float last week's held book to
        # TODAY's prices and write it back, so the PM reasons from the TRUE
        # current weights (where the book actually drifted to), not last week's
        # stale targets. Best-effort — a price hiccup leaves the book unchanged
        # (no worse than the old behaviour). Skipped on the first run (no prior
        # date to drift from). The NAV layer marks the same returns in lockstep.
        if not is_first:
            self._drift_current_book(portfolio, weekday, tdir, date_str)

        # The weights this run STARTS from (current pointer, post-drift,
        # pre-rebalance) — so the full state trajectory is captured in history.
        start_weights = self._current_weights(portfolio, weekday)

        self._log(f"[forward/{tag}] rebalance as of {date_str}")
        pg = PortfolioGraph(config=cfg, debug=self.debug)
        state, _report_md = pg.propagate(None, pf, date_str)   # pure rebalance

        decision = state.get("final_decision", {}) or {}
        target_weights = decision.get("target_weights", {})
        signals = _collect_signals(state.get("per_ticker_reports", {}))

        record = {
            "portfolio":      portfolio,
            "weekday":        weekday,
            "date":           date_str,
            "run_at":         datetime.now().astimezone().isoformat(),
            "first_run":      is_first,
            "start_weights":  start_weights,           # book going INTO this run
            "target_weights": target_weights,          # book AFTER this rebalance
            "changes":        decision.get("changes", []),
            "proposed_changes": decision.get("proposed_changes", []),  # full-conviction (pre-damping)
            "constraints_applied": decision.get("constraints_applied", {}),
            "narrative":      decision.get("narrative", ""),
            "signals":        signals,
        }
        self._append_history(portfolio, weekday, record)
        if target_weights:
            self._write_back_weights(portfolio, weekday, date_str, target_weights)

        # Mark-to-market: advance our model + lockstep benchmarks' NAV over the
        # week that just elapsed, and set next week's held weights. Best-effort —
        # a NAV/price hiccup must not fail the (already-completed) rebalance.
        nav_row = None
        if target_weights:
            try:
                nav_row = self._update_nav(portfolio, weekday, tdir, date_str,
                                           target_weights, state, start_weights)
            except Exception as exc:  # noqa: BLE001
                self._log(f"[forward/{tag}] NAV update failed: {exc}")
        if nav_row:
            record["nav"] = nav_row.get("nav", {})
            record["exec_turnover"] = nav_row.get("exec_turnover", {})

        self._log(
            f"[forward/{tag}] done: {len(target_weights)} positions, "
            f"{sum(len(v) for v in signals.values())} analyst signals captured"
        )
        return record

    def _drift_current_book(self, portfolio: str, weekday: str, tdir,
                            date_str: str) -> None:
        """Float the working portfolio.json from last NAV date to date_str.

        Buy-and-hold drift of the CURRENT holdings to today's prices, so the PM
        sees the true present weights. Reads the prior mark date from the NAV
        state; if absent (NAV never ran) it is a no-op. Updates only the mutable
        `portfolio.json` pointer (no dated snapshot — that is reserved for actual
        rebalances). Best-effort: any failure leaves the book untouched.
        """
        import json  # noqa: PLC0415

        from agentic.quant.nav import drift_weights, fetch_asset_returns  # noqa: PLC0415

        state_path = tdir / "nav_state.json"
        if not state_path.exists():
            return
        try:
            last_date = json.loads(state_path.read_text(encoding="utf-8")).get("last_date")
        except Exception:  # noqa: BLE001
            return
        if not last_date or last_date == date_str:
            return

        current = self._current_weights(portfolio, weekday)
        if not current:
            return
        tickers = [t for t in current if t.upper() != "CASH"]
        try:
            rets = fetch_asset_returns(tickers, last_date, date_str)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[forward/{portfolio}/{weekday}] drift fetch failed ({exc}); "
                      "book held flat")
            return
        if not rets:
            return
        drifted = drift_weights(current, rets)

        pf = self._portfolio_path(portfolio, weekday)
        data = json.loads(pf.read_text(encoding="utf-8"))
        old = data.get("holdings", {})

        def _type_for(t: str) -> str:
            prev = old.get(t)
            if isinstance(prev, dict) and prev.get("type"):
                return prev["type"]
            return "cash" if t.upper() == "CASH" else "equity"

        data["holdings"] = {t: {"weight": round(float(w), 6), "type": _type_for(t)}
                            for t, w in drifted.items()}
        data["drifted_to"] = date_str
        pf.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _update_nav(self, portfolio: str, weekday: str, tdir, date_str: str,
                    target_weights: dict, state: dict,
                    start_weights: dict | None = None) -> dict:
        """Mark the track's NAV books (lazy import to keep cold paths light)."""
        from agentic.agents.portfolio.portfolio_manager import apply_changes  # noqa: PLC0415
        from agentic.dataflows.constraints import (  # noqa: PLC0415
            PortfolioConstraints, enforce_constraints)
        from agentic.forward.nav_tracker import NavTracker  # noqa: PLC0415

        ticker_types = state.get("ticker_types", {})
        constraints = state.get("constraints") or PortfolioConstraints()

        # The `hold_built` book = the LLM's FULL-conviction initial build (the PM's
        # proposed changes applied UNDAMPED, i.e. cadence_factor=1.0), then held
        # forever. We reconstruct it from the PM's `proposed_changes` against the
        # pre-rebalance `start_weights`, then pass it through the SAME hard-limit
        # layer the live book gets so the comparison is apples-to-apples. Only
        # consumed at seed time (the book is fixed thereafter).
        # hold_built is seeded ONLY for the full-coverage (1.0) run (self.
        # seed_hold_built). The focused (0.5) run leaves it out, so each run's
        # comparator set is exactly what the user asked for.
        decision = state.get("final_decision", {}) or {}
        proposed = decision.get("proposed_changes", []) or []
        hold_built_weights = None
        if self.seed_hold_built and start_weights and proposed:
            full, _rat, _warn = apply_changes(
                dict(start_weights), proposed, state.get("main_ticker", ""))
            hold_built_weights = enforce_constraints(full, ticker_types, constraints)

        # From the SHARED incremental return cache: each ticker fetched once
        # project-wide, only the new week appended on later runs.
        #   - returns_for_garch: FULL per-asset history (GARCH fits per asset on
        #     its own max history — SPY 15y even if BITO has 4.6y).
        #   - aligned: date-aligned common-overlap frame (Ulcer vol target, set
        #     once at seed; back only as far as ALL holdings exist).
        cache = self._return_cache()
        returns_for_garch = cache.returns_for(list(target_weights), date_str)

        # The vol TARGET measures "did the LLM stay near the CLIENT's risk
        # profile", so it must come from the CLIENT's stated portfolio (the raw
        # imported seed), NOT the LLM's post-rebalance output (which would be
        # circular). `_target_basis_weights` reads the original fixture and
        # handles a degenerate all-cash seed (→ equal-weight the risky menu).
        basis_weights = self._target_basis_weights(portfolio)
        aligned = cache.aligned_returns(list(basis_weights), date_str, update=False)

        tracker = NavTracker(
            tdir, ticker_types=ticker_types, constraints=constraints,
            rebalance_period=self.config_value("rebalance_period", "weekly"),
            cadence_factor=float(self.config_value("cadence_factor", 0.1)),
            annual_vol_target=float(self.config_value("annual_vol_target", 0.10)),
            fat_tail_adj=float(self.config_value("vol_target_fat_tail_adj", 1.0)),
        )
        return tracker.update(date_str, target_weights, returns_for_garch,
                              aligned_returns=aligned, target_basis_weights=basis_weights,
                              hold_built_weights=hold_built_weights)

    def _target_basis_weights(self, portfolio: str) -> dict:
        """The CLIENT's risk-fingerprint weights for the vol target — from the
        raw imported fixture, not the LLM output.

        If the seed is degenerate (mostly/all CASH, like the conservative book
        that starts 100% cash), an all-cash 'portfolio' has ~zero drawdown and no
        meaningful risk profile — so we fall back to an EQUAL-WEIGHT of the menu's
        risky (non-cash) names, i.e. the client's intended investable universe.

        EXCEPTION — `conservative` only (2026-07-18): equal-weighting the whole
        menu produced an overstated 8.80% target (effectively a balanced ~50/50
        equity/bond posture), which doesn't represent a 100%-cash-seeded,
        capital-preservation-first investor. NOTE: `_derive_vol_target` measures
        only the RISKY sleeve and renormalizes its weights, so CASH's 0-vol drag
        never actually reaches the target regardless of how much cash this basis
        nominally holds — only the ratio among the risky names matters. Given
        that, the lowest defensible reachable target (scanned across bond-only
        and bond-tilted combinations) is a pure AGG/TIP 60/40 mix (~6.96%),
        which also matches the objective's stated "income-generating... low
        volatility" (AGG) + "inflation protection" (TIP) language directly,
        without equal-weighting in higher-vol names (QQQ/VWO/DBC/broad equity)
        that were inflating the old target. Every other portfolio keeps the
        general equal-weight fallback unchanged.
        """
        if portfolio == "conservative":
            return {"AGG": 0.60, "TIP": 0.40}

        data = json.loads(Path(self.portfolios[portfolio]).read_text(encoding="utf-8"))
        holdings = data.get("holdings", {})
        weights, risky_menu = {}, []
        for t, v in holdings.items():
            tt = (v.get("type") if isinstance(v, dict) else None) or \
                 ("cash" if t.upper() == "CASH" else "equity")
            w = float(v["weight"]) if isinstance(v, dict) else float(v)
            if t.upper() != "CASH":
                risky_menu.append(t.upper())
                if w > 0:
                    weights[t.upper()] = w
        if weights:
            return weights
        # degenerate seed → equal-weight the investable menu
        return {t: 1.0 / len(risky_menu) for t in risky_menu} if risky_menu else {}

    def _return_cache(self):
        """Lazily build a single shared ReturnCache for this ForwardTest run."""
        cache = getattr(self, "_rc", None)
        if cache is None:
            from agentic.quant.return_cache import ReturnCache  # noqa: PLC0415
            cache = ReturnCache()
            self._rc = cache
        return cache

    def config_value(self, key: str, default):
        return self._base_overrides.get(key, default)

    def _append_history(self, portfolio: str, weekday: str, record: dict) -> None:
        hist = self._track_dir(portfolio, weekday) / "history.jsonl"
        with hist.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _log(self, msg: str) -> None:
        print(msg)
