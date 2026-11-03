"""Per-track NAV tracker — marks our model + lockstep benchmarks each run.

Lives inside a forward track dir. Persists its state (each book's current held
weights + NAV) in `nav_state.json` so the equity curves survive between weekly
runs, and appends a human-glanceable `nav.jsonl` (one row per run with every
strategy's NAV). This is what turns the forward test from a log of DECISIONS
into a live, comparable TRACK RECORD.

All books DRIFT their weights with realised returns between marks (true
buy-and-hold accounting — winners' weights grow, losers' shrink); none is
silently rebalanced back to a target for free. The actively-managed books
(our_model, garch) then apply their fresh target ON TOP of the drifted book each
week; the passive books (hold_*, random_*) are never retraded, so their drifted
weights simply carry forward.

Lockstep books (all marked over the same week, same realised prices):
  * our_model    — last week's target, drifted over the week, then re-pinned to
                   this week's fresh target.
  * garch        — our own mix scaled by the GARCH vol-target exposure scalar
                   (Decision A: the overlay scales gross exposure, never the mix).
  * hold_initial — inception weights, drifted, never retraded.
  * hold_built   — the LLM's FULL-conviction initial build (cadence_factor=1.0),
                   drifted, never retraded. Isolates the value of ongoing weekly
                   rebalancing: our_model (rebalances) vs hold_built (built once).

The random buy-and-hold comparison CLOUD is NOT a live book here: being passive +
deterministic (fixed seed weights × prices), it is reconstructed OFFLINE on demand
from the cached prices (`scoreboard_run.py`), so its count is unconstrained by the
nightly run's state size.

Returns are marked GROSS — no transaction cost is deducted. Cost is instead
surfaced as a diagnostic: each run records EXECUTION turnover (½Σ|target −
drifted|), the weight actually traded to correct drift back to target. A passive
book trades nothing, so its execution turnover is 0 by construction.

Sequence per run (from the 2nd run on; the 1st just initialises at NAV 1.0):
  1. fetch realised per-ticker returns from last run's date to this run's date;
  2. MARK every book with the weights it HELD over that week → compounds NAV;
  3. DRIFT every book's held weights by those same returns (buy-and-hold float);
  4. for the active books only, set this week's fresh held weights ON TOP of the
     drifted book (our_model = fresh target; garch = fresh mix × vol scalar) and
     record the execution turnover vs the drifted book. Passive books keep their
     drifted weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agentic.dataflows.constraints import PortfolioConstraints, enforce_constraints
from agentic.quant.benchmark import GarchVolBenchmark
from agentic.quant.nav import NavBook, drift_weights, fetch_asset_returns

_HIST_LOOKBACK_DAYS = 400


class NavTracker:
    """Owns the NAV books for one forward track; persisted to the track dir."""

    def __init__(self, track_dir: str | Path, ticker_types: dict | None = None,
                 constraints: PortfolioConstraints | None = None,
                 rebalance_period: str = "weekly", cadence_factor: float = 0.1,
                 annual_vol_target: float = 0.10, fat_tail_adj: float = 1.0) -> None:
        self.dir = Path(track_dir)
        self.ticker_types = {k.upper(): v for k, v in (ticker_types or {}).items()}
        self.constraints = constraints or PortfolioConstraints()
        self.rebalance_period = rebalance_period
        self.cadence_factor = cadence_factor
        self.annual_vol_target = annual_vol_target   # fallback when no history
        self.fat_tail_adj = fat_tail_adj             # widens the UI→vol constant
        self._state_path = self.dir / "nav_state.json"
        self._nav_path = self.dir / "nav.jsonl"
        self._state = self._load_state()

    # ---- persistence ------------------------------------------------------

    def _load_state(self) -> dict:
        if self._state_path.exists():
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            # Evict any legacy random_* books left in pre-existing state: the
            # random comparison cloud is now offline-only (reconstructed from the
            # price cache by the scoreboard), so a stray random_* in persisted
            # state must not get marked/drifted by the nightly run.
            books = state.get("books") or {}
            for name in [n for n in books if n.startswith("random_")]:
                del books[name]
            return state
        return {"last_date": None, "books": {}}   # books: name -> {nav, held_weights}

    def _save_state(self) -> None:
        self._state_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    def _append_nav_row(self, row: dict) -> None:
        with self._nav_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    # ---- the per-run update ----------------------------------------------

    def update(self, date: str, our_target_weights: dict,
               returns_for_garch: dict | None = None,
               aligned_returns=None, target_basis_weights: dict | None = None,
               hold_built_weights: dict | None = None) -> dict:
        """Mark the elapsed week and set next week's held weights.

        `our_target_weights` is THIS run's fresh allocation (post enforce).
        `returns_for_garch` = {ticker: full per-asset return array} for the GARCH
        fits; `aligned_returns` = a date-aligned DataFrame over the common window
        for the vol target (set once at seed). `target_basis_weights` = the
        CLIENT's raw-seed risky weights the vol target is derived from (defaults
        to `our_target_weights` if not supplied). `hold_built_weights` = the LLM's
        FULL-conviction initial build (cadence_factor=1.0) used to seed the fixed
        `hold_built` book; only consumed at seed time. On the first call we seed
        every book at NAV 1.0. Returns the appended nav row.
        """
        books = self._state["books"]
        last_date = self._state["last_date"]

        # ---- first run for this track: initialise, no marking yet ----------
        if last_date is None or not books:
            self._seed(date, our_target_weights, returns_for_garch, aligned_returns,
                       target_basis_weights, hold_built_weights)
            row = {"date": date, "first": True,
                   "nav": {n: 1.0 for n in books}}
            self._append_nav_row(row)
            self._state["last_date"] = date
            self._save_state()
            return row

        # ---- mark the week that just elapsed -------------------------------
        held_tickers = sorted({t for b in books.values()
                               for t in b["held_weights"] if t.upper() != "CASH"})
        try:
            asset_rets = fetch_asset_returns(held_tickers, last_date, date)
        except Exception as exc:  # noqa: BLE001 — never break the run on a fetch
            print(f"[nav] WARNING: return fetch failed ({exc}); marking week flat")
            asset_rets = {}

        marked = {}
        for name, b in books.items():
            nb = NavBook(name=name, nav=b["nav"])
            ret = nb.mark(date, b["held_weights"], asset_rets)
            b["nav"] = nb.nav
            marked[name] = {"ret": round(ret, 6), "nav": round(nb.nav, 6)}
            # Buy-and-hold drift: float each book's weights by the same returns.
            # Passive books keep these; active books overwrite in _reweight below.
            b["held_weights"] = drift_weights(b["held_weights"], asset_rets)

        # Where our actual book drifted to over the week (BEFORE we re-pin it to
        # this week's target). This is the TRUE current holding the next run's PM
        # must reason from — the runner writes it into portfolio.json.
        drifted_our_model = dict(books["our_model"]["held_weights"])

        # ---- set this week's fresh held weights on the ACTIVE books --------
        # (applied on top of the drifted book; records execution turnover)
        exec_turnover = self._reweight(date, our_target_weights, returns_for_garch)

        row = {"date": date, "returns": {n: m["ret"] for n, m in marked.items()},
               "nav": {n: m["nav"] for n, m in marked.items()},
               "exec_turnover": exec_turnover,
               "drifted_weights": {k: round(v, 6) for k, v in drifted_our_model.items()}}
        self._append_nav_row(row)
        self._state["last_date"] = date
        self._save_state()
        return row

    # ---- helpers ----------------------------------------------------------

    def _seed(self, date: str, our_weights: dict, returns_for_garch: dict | None,
              aligned_returns=None, target_basis_weights: dict | None = None,
              hold_built_weights: dict | None = None) -> None:
        """Create the books at NAV 1.0 with their inception held weights.

        The GARCH vol TARGET is derived ONCE here and FROZEN for the track's life
        (a moving target would defeat the comparison). It uses the CLIENT's
        raw-seed risky weights (`target_basis_weights`) — the client's risk
        fingerprint — NOT the LLM's post-rebalance mix.
        """
        basis = target_basis_weights or our_weights
        self._state["vol_target"] = self._derive_vol_target(
            basis, returns_for_garch, aligned_returns)
        books = {}
        books["our_model"] = {"nav": 1.0, "held_weights": dict(our_weights)}
        books["hold_initial"] = {"nav": 1.0, "held_weights": dict(our_weights)}
        # The full-conviction initial build, then held forever (fixed). Created
        # ONLY when hold_built_weights is supplied (the full-coverage run); the
        # focused run passes None and gets no hold_built book.
        if hold_built_weights:
            books["hold_built"] = {"nav": 1.0, "held_weights": dict(hold_built_weights)}
        books["garch"] = {"nav": 1.0,
                          "held_weights": self._garch_weights(our_weights, returns_for_garch)}
        # NB: the random buy-and-hold comparison CLOUD is intentionally NOT carried
        # here. It is a passive, deterministic object (fixed seed weights × price
        # history), so it is reconstructed OFFLINE on demand from the cached prices
        # (`scoreboard_run.py` → `random_strategies`), not marked nightly. Keeping
        # it out of the live state keeps nav_state.json small and lets the cloud
        # use any count (1,000+) without bloating the daily run.
        self._state["books"] = books

    def _derive_vol_target(self, weights: dict, returns_for_garch: dict | None,
                           aligned_returns=None) -> dict:
        """Per-portfolio vol target from the inception mix's Ulcer Index.

        Builds the book's historical return series over the COMMON-OVERLAP window
        (a date-aligned frame — back only as far as ALL holdings exist, bounded by
        the youngest), computes its Ulcer Index, and converts to an implied annual
        vol via the Brownian UI→vol bridge. Prefers the aligned frame; falls back
        to count-aligning the per-asset arrays, then to the default if no history.
        Stores the full breakdown + the window length so the target is auditable.
        """
        from agentic.quant.metrics import trend_ulcer_vol_target  # noqa: PLC0415

        # Risky sleeve = non-cash holdings with positive weight.
        risky_w = {t.upper(): w for t, w in weights.items()
                   if t.upper() != "CASH" and w > 0}
        if not risky_w:
            return {"vol_target": self.annual_vol_target, "source": "fallback_default"}

        port = None
        window_basis = None
        # Preferred: date-aligned common window across the held tickers.
        if aligned_returns is not None and not aligned_returns.empty:
            cols = [t for t in aligned_returns.columns if t.upper() in risky_w]
            if cols:
                sub = aligned_returns[cols].dropna(how="any")
                if len(sub) >= 60:
                    w = np.array([risky_w[c.upper()] for c in cols])
                    w = w / w.sum()
                    port = (sub.values @ w)
                    window_basis = "common_overlap"

        # Fallback: count-align the raw per-asset arrays (legacy behaviour).
        if port is None and returns_for_garch:
            avail = {t: returns_for_garch[t] for t in risky_w
                     if t in returns_for_garch}
            if avail:
                tot = sum(risky_w[t] for t in avail)
                wn = {t: risky_w[t] / tot for t in avail}
                n = min(len(avail[t]) for t in avail)
                port = sum(wn[t] * np.asarray(avail[t])[-n:] for t in avail)
                window_basis = "count_aligned"

        if port is None or len(port) < 60:
            return {"vol_target": self.annual_vol_target, "source": "fallback_default"}

        # Drift-neutral rolling trend-Ulcer → vol target (price built from returns).
        port = np.asarray(port)
        prices = 100.0 * np.cumprod(1.0 + port)
        res = trend_ulcer_vol_target(prices, port)
        res["window_basis"] = window_basis
        res["window_days"] = int(len(port))
        return res

    def _vol_target(self) -> float:
        vt = self._state.get("vol_target")
        if isinstance(vt, dict):
            return float(vt.get("vol_target", self.annual_vol_target))
        return float(vt) if vt is not None else self.annual_vol_target

    def _reweight(self, date: str, our_weights: dict,
                  returns_for_garch: dict | None) -> dict:
        """Re-pin the ACTIVE books to this week's fresh target, on top of the
        already-drifted book. Returns each active book's EXECUTION turnover
        (½Σ|target − drifted|) — the weight actually traded to correct drift.

        The passive books (hold_initial, hold_built, random_*) are NOT touched
        here; their drifted weights (set in the mark loop) simply carry forward,
        so they trade nothing and their execution turnover is 0.
        """
        from agentic.quant.metrics import execution_turnover  # noqa: PLC0415

        books = self._state["books"]
        # our_model tracks the fresh target; garch = fresh mix x vol scalar.
        our_target = dict(our_weights)
        garch_target = self._garch_weights(our_weights, returns_for_garch)
        exec_turnover = {
            "our_model": execution_turnover(
                [(books["our_model"]["held_weights"], our_target)]),
            "garch": execution_turnover(
                [(books["garch"]["held_weights"], garch_target)]),
        }
        books["our_model"]["held_weights"] = our_target
        books["garch"]["held_weights"] = garch_target
        return {k: round(v, 6) for k, v in exec_turnover.items()}

    def _garch_weights(self, base_weights: dict, returns_for_garch: dict | None) -> dict:
        """GARCH vol-target book = base mix scaled by the exposure scalar.

        Per Decision A the overlay only scales gross exposure; the cross-sectional
        mix stays = our model's. Needs per-asset return history; if unavailable
        (early track), fall back to the base mix unscaled.
        """
        if not returns_for_garch:
            return dict(base_weights)
        try:
            bm = GarchVolBenchmark(self._vol_target(), self.rebalance_period,
                                   self.cadence_factor)
            prior = self._state["books"].get("garch", {}).get("exposure_scalar", 1.0)
            dec = bm.step(base_weights, returns_for_garch, self.ticker_types,
                          self.constraints, prior_scalar=prior)
            self._state["books"].setdefault("garch", {})["exposure_scalar"] = dec.exposure_scalar
            return dec.target_weights
        except Exception as exc:  # noqa: BLE001
            print(f"[nav] WARNING: garch reweight failed ({exc}); holding base mix")
            return dict(base_weights)

