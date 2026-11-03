"""Incremental, shared on-disk return cache for the GARCH layer.

Motivation: the forward test runs 15 tracks (3 portfolios × 5 weekdays), and the
same tickers recur across portfolios (SPY in diversified AND concentrated, etc.).
Re-downloading ~400 days of history for every ticker, every track, every night
is wasteful. This cache fetches each ticker's price history ONCE, then on each
subsequent run pulls only the NEW days since `last_date` and APPENDS them — so as
the forward test walks through time the cache grows a week at a time instead of
re-pulling the whole window.

Design (per the locked decisions):
  * Stored per ticker at outputs/garch_cache/<TICKER>.json as
    {ticker, last_date, closes: {YYYY-MM-DD: adj_close}}. Closes (not returns)
    are cached so a new week can be chained onto the prior close exactly.
  * SHARED project-wide: one cache dir, reused across all 15 tracks and runs.
  * `get_returns(ticker, as_of)` returns the log-return series up to `as_of`,
    fetching only (last_date, as_of] when warm. POINT-IN-TIME: never returns a
    close dated after `as_of`.
  * The GARCH PARAMETERS are NOT cached — callers re-fit the full MLE off the
    cached series each run (cheap; keeps params at the exact current MLE). The
    cache eliminates the re-DOWNLOAD, which is the expensive part.

Concurrency note: the nightly job is sequential (one process), so plain file
read/write is fine. If tracks ever run in parallel, add a per-ticker lock.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_DIR = Path("outputs/garch_cache")
# ~15 calendar years on first sight of a ticker (capped naturally by the
# asset's own inception — yfinance just returns from listing). 15y spans the
# 2018/2020/2022 stress regimes for the older assets; young assets (BITO ~4.6y,
# PLTR ~5.7y) return their full available history. GARCH fits each asset on its
# OWN full series; the portfolio Ulcer target uses the common-overlap window.
_BOOTSTRAP_DAYS = 365 * 15 + 4
_MIN_RETURNS = 30             # below this, GARCH callers fall back to sample vol


class ReturnCache:
    """Shared on-disk cache of adjusted closes, fetched incrementally."""

    def __init__(self, cache_dir: str | Path = _DEFAULT_DIR):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---- storage ----------------------------------------------------------

    def _path(self, ticker: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in ticker.upper())
        return self.dir / f"{safe}.json"

    def _load(self, ticker: str) -> dict:
        p = self._path(ticker)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"ticker": ticker.upper(), "last_date": None, "closes": {}}

    def _save(self, ticker: str, rec: dict) -> None:
        self._path(ticker).write_text(json.dumps(rec), encoding="utf-8")

    # ---- fetch ------------------------------------------------------------

    @staticmethod
    def _download_closes(ticker: str, start: str, end: str) -> dict[str, float]:
        """Adjusted closes for [start, end) as {YYYY-MM-DD: close}. Network."""
        import yfinance as yf  # noqa: PLC0415

        raw = yf.Ticker(ticker).history(start=start, end=end, interval="1d",
                                        auto_adjust=True)
        if raw.empty or "Close" not in raw:
            return {}
        out = {}
        for idx, val in raw["Close"].dropna().items():
            out[pd.Timestamp(idx).strftime("%Y-%m-%d")] = float(val)
        return out

    def update(self, ticker: str, as_of_date: str) -> dict:
        """Ensure the cache holds closes up to `as_of_date`; fetch only the gap.

        Returns the (in-memory) record. Cold cache → bootstrap ~800 days ending
        at as_of. Warm cache → fetch only (last_date, as_of] and append.
        """
        rec = self._load(ticker)
        as_of = pd.Timestamp(as_of_date)
        end_excl = (as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        last = rec.get("last_date")
        if last is None:
            start = (as_of - pd.Timedelta(days=_BOOTSTRAP_DAYS)).strftime("%Y-%m-%d")
            new = self._download_closes(ticker, start, end_excl)
        elif pd.Timestamp(last) >= as_of:
            return rec   # already current through as_of — nothing to fetch
        else:
            # incremental: only the days after the last cached close
            start = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            new = self._download_closes(ticker, start, end_excl)

        if new:
            rec["closes"].update(new)
            rec["last_date"] = max(rec["closes"])
            self._save(ticker, rec)
        # else: nothing fetched (unknown ticker / no new data) — don't write an
        # empty file. A warm cache with no new data is already persisted.
        return rec

    # ---- consumption ------------------------------------------------------

    def get_returns(self, ticker: str, as_of_date: str,
                    update: bool = True) -> np.ndarray:
        """Log-return series for `ticker` up to `as_of_date` (point-in-time).

        `update=True` tops the cache up first (incremental fetch). Returns an
        empty array if too little history exists.
        """
        rec = self.update(ticker, as_of_date) if update else self._load(ticker)
        closes = rec.get("closes", {})
        if not closes:
            return np.array([])
        s = pd.Series(closes)
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        s = s[s.index <= pd.Timestamp(as_of_date)]   # never peek past as_of
        if s.size < 2:
            return np.array([])
        return np.log(s / s.shift(1)).dropna().values

    def returns_for(self, tickers, as_of_date: str, update: bool = True) -> dict:
        """{ticker: return_array} for the non-CASH tickers with enough history.

        Each array is that ticker's FULL cached history up to `as_of` — used for
        per-asset GARCH fits (which want max data per asset). For a date-ALIGNED
        common window (the portfolio Ulcer target), use `aligned_returns`.
        """
        out = {}
        for t in tickers:
            if str(t).upper() == "CASH":
                continue
            r = self.get_returns(t, as_of_date, update=update)
            if r.size >= _MIN_RETURNS:
                out[t] = r
        return out

    def aligned_returns(self, tickers, as_of_date: str,
                        update: bool = True) -> pd.DataFrame:
        """Date-aligned daily returns (inner-join on dates) for the non-CASH set.

        This is the COMMON-OVERLAP window — it extends back only as far as ALL
        requested tickers have data (bounded by the youngest holding, e.g. BITO
        ~4.6y). Used for the portfolio Ulcer/vol target so every holding's return
        is measured over the same dates. Columns = tickers, index = dates.
        """
        cols = {}
        for t in tickers:
            if str(t).upper() == "CASH":
                continue
            rec = self.update(t, as_of_date) if update else self._load(t)
            closes = rec.get("closes", {})
            if not closes:
                continue
            s = pd.Series(closes)
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            s = s[s.index <= pd.Timestamp(as_of_date)]
            if s.size >= 2:
                cols[t] = np.log(s / s.shift(1)).dropna()
        if not cols:
            return pd.DataFrame()
        # inner join → common dates across ALL holdings
        return pd.DataFrame(cols).dropna(how="any")
