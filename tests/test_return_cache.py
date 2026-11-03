"""Incremental return-cache tests (downloader mocked — no network).

The key correctness property: building the cache INCREMENTALLY (bootstrap, then
append a week at a time) yields the SAME return series as one full fetch — and
the cache never returns a close dated after `as_of` (point-in-time).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agentic.quant import return_cache as rc_mod
from agentic.quant.return_cache import ReturnCache

# A synthetic daily close series (business days) we can slice deterministically.
_ALL = pd.Series(
    100 * np.cumprod(1 + 0.001 * np.ones(120)),
    index=pd.bdate_range("2026-01-01", periods=120),
)


def _fake_download(ticker, start, end):
    """Return cached-shape {date: close} for [start, end) from the synthetic series."""
    s = _ALL[(_ALL.index >= pd.Timestamp(start)) & (_ALL.index < pd.Timestamp(end))]
    return {d.strftime("%Y-%m-%d"): float(v) for d, v in s.items()}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ReturnCache, "_download_closes", staticmethod(_fake_download))
    return ReturnCache(cache_dir=tmp_path / "garch_cache")


def _full_returns_upto(as_of):
    s = _ALL[_ALL.index <= pd.Timestamp(as_of)]
    return np.log(s / s.shift(1)).dropna().values


def test_cold_fetch_matches_full_history(cache):
    got = cache.get_returns("SPY", "2026-03-01")
    exp = _full_returns_upto("2026-03-01")
    assert np.allclose(got, exp)


def test_incremental_equals_full(cache):
    # Build the cache a week at a time, then compare to a single full fetch.
    for d in ["2026-02-01", "2026-02-08", "2026-02-15", "2026-02-22", "2026-03-01"]:
        cache.get_returns("SPY", d)
    incremental = cache.get_returns("SPY", "2026-03-01")
    full = _full_returns_upto("2026-03-01")
    assert np.allclose(incremental, full)


def test_incremental_only_fetches_the_gap(cache, monkeypatch):
    cache.get_returns("SPY", "2026-02-15")          # bootstrap
    last_cached = cache._load("SPY")["last_date"]   # last business day <= 02-15
    calls = {"ranges": []}
    orig = cache._download_closes

    def spy_download(ticker, start, end):
        calls["ranges"].append((start, end))
        return orig(ticker, start, end)

    monkeypatch.setattr(cache, "_download_closes", spy_download)
    cache.get_returns("SPY", "2026-02-22")          # warm: only the new gap
    assert len(calls["ranges"]) == 1
    start, _ = calls["ranges"][0]
    # incremental fetch starts the day AFTER the last cached close (not from
    # scratch, and not before what we already have)
    assert start > last_cached


def test_point_in_time_never_returns_future(cache):
    # Warm the cache far forward, then ask for an earlier as_of: must not leak.
    cache.get_returns("SPY", "2026-04-01")
    early = cache.get_returns("SPY", "2026-02-15", update=False)
    full_early = _full_returns_upto("2026-02-15")
    assert np.allclose(early, full_early)
    assert early.size < cache.get_returns("SPY", "2026-04-01", update=False).size


def test_returns_for_skips_cash_and_short(cache):
    out = cache.returns_for(["SPY", "CASH", "BND"], "2026-03-01")
    assert "CASH" not in out
    assert "SPY" in out and "BND" in out


def test_cache_persists_to_disk(cache, tmp_path):
    cache.get_returns("SPY", "2026-03-01")
    f = tmp_path / "garch_cache" / "SPY.json"
    assert f.exists()
    rec = json.loads(f.read_text())
    assert rec["ticker"] == "SPY" and rec["last_date"] <= "2026-03-01"
    assert len(rec["closes"]) > 30
