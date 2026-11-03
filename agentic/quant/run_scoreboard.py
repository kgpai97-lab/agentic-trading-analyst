"""Glue: build the strategy scoreboard from a forward-test track's records.

Reads a weekday track's history.jsonl (our model's rebalances), fetches price
history for the union of held tickers, builds the comparators (our model, GARCH
benchmark, hold-initial-no-rebalance, a random buy-and-hold distribution, and the
mean-variance efficient-frontier tangency portfolio) on the SAME
dates/universe/constraints, and renders the metrics table.

The random distribution + frontier are OFFLINE-only (this script), reconstructed
from prices on demand — they are deliberately NOT carried in the nightly NAV run.

Runs over the forward records (Phase 3) once a track has >= 2 rebalances; the
deferred backtest (Phase 8) can reuse `build_scoreboard` with its own rebalance
list and price frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agentic.dataflows.constraints import PortfolioConstraints
from agentic.quant.benchmark import GarchVolBenchmark
from agentic.quant.scoreboard import (
    Scoreboard,
    Strategy,
    hold_initial_strategy,
    random_strategies,
)

_PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}


def load_track_rebalances(history_path: str | Path) -> list[tuple[str, dict]]:
    """Read [(date, target_weights), ...] from a forward track's history.jsonl."""
    out = []
    for line in Path(history_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        tw = rec.get("target_weights") or {}
        if tw:
            out.append((rec["date"], tw))
    out.sort(key=lambda x: x[0])
    return out


def fetch_price_frame(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Adjusted-close DataFrame (dates × tickers) over [start, end]. CASH dropped."""
    import yfinance as yf  # noqa: PLC0415

    syms = [t for t in tickers if t.upper() != "CASH"]
    if not syms:
        return pd.DataFrame()
    raw = yf.download(syms, start=start, end=end, progress=False, auto_adjust=True)
    # yfinance returns a column MultiIndex for multiple tickers.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": syms[0]})
    return close.dropna(how="all")


def build_scoreboard(
    rebalances: list[tuple[str, dict]],
    prices: pd.DataFrame,
    *,
    rebalance_period: str = "weekly",
    constraints: PortfolioConstraints | None = None,
    annual_vol_target: float = 0.10,
    cadence_factor: float = 0.1,
    n_random: int = 1000,
    ticker_types: dict | None = None,
) -> Scoreboard:
    """Assemble all four comparators and score them on the same prices."""
    constraints = constraints or PortfolioConstraints()
    ppy = _PERIODS_PER_YEAR.get(rebalance_period, 52)
    # Scoreboard works on the SIMULATED daily series; metrics annualise with 252.
    sb = Scoreboard(periods_per_year=252)

    dates = [d for d, _ in rebalances]
    tickers = sorted({t for _, w in rebalances for t in w})

    # 1) our model
    sb.add(Strategy("our_model", rebalances), prices)

    # 2) GARCH benchmark — same dates, same base mix per date, scaled by vol target.
    bm = GarchVolBenchmark(annual_vol_target, rebalance_period, cadence_factor)
    types = ticker_types or {t: ("cash" if t.upper() == "CASH" else "equity") for t in tickers}
    bench_rebs = []
    prior = 1.0
    daily = prices.sort_index().pct_change()
    for d, w in rebalances:
        # return history up to date d for each held ticker
        hist = daily.loc[daily.index <= pd.Timestamp(d)]
        rets = {t: hist[t].dropna().values for t in w
                if t.upper() != "CASH" and t in hist.columns}
        if not rets:
            bench_rebs.append((d, dict(w)))
            continue
        dec = bm.step(w, rets, types, constraints, prior_scalar=prior)
        prior = dec.exposure_scalar
        bench_rebs.append((d, dec.target_weights))
    sb.add(Strategy("garch", bench_rebs), prices)

    # 3) hold-initial-no-rebalance
    sb.add(hold_initial_strategy(rebalances[0][1], dates), prices)

    # 4) random distribution (buy-and-hold random mixes of the same tickers)
    randoms = random_strategies([t for t in tickers if t.upper() != "CASH"],
                                dates, n=n_random)
    sb.add_random_distribution(randoms, prices, our_name="our_model")

    # 5) mean-variance efficient frontier — the optimal edge of the random cloud;
    #    its tangency (max-Sharpe) portfolio is the reference our_model is held to.
    sb.add_efficient_frontier([t for t in tickers if t.upper() != "CASH"],
                              prices, our_name="our_model")
    return sb


def scoreboard_for_track(
    track_dir: str | Path,
    *,
    rebalance_period: str = "weekly",
    n_random: int = 1000,
) -> str:
    """End-to-end: read a weekday track and return the rendered table.

    Raises if the track has < 2 rebalances (nothing to compare yet).
    """
    track = Path(track_dir)
    rebs = load_track_rebalances(track / "history.jsonl")
    if len(rebs) < 2:
        raise ValueError(
            f"track {track} has {len(rebs)} rebalance(s); need >= 2 to score."
        )
    tickers = sorted({t for _, w in rebs for t in w})
    start = rebs[0][0]
    end = (pd.Timestamp(rebs[-1][0]) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    prices = fetch_price_frame(tickers, start, end)
    if prices.empty:
        raise ValueError("no price history fetched for the track's tickers.")

    # Recover ticker types from the seed portfolio if present.
    types = None
    seed = track / "portfolio.json"
    if seed.exists():
        data = json.loads(seed.read_text(encoding="utf-8"))
        types = {t.upper(): v.get("type", "equity") if isinstance(v, dict) else "equity"
                 for t, v in data.get("holdings", {}).items()}
        types.setdefault("CASH", "cash")
        constraints = PortfolioConstraints.from_dict(data.get("constraints"))
    else:
        constraints = PortfolioConstraints()

    sb = build_scoreboard(rebs, prices, rebalance_period=rebalance_period,
                          constraints=constraints, n_random=n_random,
                          ticker_types=types)
    return sb.to_table()
