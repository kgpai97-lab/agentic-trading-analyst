"""Mark-to-market NAV tracking for the forward test.

The forward record stores DECISIONS (weights, signals). This module adds the
missing half — what actually happened to the money. Each weekly run marks the
book to market: it takes the weights that were ACTUALLY HELD over the elapsed
week and the realised price move of each holding, and compounds a NAV (net asset
value, indexed to 1.0 at inception). That is the honest equity curve — the live
track record the whole forward test exists to produce.

It also marks a set of comparator books in LOCKSTEP over the same week, same
prices, so our-model vs benchmark equity curves grow together and stay directly
comparable (Decision: lockstep, not after-the-fact replay):
  * our_model    — the weights we actually held.
  * garch        — our own mix scaled by the GARCH vol-target exposure scalar
                   (Decision A: overlay scales gross exposure, not the mix).
  * hold_initial — the inception weights, drifted, never retraded.
  * random_k     — random fixed-weight books seeded once at inception (stable).

Core functions are pure (take a returns dict); a thin helper fetches the realised
per-ticker returns between two dates via yfinance (point-in-time-safe: it asks
for closes up to each date).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def period_return(weights: dict, asset_returns: dict[str, float]) -> float:
    """Portfolio return over a period = Σ wᵢ·rᵢ for held assets.

    CASH (and any ticker missing from `asset_returns`) earns 0. Weights need not
    sum to 1 (a cash residual is implicit); we apply each weight to its asset's
    realised return. This is a constant-mix mark over the period.
    """
    r = 0.0
    for t, w in weights.items():
        if t.upper() == "CASH":
            continue
        ar = asset_returns.get(t)
        if ar is not None and np.isfinite(ar):
            r += w * float(ar)
    return r


def drift_weights(weights: dict, asset_returns: dict[str, float]) -> dict:
    """Evolve weights by realised returns over a period (buy-and-hold drift).

    A held book is NOT rebalanced between marks: each position's value grows by
    its own return, so its weight floats — winners' weights rise, losers' fall:

        wᵢ' = wᵢ(1+rᵢ) / Σⱼ wⱼ(1+rⱼ)

    CASH (and any ticker missing from `asset_returns`) earns 0 → it grows by a
    factor of 1, so its weight still shifts relative to risky names that moved.
    A residual (weights not summing to 1) is preserved as an implicit cash
    sleeve that also earns 0. Returns a new dict; the input is not mutated.
    """
    grown = {}
    total = 0.0
    for t, w in weights.items():
        if t.upper() == "CASH":
            factor = 1.0
        else:
            ar = asset_returns.get(t)
            factor = 1.0 + float(ar) if ar is not None and np.isfinite(ar) else 1.0
        g = w * factor
        grown[t] = g
        total += g
    if total <= 0:
        return dict(weights)
    return {t: g / total for t, g in grown.items()}


def fetch_asset_returns(tickers: list[str], prev_date: str, this_date: str) -> dict[str, float]:
    """Realised total return of each ticker from prev_date to this_date.

    Uses the last adjusted close on/before each date (so a non-trading as-of date
    still resolves to the prior session). CASH and unresolved tickers are omitted
    (treated as 0 return by `period_return`).
    """
    import yfinance as yf  # noqa: PLC0415

    syms = [t for t in tickers if t.upper() != "CASH"]
    if not syms:
        return {}
    # Pad the start so we always have a close on/before prev_date.
    start = (pd.Timestamp(prev_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(this_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(syms, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        return {}
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else \
        raw[["Close"]].rename(columns={"Close": syms[0]})

    out: dict[str, float] = {}
    pts, tts = pd.Timestamp(prev_date), pd.Timestamp(this_date)
    for t in syms:
        if t not in close.columns:
            continue
        s = close[t].dropna()
        prev = s[s.index <= pts]
        cur = s[s.index <= tts]
        if len(prev) and len(cur) and prev.iloc[-1] > 0:
            out[t] = float(cur.iloc[-1] / prev.iloc[-1] - 1.0)
    return out


@dataclass
class NavBook:
    """One strategy's compounding NAV over the forward track."""

    name: str
    nav: float = 1.0
    history: list[dict] = field(default_factory=list)   # [{date, ret, nav}, ...]

    def mark(self, date: str, weights: dict, asset_returns: dict[str, float]) -> float:
        """Apply the period return for `weights` and append a NAV point."""
        ret = period_return(weights, asset_returns)
        self.nav *= (1.0 + ret)
        self.history.append({"date": date, "ret": round(ret, 6), "nav": round(self.nav, 6)})
        return ret
