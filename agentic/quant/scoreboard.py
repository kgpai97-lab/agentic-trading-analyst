"""Strategy scoreboard — run every comparator on the same data and rank them.

Phase 4c. The whole point of the project's "does any of this work?" question:
put the LLM desk's portfolio next to dumb baselines on the SAME universe, dates,
cadence, and hard constraints, and report return AND risk side by side.

Comparators (per the locked plan):
  * our_model               — the LLM desk's rebalances (from the forward record).
  * garch                   — the GARCH vol-target benchmark (Phase 4a).
  * hold_initial            — the initial allocation, never rebalanced (the
                              "does rebalancing add anything?" baseline — the most
                              damning loss).
  * random                  — a distribution of random portfolios → our Sharpe as
                              a percentile + DEFLATED Sharpe ("better than luck?").

A strategy is a time series of (date, target_weights). The simulator turns that
+ price history into a realised period-return series; the metrics module scores
it. This module is pure numpy/pandas — it does NOT call the LLM. It runs over
the forward records now; the deferred backtest (Phase 8) reuses it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agentic.quant import metrics


@dataclass
class Strategy:
    """A comparator: an ordered list of (date_str, weights) rebalances."""

    name: str
    rebalances: list[tuple[str, dict]]   # [(YYYY-MM-DD, {ticker: weight}), ...]

    @property
    def weight_series(self) -> list[dict]:
        return [w for _, w in self.rebalances]


def simulate_returns(
    rebalances: list[tuple[str, dict]],
    prices: pd.DataFrame,
) -> np.ndarray:
    """Realised daily return series for a weight series held between rebalances.

    prices: a DataFrame indexed by date (DatetimeIndex), columns = tickers,
    values = adjusted close. Between consecutive rebalance dates the weights are
    held FIXED at the most recent target (the portfolio drifts with prices; we
    apply the *target* weights to each day's asset returns — a standard
    constant-mix approximation that ignores intra-period drift, which is fine for
    comparing decision quality across strategies on the same data).

    CASH (and any ticker absent from `prices`) earns zero return.
    """
    if not rebalances or prices.empty:
        return np.array([])

    daily = prices.sort_index().pct_change().dropna(how="all")
    if daily.empty:
        return np.array([])

    reb_dates = [pd.Timestamp(d) for d, _ in rebalances]
    reb_weights = [w for _, w in rebalances]

    port_rets = []
    for day, row in daily.iterrows():
        # Pick the most recent rebalance on/before this day.
        idx = None
        for i, rd in enumerate(reb_dates):
            if rd <= day:
                idx = i
            else:
                break
        if idx is None:
            continue  # before the first rebalance — not invested yet
        w = reb_weights[idx]
        r = 0.0
        for ticker, weight in w.items():
            if ticker.upper() == "CASH" or ticker not in row.index:
                continue
            ar = row[ticker]
            if pd.notna(ar):
                r += weight * float(ar)
        port_rets.append(r)
    return np.asarray(port_rets, dtype=float)


def hold_initial_strategy(initial_weights: dict, dates: list[str]) -> Strategy:
    """The 'never rebalance' baseline: one rebalance at t0, nothing after."""
    return Strategy("hold_initial", [(dates[0], dict(initial_weights))])


def random_strategies(
    tickers: list[str],
    dates: list[str],
    n: int = 1000,
    seed: int = 0,
    long_only: bool = True,
) -> list[Strategy]:
    """n random fixed-weight portfolios over `tickers` (held all period).

    Dirichlet(1) weights → uniform over the simplex (long-only, sums to 1).
    Each is a single-rebalance strategy (buy-and-hold a random mix).
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        w = rng.dirichlet(np.ones(len(tickers)))
        weights = {t: float(wi) for t, wi in zip(tickers, w)}
        out.append(Strategy(f"random_{i}", [(dates[0], weights)]))
    return out


def efficient_frontier(
    tickers: list[str],
    prices: pd.DataFrame,
    n_points: int = 40,
    periods_per_year: int = 252,
) -> dict:
    """Long-only mean-variance efficient frontier from the price history.

    Companion to the random CLOUD: the cloud samples the simplex, the frontier is
    the OPTIMAL edge of it. Estimates μ (mean daily return) and Σ (covariance)
    from `prices` over `tickers`, then solves, for a grid of target returns r*:

        min_w  wᵀΣw   s.t.  Σw = 1,  w ≥ 0,  μᵀw = r*

    (a long-only QP via scipy SLSQP). Returns the frontier as parallel
    annualised-vol / annualised-return arrays plus the TANGENCY (max-Sharpe)
    portfolio — the single point we hold our_model up against. CASH is excluded
    (it's the risk-free leg); a residual is the investor's own cash choice.
    Returns {} if fewer than 2 priced tickers (no frontier to speak of).
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    syms = [t for t in tickers if t.upper() != "CASH" and t in prices.columns]
    if len(syms) < 2:
        return {}
    daily = prices[syms].sort_index().pct_change().dropna(how="any")
    if len(daily) < len(syms) + 2:        # too few obs to estimate Σ stably
        return {}

    mu = daily.mean().values              # daily expected return
    cov = daily.cov().values
    n = len(syms)
    ann = float(periods_per_year)

    def port_vol(w):
        return float(np.sqrt(max(w @ cov @ w, 0.0)))

    bounds = [(0.0, 1.0)] * n
    sum_to_one = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    w0 = np.full(n, 1.0 / n)

    targets = np.linspace(float(mu.min()), float(mu.max()), n_points)
    vols, rets, wts = [], [], []
    for r_star in targets:
        cons = [sum_to_one,
                {"type": "eq", "fun": lambda w, r=r_star: float(mu @ w) - r}]
        res = minimize(port_vol, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
        if not res.success:
            continue
        w = res.x
        vols.append(port_vol(w) * np.sqrt(ann))
        rets.append(float(mu @ w) * ann)
        wts.append({t: float(wi) for t, wi in zip(syms, w)})
    if not vols:
        return {}

    vols, rets = np.array(vols), np.array(rets)
    sharpes = np.divide(rets, vols, out=np.zeros_like(rets), where=vols > 0)
    k = int(np.argmax(sharpes))
    return {
        "tickers": syms,
        "vol": vols, "ret": rets,             # annualised, frontier-ordered
        "tangency_weights": wts[k],
        "tangency_sharpe": float(sharpes[k]),
        "tangency_vol": float(vols[k]),
        "tangency_ret": float(rets[k]),
        "n_points": len(vols),
    }


@dataclass
class Scoreboard:
    periods_per_year: int = 252
    rows: dict = field(default_factory=dict)        # name -> metrics dict
    extras: dict = field(default_factory=dict)      # name -> extra fields

    def add(self, strat: Strategy, prices: pd.DataFrame) -> None:
        rets = simulate_returns(strat.rebalances, prices)
        m = metrics.all_metrics(rets, self.periods_per_year, strat.weight_series)
        m["n_periods"] = int(rets.size)
        self.rows[strat.name] = m
        self._returns_cache = getattr(self, "_returns_cache", {})
        self._returns_cache[strat.name] = rets

    def add_random_distribution(
        self, randoms: list[Strategy], prices: pd.DataFrame, our_name: str = "our_model"
    ) -> None:
        """Score the random distribution and attach percentile + deflated Sharpe
        for our model vs. the random Sharpe distribution."""
        sharpes = []
        for s in randoms:
            rets = simulate_returns(s.rebalances, prices)
            if rets.size:
                sharpes.append(metrics.sharpe(rets, self.periods_per_year))
        sharpes = np.array(sharpes)
        if sharpes.size == 0 or our_name not in self.rows:
            return
        our_sharpe = self.rows[our_name]["sharpe"]
        pct = float((sharpes < our_sharpe).mean() * 100.0)
        our_rets = getattr(self, "_returns_cache", {}).get(our_name, np.array([]))
        dsr = metrics.deflated_sharpe(our_sharpe, our_rets, n_trials=len(randoms),
                                      periods_per_year=self.periods_per_year)
        # Add a summary row for the random distribution (median strategy).
        self.rows["random_median"] = {
            "cagr": float("nan"), "annual_vol": float("nan"),
            "sharpe": float(np.median(sharpes)), "sortino": float("nan"),
            "max_drawdown": float("nan"), "calmar": float("nan"),
            "n_periods": int(our_rets.size),
        }
        self.extras[our_name] = {
            "sharpe_percentile_of_random": pct,
            "deflated_sharpe": dsr,
            "n_random": len(randoms),
        }

    def add_efficient_frontier(
        self, tickers: list[str], prices: pd.DataFrame, our_name: str = "our_model"
    ) -> None:
        """Attach the mean-variance frontier's tangency portfolio as a reference
        point and record where our_model's Sharpe sits relative to it."""
        fr = efficient_frontier(tickers, prices, periods_per_year=self.periods_per_year)
        if not fr or our_name not in self.rows:
            return
        our_sharpe = self.rows[our_name]["sharpe"]
        self.rows["frontier_tangency"] = {
            "cagr": float("nan"), "annual_vol": fr["tangency_vol"],
            "sharpe": fr["tangency_sharpe"], "sortino": float("nan"),
            "max_drawdown": float("nan"), "calmar": float("nan"),
            "n_periods": int(self.rows[our_name].get("n_periods", 0)),
        }
        self.extras.setdefault(our_name, {})
        self.extras[our_name].update({
            "frontier_tangency_sharpe": fr["tangency_sharpe"],
            "sharpe_gap_to_frontier": our_sharpe - fr["tangency_sharpe"],
            "frontier_n_points": fr["n_points"],
        })

    def to_table(self) -> str:
        """Render a markdown metrics table, our_model first."""
        cols = ["cagr", "annual_vol", "sharpe", "sortino", "max_drawdown",
                "calmar", "turnover", "n_periods"]
        order = sorted(self.rows, key=lambda n: (n != "our_model", n))
        head = "| strategy | " + " | ".join(cols) + " |"
        sep = "|" + "---|" * (len(cols) + 1)
        lines = [head, sep]
        for name in order:
            m = self.rows[name]
            cells = []
            for c in cols:
                v = m.get(c)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    cells.append("—")
                elif c in ("cagr", "annual_vol", "max_drawdown"):
                    cells.append(f"{v:.2%}")
                elif c == "n_periods":
                    cells.append(str(int(v)))
                else:
                    cells.append(f"{v:.3f}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        out = "\n".join(lines)
        for name, ex in self.extras.items():
            if "sharpe_percentile_of_random" in ex:
                out += (f"\n\n**{name}** vs random: Sharpe at the "
                        f"{ex['sharpe_percentile_of_random']:.1f}th percentile of "
                        f"{ex['n_random']} random portfolios; deflated Sharpe "
                        f"(P[skill]) = {ex['deflated_sharpe']:.3f}.")
            if "frontier_tangency_sharpe" in ex:
                gap = ex["sharpe_gap_to_frontier"]
                out += (f"\n\n**{name}** vs efficient frontier: tangency "
                        f"(max-Sharpe) portfolio Sharpe = "
                        f"{ex['frontier_tangency_sharpe']:.3f} over "
                        f"{ex['frontier_n_points']} frontier points; our Sharpe is "
                        f"{gap:+.3f} {'above' if gap >= 0 else 'below'} it.")
        return out
