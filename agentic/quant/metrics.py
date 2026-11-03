"""Performance & risk metrics — pure functions over a return series.

All take a 1-D array of PERIOD returns (decimal, e.g. daily) and the number of
periods per year for annualisation. Kept dependency-free (numpy only) and
separate from the portfolio simulator so they are trivially unit-testable
against closed-form cases.

Deflated Sharpe (Bailey & López de Prado 2014) is included because the
scoreboard tests "better than luck?" against many random portfolios — the more
strategies you try, the higher the max Sharpe you'd expect by chance, so a raw
Sharpe percentile overstates skill. The deflated Sharpe adjusts the observed
Sharpe for the number of trials and the return distribution's skew/kurtosis.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

_EPS = 1e-12


def cagr(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    growth = float(np.prod(1.0 + r))
    if growth <= 0:
        return -1.0
    years = r.size / periods_per_year
    return growth ** (1.0 / years) - 1.0 if years > 0 else 0.0


def annual_vol(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year)) if r.size > 1 else 0.0


def sharpe(returns: np.ndarray, periods_per_year: int = 252, rf: float = 0.0) -> float:
    """Annualised Sharpe. `rf` is the per-period risk-free rate (decimal)."""
    r = np.asarray(returns, dtype=float) - rf
    sd = np.std(r, ddof=1)
    if sd < _EPS or r.size < 2:
        return 0.0
    return float(np.mean(r) / sd * np.sqrt(periods_per_year))


def sortino(returns: np.ndarray, periods_per_year: int = 252, rf: float = 0.0) -> float:
    """Annualised Sortino — like Sharpe but penalises only downside deviation."""
    r = np.asarray(returns, dtype=float) - rf
    downside = r[r < 0]
    dd = np.sqrt(np.mean(downside ** 2)) if downside.size else 0.0
    if dd < _EPS:
        return 0.0
    return float(np.mean(r) / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    """Worst peak-to-trough decline of the cumulative-return curve (negative)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    dd = curve / peak - 1.0
    return float(dd.min())


def calmar(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """CAGR / |max drawdown|."""
    mdd = max_drawdown(returns)
    if abs(mdd) < _EPS:
        return 0.0
    return cagr(returns, periods_per_year) / abs(mdd)


# ---------------------------------------------------------------------------
# Ulcer Index → implied volatility target  (per-portfolio risk anchor)
#
# The GARCH overlay needs a per-portfolio vol TARGET. Standard vol is symmetric
# and memoryless; the Ulcer Index measures DOWNSIDE pain with memory (depth ×
# duration underwater). We compute each book's historical UI, then convert it to
# a volatility-equivalent so the GARCH scalar math (target_vol / forecast_vol)
# still works. UI defines WHERE we want to be; GARCH manages HOW we get there —
# the two operate independently (UI plays no further role after setting target).
#
# ⚠️ Assumptions (flag these wherever the target is reported):
#   • The UI→vol constant comes from Brownian-motion theory (see below) and is
#     APPROXIMATE; for fat-tailed real returns it is nudged up via `fat_tail_adj`.
#   • The history window (e.g. ~800d) may be unusually calm/stressed → the target
#     inherits that bias. It is BACKWARD-LOOKING and STATIC (set once, frozen).
#   • The drawdown-duration T must be defined consistently (see below).
# ---------------------------------------------------------------------------

def _drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Fractional drawdown D_t = price/peak − 1 (≤ 0) of the cumulative curve."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return np.array([])
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    return curve / peak - 1.0


def ulcer_index(returns: np.ndarray) -> float:
    """Ulcer Index = RMS of percentage drawdowns from the running peak.

        UI = sqrt( mean( D_t² ) ),  D_t in PERCENT (so a 10% drawdown = 10).

    Only nonzero when underwater; squaring penalises deep drawdowns more than
    shallow ones. Returned in percentage points (matching the UI→vol formula).
    """
    dd = _drawdown_series(returns)
    if dd.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((dd * 100.0) ** 2)))


def avg_drawdown_duration(returns: np.ndarray) -> float:
    """Average length (in periods) of underwater stretches.

    T for the UI→vol conversion. We define a drawdown episode as a maximal run
    of consecutive periods with D_t < 0 (strictly below the running peak), and
    return the mean run length. If the series never goes underwater, returns 1.0
    (a floor, so the √T conversion is well-defined).
    """
    dd = _drawdown_series(returns)
    if dd.size == 0:
        return 1.0
    underwater = dd < -_EPS
    runs, cur = [], 0
    for u in underwater:
        if u:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return float(np.mean(runs)) if runs else 1.0


def _price_sq_weighted_ma(prices: np.ndarray, window: int) -> np.ndarray:
    """Trailing price²-weighted moving average: Σ P_i² / Σ P_i over the window.

    At each t, the trend is the price-weighted mean of the trailing `window`
    prices — bigger prices dominate, so it tracks the level (and rises with an
    up-trending market) rather than lagging like a simple MA. The first `window`
    points use an expanding trailing window (so early data isn't dropped).
    Returns an array the same length as `prices`.
    """
    p = np.asarray(prices, dtype=float)
    n = p.size
    out = np.empty(n)
    csum_p = np.cumsum(p)
    csum_p2 = np.cumsum(p * p)
    for t in range(n):
        lo = max(0, t - window + 1)
        sp = csum_p[t] - (csum_p[lo - 1] if lo > 0 else 0.0)
        sp2 = csum_p2[t] - (csum_p2[lo - 1] if lo > 0 else 0.0)
        out[t] = sp2 / sp if sp > _EPS else p[t]
    return out


def trend_relative_drawdown(prices: np.ndarray, trend_window: int) -> np.ndarray:
    """Drawdown of price BELOW its trailing price²-weighted trend (≤ 0).

    D_t = min( (P_t − WMA_t) / WMA_t , 0 ). Drift-neutral: because the trend
    rises with the market, this measures dips relative to trend, not absolute
    peak-to-trough — so a long, up-trending window does NOT inflate the Ulcer the
    way peak-based drawdowns do.
    """
    p = np.asarray(prices, dtype=float)
    if p.size == 0:
        return np.array([])
    wma = _price_sq_weighted_ma(p, trend_window)
    dev = (p - wma) / np.where(wma > _EPS, wma, 1.0)
    return np.minimum(dev, 0.0)


def trend_ulcer_index(prices: np.ndarray, trend_window: int) -> float:
    """Ulcer Index on TREND-relative drawdowns (RMS, in percent)."""
    dd = trend_relative_drawdown(prices, trend_window)
    if dd.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((dd * 100.0) ** 2)))


def rolling_trend_ulcer(
    prices: np.ndarray,
    periods_per_year: int = 252,
    trend_years: float = 2.5,
    roll_years: float = 4.0,
) -> dict:
    """Average of rolling-window trend-relative Ulcer indices over a price series.

    Per the spec: at each point the trend is a trailing `trend_years` price²-
    weighted MA; drawdowns are measured below that trend; the Ulcer Index is
    computed on each rolling `roll_years` window; the AVERAGE rolling UI is the
    target basis. Captures how the book's trend-relative drawdown character
    EVOLVED across the (e.g. 15-year) history, not one window's snapshot.

    Returns {avg_ulcer, rolling_uis, n_windows, trend_window, roll_window}.
    Falls back to a single whole-sample trend-Ulcer if the series is shorter
    than one rolling window.
    """
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p)]
    trend_w = int(trend_years * periods_per_year)
    roll_w = int(roll_years * periods_per_year)
    if p.size < roll_w:
        ui = trend_ulcer_index(p, trend_w)
        return {"avg_ulcer": ui, "rolling_uis": [round(ui, 4)], "n_windows": 1,
                "trend_window": trend_w, "roll_window": roll_w, "basis": "single_window"}

    # Slide the 4y window across the series; step ~monthly to bound cost.
    step = max(1, periods_per_year // 12)
    uis = []
    for end in range(roll_w, p.size + 1, step):
        win = p[end - roll_w:end]
        uis.append(trend_ulcer_index(win, trend_w))
    avg = float(np.mean(uis)) if uis else 0.0
    return {"avg_ulcer": avg, "rolling_uis": [round(u, 4) for u in uis],
            "n_windows": len(uis), "trend_window": trend_w, "roll_window": roll_w,
            "basis": "rolling"}


def implied_vol_target_from_ulcer(
    returns: np.ndarray,
    periods_per_year: int = 252,
    fat_tail_adj: float = 1.0,
    floor: float = 0.0,
    cap: float = 1.00,
) -> dict:
    """Convert a return series' historical Ulcer Index into an annual vol target.

    Bridge (Brownian-motion approximation): for a driftless random walk the
    expected drawdown magnitude scales like σ·√T·√(2/π), so

        implied_per_period_vol = UI_fraction · √(π/2) / √T

    where UI_fraction = UI(%)/100 and T = average drawdown duration (periods).
    We annualise by √periods_per_year, multiply by `fat_tail_adj` (≥1, to widen
    the target for fat tails vs. the Gaussian constant 1.2533), and clamp to
    [floor, cap]. The default band is [0%, 100%] — a SANITY rail only (a target
    above 100% annual vol from a degenerate window would tell the overlay to
    never de-risk), NOT a risk-policy ceiling: the book's true Ulcer-implied
    target stands (e.g. an aggressive single-name book reaching ~58%).

    Returns a dict with the target AND its inputs, so the assumptions are
    auditable wherever it's used.
    """
    ui = ulcer_index(returns)                      # percent
    T = avg_drawdown_duration(returns)
    ui_frac = ui / 100.0
    const = np.sqrt(np.pi / 2.0) * float(fat_tail_adj)   # ≈1.2533 × adj
    per_period = ui_frac * const / np.sqrt(max(T, 1.0))
    annual = per_period * np.sqrt(periods_per_year)
    target = float(np.clip(annual, floor, cap))
    return {
        "vol_target": target,
        "ulcer_index": round(ui, 4),
        "avg_dd_duration": round(T, 2),
        "raw_annual": round(float(annual), 4),
        "clamped": bool(abs(target - annual) > 1e-9),
        "fat_tail_adj": float(fat_tail_adj),
        "n_periods": int(np.asarray(returns).size),
    }


def trend_ulcer_vol_target(
    prices: np.ndarray,
    returns: np.ndarray | None = None,
    periods_per_year: int = 252,
    trend_years: float = 2.5,
    roll_years: float = 4.0,
    floor: float = 0.0,
    cap: float = 1.00,
) -> dict:
    """Per-portfolio vol target from the DRIFT-NEUTRAL rolling trend-Ulcer.

    The full pipeline that fixes the drift inflation of plain peak-based Ulcer
    over a long history:
      1. trailing 2.5y price²-weighted MA = the trend (drift carrier);
      2. drawdowns measured BELOW that trend (drift-neutral);
      3. rolling 4y trend-Ulcer, averaged over the (e.g. 15y) history;
      4. UI→vol with an EMPIRICALLY-CALIBRATED constant: instead of the
         theoretical √(π/2)/√T, we use the realized-vol-to-trend-Ulcer ratio
         measured on THIS book's own series — so the conversion reflects the
         book's actual drift/tail behaviour, not a Brownian idealisation.

    `prices` drives the trend/Ulcer; `returns` (if given) is used for the
    realized-vol calibration (else derived from prices). Returns the target plus
    the full breakdown for audit.
    """
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p)]
    if p.size < 60:
        return {"vol_target": floor if floor > 0 else 0.10, "source": "fallback_short"}

    roll = rolling_trend_ulcer(p, periods_per_year, trend_years, roll_years)
    avg_ui = roll["avg_ulcer"]                 # percent, trend-relative

    if returns is None:
        r = np.diff(np.log(p))
    else:
        r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    realized_annual_vol = float(np.std(r, ddof=1) * np.sqrt(periods_per_year)) if r.size > 1 else 0.0

    # Empirical calibration constant c = realized_annual_vol / avg_trend_ulcer_frac.
    # This is the book's OWN measured "how much annual vol per unit of trend-
    # relative drawdown" — drift-neutral (the trend carries the drift) and tail-
    # aware (real returns), replacing the Brownian √(π/2). Reported for audit; it
    # is the analogue of the theoretical constant, but measured.
    avg_ui_frac = avg_ui / 100.0
    calib_const = (realized_annual_vol / avg_ui_frac) if avg_ui_frac > _EPS else float("nan")

    # The TARGET is realized vol scaled by the REGIME ratio: the book's current
    # (whole-sample) trend-Ulcer vs. its rolling-average trend-Ulcer. <1 ⇒ the
    # book is calmer now than its long-run norm ⇒ target a touch below realized
    # vol; >1 ⇒ currently more stressed ⇒ a touch above. Bounded to [0.5, 2].
    # This is non-circular (≠ realized vol) and uses the rolling-UI history.
    whole_ui_frac = trend_ulcer_index(p, roll["trend_window"]) / 100.0
    regime_adj = float(np.clip(
        (whole_ui_frac / avg_ui_frac) if avg_ui_frac > _EPS else 1.0, 0.5, 2.0))
    annual = realized_annual_vol * regime_adj

    target = float(np.clip(annual, floor, cap))
    return {
        "vol_target": target,
        "source": "trend_ulcer",
        "avg_trend_ulcer_pct": round(avg_ui, 4),
        "whole_sample_trend_ulcer_pct": round(whole_ui_frac * 100, 4),
        "realized_annual_vol": round(realized_annual_vol, 4),
        "calib_const": round(calib_const, 3) if np.isfinite(calib_const) else None,
        "regime_adj": round(regime_adj, 3),
        "raw_annual": round(float(annual), 4),
        "clamped": bool(abs(target - annual) > 1e-9),
        "n_windows": roll["n_windows"],
        "n_prices": int(p.size),
    }


def turnover(weight_series: list[dict]) -> float:
    """Average one-way turnover per rebalance: ½ Σ|wₜ − wₜ₋₁|, averaged.

    weight_series is the ordered list of target-weight dicts at each rebalance.
    Drift between rebalances is ignored (a standard simplification); this is the
    *decision* turnover, which is what distinguishes the strategies.
    """
    if len(weight_series) < 2:
        return 0.0
    totals = []
    for prev, cur in zip(weight_series, weight_series[1:]):
        keys = set(prev) | set(cur)
        l1 = sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
        totals.append(0.5 * l1)
    return float(np.mean(totals)) if totals else 0.0


def execution_turnover(pairs: list[tuple[dict, dict]]) -> float:
    """Average one-way turnover ACTUALLY TRADED per rebalance: ½ Σ|target − drifted|.

    Unlike `turnover` (target→target = *decision* turnover, what the PM intended
    to move), this is *execution* turnover: each pair is (drifted_book, target),
    where `drifted_book` is where the prior holdings floated to under realised
    returns by the rebalance date. It is what you would have to TRADE to reach the
    target — the chargeable number a transaction-cost model would bill. A pure
    buy-and-hold book (target always == drifted) has execution turnover 0.
    """
    if not pairs:
        return 0.0
    totals = []
    for drifted, target in pairs:
        keys = set(drifted) | set(target)
        l1 = sum(abs(target.get(k, 0.0) - drifted.get(k, 0.0)) for k in keys)
        totals.append(0.5 * l1)
    return float(np.mean(totals)) if totals else 0.0


def deflated_sharpe(
    observed_sharpe: float,
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int = 252,
) -> float:
    """Probability the TRUE Sharpe > 0 given the observed Sharpe and n_trials.

    Bailey & López de Prado (2014). Corrects for (a) selection across n_trials
    strategies and (b) non-normal returns (skew γ₃, kurtosis γ₄). Returns a
    probability in [0,1]; >0.95 ≈ "skill, not luck". Inputs/observed Sharpe are
    in ANNUALISED units; we de-annualise internally to per-period.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 4 or n_trials < 1:
        return float("nan")

    sr = observed_sharpe / np.sqrt(periods_per_year)     # per-period Sharpe
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))        # non-excess kurtosis

    # Expected max Sharpe across n_trials independent N(0,1) trials (via the
    # extreme-value approximation used in the DSR paper).
    e = np.e
    gamma = 0.5772156649  # Euler–Mascheroni
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    sr0 = z1 * (1.0 - gamma) + z2 * gamma                # expected max under H0 (per-period)

    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, _EPS))
    dsr_stat = (sr - sr0) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(dsr_stat))


def all_metrics(returns: np.ndarray, periods_per_year: int = 252,
                weight_series: list[dict] | None = None) -> dict:
    """Bundle the core metrics into one dict (turnover only if weights given)."""
    out = {
        "cagr":         cagr(returns, periods_per_year),
        "annual_vol":   annual_vol(returns, periods_per_year),
        "sharpe":       sharpe(returns, periods_per_year),
        "sortino":      sortino(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar":       calmar(returns, periods_per_year),
    }
    if weight_series is not None:
        out["turnover"] = turnover(weight_series)
    return out
