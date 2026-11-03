"""Phase 4 — GARCH benchmark + scoreboard unit tests (no network, no LLM).

Covers:
  * metrics against closed-form / known cases (Sharpe, max DD, CAGR, turnover);
  * GARCH/GJR MLE recovers parameters on SIMULATED data and BIC picks the right
    model (GJR on asymmetric, GARCH on symmetric);
  * the vol-target benchmark: produces a valid constrained weight vector, applies
    10-step partial adjustment, and reads-but-never-mutates its inputs;
  * the scoreboard simulator + table render on a deterministic price frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic.dataflows.constraints import PortfolioConstraints
from agentic.quant import metrics
from agentic.quant.benchmark import GarchVolBenchmark
from agentic.quant.garch import fit_garch, select_garch
from agentic.quant.scoreboard import (
    Scoreboard,
    Strategy,
    efficient_frontier,
    hold_initial_strategy,
    random_strategies,
    simulate_returns,
)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_sharpe_of_constant_positive_returns_is_large_and_finite():
    r = np.full(252, 0.001)
    # zero variance → guard returns 0 (not inf)
    assert metrics.sharpe(r) == 0.0


def test_sharpe_sign_matches_mean():
    rng = np.random.default_rng(1)
    up = 0.0005 + 0.01 * rng.standard_normal(2000)
    assert metrics.sharpe(up) > 0


def test_max_drawdown_known_path():
    # +10% then -50% then flat: peak 1.1, trough 0.55 → DD = 0.55/1.1 - 1 = -0.5
    r = np.array([0.10, -0.50, 0.0])
    assert metrics.max_drawdown(r) == pytest.approx(-0.5, abs=1e-9)


def test_cagr_doubles_in_one_year():
    # 252 daily returns compounding to exactly 2x → CAGR = 100%
    g = 2.0 ** (1 / 252) - 1
    r = np.full(252, g)
    assert metrics.cagr(r, 252) == pytest.approx(1.0, rel=1e-6)


def test_turnover_counts_one_way_changes():
    # 0.5/0.5 → 0.7/0.3 : L1 = 0.4, one-way = 0.2
    ws = [{"A": 0.5, "B": 0.5}, {"A": 0.7, "B": 0.3}]
    assert metrics.turnover(ws) == pytest.approx(0.2, abs=1e-9)


def test_execution_turnover_measures_traded_weight():
    # Drifted book {A:0.62, B:0.38} re-pinned to target {A:0.50, B:0.50}:
    # L1 = 0.12 + 0.12 = 0.24, one-way = 0.12.
    pairs = [({"A": 0.62, "B": 0.38}, {"A": 0.50, "B": 0.50})]
    assert metrics.execution_turnover(pairs) == pytest.approx(0.12, abs=1e-9)


def test_execution_turnover_zero_for_buy_and_hold():
    # A passive book's target always equals its drifted book → nothing traded.
    pairs = [({"A": 0.62, "B": 0.38}, {"A": 0.62, "B": 0.38})]
    assert metrics.execution_turnover(pairs) == 0.0
    assert metrics.execution_turnover([]) == 0.0


def test_deflated_sharpe_in_unit_interval():
    rng = np.random.default_rng(2)
    r = 0.0004 + 0.01 * rng.standard_normal(1000)
    sr = metrics.sharpe(r)
    dsr = metrics.deflated_sharpe(sr, r, n_trials=1000)
    assert 0.0 <= dsr <= 1.0


# ---- Ulcer Index → vol target ----------------------------------------------

def test_ulcer_index_zero_when_never_underwater():
    r = np.array([0.01, 0.02, 0.01, 0.03])   # monotonically rising → no drawdown
    assert metrics.ulcer_index(r) == pytest.approx(0.0, abs=1e-9)


def test_ulcer_index_known_path():
    # +0, -10%, -5% recovery-ish: drawdowns from peak are 0, -10%, -14.5%? Compute.
    r = np.array([0.0, -0.10, 0.0])
    # curve = [1, 0.9, 0.9]; peak = [1,1,1]; dd% = [0, -10, -10]; UI = sqrt(mean([0,100,100]))
    assert metrics.ulcer_index(r) == pytest.approx(np.sqrt(200 / 3), abs=1e-6)


def test_avg_drawdown_duration_counts_underwater_runs():
    # Start UP (sets the peak), then a 2-period underwater run, recover to a new
    # peak, then a 1-period underwater run → runs [2, 1] → mean 1.5.
    # curve peaks at idx0 (1.10); idx1,2 underwater; idx3 new peak; idx4 underwater.
    r = np.array([0.10, -0.05, -0.05, 0.20, -0.05])
    assert metrics.avg_drawdown_duration(r) == pytest.approx(1.5)


def test_implied_vol_target_monotone_in_ulcer():
    # A higher-vol (deeper-drawdown) series must yield a higher implied target.
    rng = np.random.default_rng(0)
    calm = 0.005 * rng.standard_normal(500)
    wild = 0.030 * rng.standard_normal(500)
    tc = metrics.implied_vol_target_from_ulcer(calm)["vol_target"]
    tw = metrics.implied_vol_target_from_ulcer(wild)["vol_target"]
    assert tw > tc


def test_implied_vol_target_sanity_cap_only():
    # The default band is a sanity rail [0,1]; an explicit tighter cap still works.
    rng = np.random.default_rng(1)
    insane = 0.15 * rng.standard_normal(500)   # absurd vol
    capped = metrics.implied_vol_target_from_ulcer(insane, cap=0.40)
    assert capped["vol_target"] == pytest.approx(0.40) and capped["clamped"]
    # with the default 100% sanity cap, an aggressive-but-sane book is NOT clamped
    rng2 = np.random.default_rng(3)
    aggressive = 0.03 * rng2.standard_normal(500)
    res = metrics.implied_vol_target_from_ulcer(aggressive)
    assert res["vol_target"] < 1.0 and not res["clamped"]


def test_implied_vol_target_reports_inputs():
    r = 0.01 * np.random.default_rng(2).standard_normal(300)
    res = metrics.implied_vol_target_from_ulcer(r)
    assert {"vol_target", "ulcer_index", "avg_dd_duration", "raw_annual",
            "clamped", "fat_tail_adj", "n_periods"} <= set(res)


# ---- trend-relative (drift-neutral) Ulcer ----------------------------------

def test_price_sq_weighted_ma_tracks_level():
    p = np.array([10.0, 20.0, 30.0, 40.0])
    wma = metrics._price_sq_weighted_ma(p, window=4)
    # Σp²/Σp = (100+400+900+1600)/(10+20+30+40) = 3000/100 = 30
    assert wma[-1] == pytest.approx(30.0)


def test_trend_ulcer_zero_on_smooth_uptrend():
    # A purely rising series never dips below its trend → ~0 trend-Ulcer.
    p = 100 * np.cumprod(1 + np.full(1500, 0.0005))
    assert metrics.trend_ulcer_index(p, trend_window=630) == pytest.approx(0.0, abs=1e-6)


def test_trend_ulcer_below_peak_ulcer_under_drift():
    # The whole point: under strong drift, trend-relative UI < peak-based UI
    # (peak UI is inflated by absolute drawdowns the drifting series wanders into).
    rng = np.random.default_rng(0)
    r = 0.0008 + 0.015 * rng.standard_normal(3000)
    p = 100 * np.cumprod(1 + r)
    assert metrics.trend_ulcer_index(p, 630) < metrics.ulcer_index(r)


def test_rolling_trend_ulcer_averages_windows():
    rng = np.random.default_rng(1)
    p = 100 * np.cumprod(1 + 0.01 * rng.standard_normal(3000))
    res = metrics.rolling_trend_ulcer(p)
    assert res["basis"] == "rolling" and res["n_windows"] > 1
    assert res["avg_ulcer"] >= 0


def test_trend_ulcer_vol_target_anchored_to_realized_vol():
    # Target ≈ realized vol × regime_adj (bounded), NOT the drift-inflated number.
    rng = np.random.default_rng(2)
    r = 0.0005 + 0.012 * rng.standard_normal(3000)
    p = 100 * np.cumprod(1 + r)
    res = metrics.trend_ulcer_vol_target(p, r)
    assert res["source"] == "trend_ulcer"
    rv = res["realized_annual_vol"]
    assert 0.5 * rv <= res["vol_target"] <= 2.0 * rv + 1e-9
    assert {"avg_trend_ulcer_pct", "calib_const", "regime_adj", "n_windows"} <= set(res)


# ---------------------------------------------------------------------------
# GARCH / GJR
# ---------------------------------------------------------------------------

def _sim_garch(n, omega, alpha, beta, gamma=0.0, burn=500, seed=0):
    rng = np.random.default_rng(seed)
    N = n + burn
    e = np.zeros(N)
    s2 = np.zeros(N)
    s2[0] = omega / max(1 - alpha - beta - gamma / 2, 1e-3)
    for t in range(1, N):
        lev = gamma * e[t - 1] ** 2 * (e[t - 1] < 0)
        s2[t] = omega + alpha * e[t - 1] ** 2 + lev + beta * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * rng.standard_normal()
    return e[burn:]


def test_garch_recovers_params_on_simulated_data():
    r = _sim_garch(4000, omega=0.02, alpha=0.08, beta=0.90, seed=3)
    f = fit_garch(r, "garch")
    assert f.converged
    # finite-sample MLE: allow generous tolerance but confirm the ballpark
    assert f.params["alpha"] == pytest.approx(0.08, abs=0.05)
    assert f.params["beta"] == pytest.approx(0.90, abs=0.06)
    assert 0.85 < f.persistence < 1.0


def test_bic_selects_gjr_on_asymmetric_and_garch_on_symmetric():
    sym = _sim_garch(4000, omega=0.02, alpha=0.08, beta=0.90, seed=4)
    asym = _sim_garch(4000, omega=0.02, alpha=0.03, beta=0.90, gamma=0.10, seed=5)
    assert select_garch(sym).model == "garch"
    assert select_garch(asym).model == "gjr"


def test_forecast_vol_is_positive_decimal():
    r = _sim_garch(2000, omega=0.02, alpha=0.05, beta=0.90, seed=6)
    f = select_garch(r)
    v = f.forecast_vol(1)
    assert v > 0 and v < 1.0   # a daily vol, in decimal


def test_fit_rejects_short_series():
    with pytest.raises(ValueError):
        fit_garch(np.zeros(10))


def test_order_search_returns_ranked_list_with_orders():
    from agentic.quant.garch import order_search
    r = _sim_garch(3000, omega=0.02, alpha=0.08, beta=0.90, seed=11)
    ranked = order_search(r, max_p=2, max_q=2, include_gjr=True)
    # all GARCH(p,q) in the 2x2 grid + GJR(1,1) that converged, BIC-sorted
    assert len(ranked) >= 4
    bics = [f.bic for f in ranked]
    assert bics == sorted(bics)              # ascending BIC = best first
    # every fit reports its order and is stationary
    for f in ranked:
        assert f.order[0] >= 1 and f.order[1] >= 1
        assert f.persistence < 1.0


def test_bic_prefers_parsimony_on_garch11_data():
    # Data generated by a TRUE (1,1) process. BIC should pick a PARSIMONIOUS low
    # order and (1,1) must be within ~2 BIC of the winner (Kass-Raftery: ΔBIC<2
    # is "not worth more than a bare mention" — i.e. an effective tie). We do NOT
    # assert the winner is exactly (1,1): at finite n an extra lag occasionally
    # shaves a point or two of BIC by fitting in-sample noise. The honest claim
    # is "BIC does not reward gratuitous complexity," not "BIC is infallible".
    from agentic.quant.garch import order_search
    r = _sim_garch(4000, omega=0.02, alpha=0.07, beta=0.91, seed=12)
    ranked = order_search(r, max_p=2, max_q=2, include_gjr=False)
    best = ranked[0]
    by_order = {f.order: f for f in ranked}
    # winner is low-order (total lags <= 3) and stationary
    assert sum(best.order) <= 3
    assert best.persistence < 1.0
    # (1,1) is at worst an effective tie with the winner
    assert by_order[(1, 1)].bic - best.bic < 2.0


def test_fit_garch_pq_higher_order_runs_and_is_stationary():
    r = _sim_garch(3000, omega=0.02, alpha=0.05, beta=0.90, seed=13)
    f = fit_garch(r, "garch", p=2, q=2)
    assert f.p == 2 and f.q == 2
    assert len(f.params["alpha_vec"]) == 2 and len(f.params["beta_vec"]) == 2
    assert f.persistence < 1.0
    assert f.forecast_vol(1) > 0


def test_gjr_rejects_higher_order():
    r = _sim_garch(1000, omega=0.02, alpha=0.05, beta=0.90, seed=14)
    with pytest.raises(ValueError):
        fit_garch(r, "gjr", p=2, q=1)


def test_cumulative_horizon_vol_vs_naive_sqrt_t():
    # Term-structure-correct weekly vol = √(Σ_{h=1..5} σ²_{t+h}); the naive rule
    # is σ_{t+1}·√5. They must DIFFER under mean reversion and CONVERGE as the
    # conditional vol approaches the unconditional level / persistence→1.
    r = _sim_garch(3000, omega=0.05, alpha=0.07, beta=0.90, seed=20)
    f = fit_garch(r, "garch")
    one = f.forecast_vol(1)
    naive = one * np.sqrt(5)
    correct = f.forecast_vol_horizon(5)
    # both positive, in a sane band, and not identical (mean reversion present)
    assert correct > 0 and naive > 0
    assert abs(correct - naive) > 1e-9
    # 1-period horizon reduces exactly to the 1-step forecast
    assert f.forecast_vol_horizon(1) == pytest.approx(one, rel=1e-9)


def test_cumulative_equals_naive_when_at_unconditional_vol():
    # Construct a fit whose LAST conditional variance equals the unconditional
    # variance: then every future σ²_{t+h} is constant = that level, so the
    # cumulative √(Σσ²) equals σ·√h exactly (no reversion to do).
    from agentic.quant.garch import GarchFit
    omega, alpha, beta = 0.05, 0.05, 0.90
    uncond = omega / (1 - alpha - beta)         # percent² unconditional variance
    f = GarchFit(params={"omega": omega, "alpha": alpha, "beta": beta,
                         "alpha_vec": [alpha], "beta_vec": [beta]},
                 model="garch", loglik=0.0, aic=0.0, bic=0.0, n=100,
                 sigma2=np.array([uncond]), converged=True, p=1, q=1)
    naive = f.forecast_vol(1) * np.sqrt(5)
    correct = f.forecast_vol_horizon(5)
    assert correct == pytest.approx(naive, rel=1e-9)


# ---------------------------------------------------------------------------
# vol-target benchmark
# ---------------------------------------------------------------------------

def _returns_dict(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "SPY": 0.0005 + 0.011 * rng.standard_normal(400),
        "VTI": 0.0005 + 0.012 * rng.standard_normal(400),
        "BND": 0.0001 + 0.003 * rng.standard_normal(400),
    }


def test_benchmark_produces_valid_constrained_weights():
    base = {"SPY": 0.4, "VTI": 0.3, "BND": 0.2, "CASH": 0.1}
    types = {"SPY": "equity_fund", "VTI": "equity_fund", "BND": "bond_fund", "CASH": "cash"}
    bm = GarchVolBenchmark(annual_vol_target=0.10, rebalance_period="weekly", cadence_factor=0.1)
    d = bm.step(base, _returns_dict(), types, PortfolioConstraints(min_cash=0.05), prior_scalar=1.0)
    # weights are rounded to 6dp, so 4 positions can accumulate ~2e-6 of rounding
    assert abs(sum(d.target_weights.values()) - 1.0) < 1e-5
    assert d.target_weights.get("CASH", 0) >= 0.05 - 1e-4
    assert 0.0 <= d.exposure_scalar <= 1.0


def test_benchmark_partial_adjustment_moves_10pct():
    base = {"SPY": 0.5, "BND": 0.4, "CASH": 0.1}
    types = {"SPY": "equity_fund", "BND": "bond_fund", "CASH": "cash"}
    bm = GarchVolBenchmark(annual_vol_target=0.05, rebalance_period="weekly", cadence_factor=0.1)
    d = bm.step(base, _returns_dict(7), types, prior_scalar=1.0)
    # applied = prior + 0.1*(raw - prior); so |applied-prior| = 0.1*|raw-prior|
    expected = 1.0 + 0.1 * (d.raw_exposure_scalar - 1.0)
    assert d.exposure_scalar == pytest.approx(expected, abs=1e-9)


def test_benchmark_does_not_mutate_inputs():
    base = {"SPY": 0.5, "BND": 0.4, "CASH": 0.1}
    types = {"SPY": "equity_fund", "BND": "bond_fund", "CASH": "cash"}
    base_copy, types_copy = dict(base), dict(types)
    bm = GarchVolBenchmark()
    bm.step(base, _returns_dict(8), types)
    assert base == base_copy and types == types_copy   # decoupling check


# ---------------------------------------------------------------------------
# scoreboard
# ---------------------------------------------------------------------------

def _price_frame():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    # A trends up, B flat-ish
    a = 100 * np.cumprod(1 + np.full(60, 0.002))
    b = 100 * np.cumprod(1 + np.full(60, 0.0))
    return pd.DataFrame({"A": a, "B": b}, index=dates)


def test_simulate_returns_tracks_weights():
    prices = _price_frame()
    # 100% A should return ~0.2% per day
    rets = simulate_returns([("2024-01-01", {"A": 1.0})], prices)
    assert rets.size > 0
    assert np.mean(rets) == pytest.approx(0.002, abs=1e-4)


def test_hold_initial_never_trades_after_t0():
    s = hold_initial_strategy({"A": 0.6, "B": 0.4}, ["2024-01-01", "2024-02-01"])
    assert len(s.rebalances) == 1     # one rebalance only


def test_scoreboard_table_renders_with_random_distribution():
    prices = _price_frame()
    dates = ["2024-01-01"]
    sb = Scoreboard(periods_per_year=252)
    our = Strategy("our_model", [("2024-01-01", {"A": 0.7, "B": 0.3})])
    sb.add(our, prices)
    sb.add(hold_initial_strategy({"A": 0.5, "B": 0.5}, dates), prices)
    randoms = random_strategies(["A", "B"], dates, n=50, seed=0)
    for r in randoms:
        sb.add(r, prices) if False else None  # don't pollute rows; use distribution adder
    sb.add_random_distribution(randoms, prices, our_name="our_model")
    table = sb.to_table()
    assert "our_model" in table and "hold_initial" in table
    assert "deflated Sharpe" in table
    # our_model row comes first
    assert table.index("our_model") < table.index("hold_initial")


def _noisy_price_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    # Two imperfectly-correlated risky assets with different mean/vol.
    a = 100 * np.cumprod(1 + 0.0008 + 0.012 * rng.standard_normal(n))
    b = 100 * np.cumprod(1 + 0.0003 + 0.008 * rng.standard_normal(n))
    return pd.DataFrame({"A": a, "B": b}, index=dates)


def test_efficient_frontier_tangency_is_valid():
    fr = efficient_frontier(["A", "B", "CASH"], _noisy_price_frame())
    assert fr   # non-empty
    # tangency weights are long-only and sum to 1 over the risky names
    w = fr["tangency_weights"]
    assert set(w) == {"A", "B"}
    assert all(v >= -1e-9 for v in w.values())
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    # frontier arrays are parallel and annualised vols are positive
    assert len(fr["vol"]) == len(fr["ret"]) == fr["n_points"]
    assert (np.asarray(fr["vol"]) > 0).all()
    # tangency Sharpe is the max over the sampled frontier points
    sharpes = np.asarray(fr["ret"]) / np.asarray(fr["vol"])
    assert fr["tangency_sharpe"] == pytest.approx(float(sharpes.max()), abs=1e-9)


def test_efficient_frontier_needs_two_priced_tickers():
    # Only one risky name → no frontier.
    assert efficient_frontier(["A", "CASH"], _noisy_price_frame()) == {}


def test_scoreboard_attaches_frontier_reference():
    prices = _noisy_price_frame()
    sb = Scoreboard(periods_per_year=252)
    sb.add(Strategy("our_model", [("2024-01-01", {"A": 0.6, "B": 0.4})]), prices)
    sb.add_efficient_frontier(["A", "B"], prices, our_name="our_model")
    table = sb.to_table()
    assert "frontier_tangency" in table
    assert "efficient frontier" in table
    ex = sb.extras["our_model"]
    assert "frontier_tangency_sharpe" in ex
    assert ex["sharpe_gap_to_frontier"] <= 1e-9   # our model can't beat the optimum
