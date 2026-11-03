"""NAV mark-to-market tests (pure engine + tracker with network mocked)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from agentic.dataflows.constraints import PortfolioConstraints
from agentic.forward import nav_tracker as nt
from agentic.forward.nav_tracker import NavTracker
from agentic.quant.nav import NavBook, drift_weights, period_return


# ---- pure engine ------------------------------------------------------------

def test_period_return_weighted_sum():
    r = period_return({"A": 0.5, "B": 0.5, "CASH": 0.0}, {"A": 0.10, "B": -0.02})
    assert r == pytest.approx(0.04)


def test_period_return_cash_and_missing_earn_zero():
    # CASH earns 0; a ticker absent from returns earns 0.
    r = period_return({"A": 0.5, "CASH": 0.3, "ZZZ": 0.2}, {"A": 0.10})
    assert r == pytest.approx(0.05)   # only A's 0.5*0.10


def test_navbook_compounds():
    nb = NavBook("x")
    nb.mark("2026-06-15", {"A": 1.0}, {"A": 0.05})
    nb.mark("2026-06-22", {"A": 1.0}, {"A": 0.05})
    assert nb.nav == pytest.approx(1.1025)
    assert len(nb.history) == 2
    assert nb.history[-1]["nav"] == pytest.approx(1.1025, abs=1e-6)


def test_drift_weights_floats_with_returns():
    # A +10%, B flat: A's weight grows, renormalised. 0.6·1.1=0.66, 0.4·1.0=0.40,
    # total 1.06 → A 0.66/1.06, B 0.40/1.06.
    d = drift_weights({"A": 0.6, "B": 0.4}, {"A": 0.10, "B": 0.0})
    assert d["A"] == pytest.approx(0.66 / 1.06)
    assert d["B"] == pytest.approx(0.40 / 1.06)
    assert sum(d.values()) == pytest.approx(1.0)


def test_drift_weights_cash_and_missing_grow_by_one():
    # CASH and tickers absent from returns grow ×1; their weight shifts only
    # relative to risky names that moved. A +20%: 0.5·1.2=0.6, CASH 0.5·1=0.5.
    d = drift_weights({"A": 0.5, "CASH": 0.5}, {"A": 0.20})
    assert d["A"] == pytest.approx(0.6 / 1.1)
    assert d["CASH"] == pytest.approx(0.5 / 1.1)


def test_drift_weights_flat_week_is_noop():
    w = {"A": 0.7, "B": 0.3}
    d = drift_weights(w, {"A": 0.0, "B": 0.0})
    assert d["A"] == pytest.approx(0.7)
    assert d["B"] == pytest.approx(0.3)


def test_navbook_negative_return():
    nb = NavBook("x")
    nb.mark("d1", {"A": 1.0}, {"A": -0.20})
    assert nb.nav == pytest.approx(0.80)


# ---- tracker (fetch mocked, no network) -------------------------------------

@pytest.fixture
def no_network(monkeypatch):
    # Default: assets are flat (0 return) unless a test overrides.
    monkeypatch.setattr(nt, "fetch_asset_returns", lambda tickers, p, t: {})
    # GARCH benchmark inside _garch_weights: make it a pass-through (base mix)
    # so the tracker logic is tested without fitting models.
    monkeypatch.setattr(nt, "GarchVolBenchmark", _PassThroughBenchmark)


class _PassThroughDecision:
    def __init__(self, w):
        self.target_weights = dict(w)
        self.exposure_scalar = 1.0


class _PassThroughBenchmark:
    def __init__(self, *a, **k): pass
    def step(self, base, returns, types, constraints=None, prior_scalar=1.0):
        return _PassThroughDecision(base)


def test_first_run_seeds_core_books_at_nav_one(tmp_path, no_network):
    tr = NavTracker(tmp_path, ticker_types={"A": "equity", "CASH": "cash"})
    row = tr.update("2026-06-15", {"A": 0.9, "CASH": 0.1}, {"A": np.zeros(50)})
    assert row["first"] is True
    assert all(v == 1.0 for v in row["nav"].values())
    # state persisted with our_model, hold_initial, garch. hold_built is NOT
    # created here (seeded only when hold_built_weights is supplied — the
    # full-coverage run; see the dedicated test below). The random comparison
    # cloud is NOT a live book — it is reconstructed offline by the scoreboard.
    state = json.loads((tmp_path / "nav_state.json").read_text())
    names = set(state["books"])
    assert {"our_model", "hold_initial", "garch"} <= names
    assert "hold_built" not in names
    assert not any(n.startswith("random_") for n in names)


def test_second_run_marks_realised_return(tmp_path, monkeypatch):
    monkeypatch.setattr(nt, "GarchVolBenchmark", _PassThroughBenchmark)
    # week 1 flat (seed), week 2: A returns +10%
    calls = {"n": 0}
    def fake_returns(tickers, prev, cur):
        calls["n"] += 1
        return {"A": 0.10}
    monkeypatch.setattr(nt, "fetch_asset_returns", fake_returns)

    tr = NavTracker(tmp_path, ticker_types={"A": "equity", "CASH": "cash"})
    tr.update("2026-06-15", {"A": 1.0}, {"A": np.zeros(50)})        # seed
    row = tr.update("2026-06-22", {"A": 1.0}, {"A": np.zeros(50)})  # mark +10%
    # our_model held 100% A → +10% → NAV 1.10
    assert row["nav"]["our_model"] == pytest.approx(1.10)
    assert row["returns"]["our_model"] == pytest.approx(0.10)


def test_hold_initial_stays_at_inception_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(nt, "GarchVolBenchmark", _PassThroughBenchmark)
    monkeypatch.setattr(nt, "fetch_asset_returns", lambda tk, p, c: {"A": 0.10, "B": 0.0})
    tr = NavTracker(tmp_path, ticker_types={"A": "equity", "B": "equity", "CASH": "cash"})
    # inception 100% A; then our model rotates to 100% B
    tr.update("2026-06-15", {"A": 1.0}, {"A": np.zeros(50), "B": np.zeros(50)})
    row = tr.update("2026-06-22", {"B": 1.0}, {"A": np.zeros(50), "B": np.zeros(50)})
    # hold_initial still holds A (+10%); our_model now holds B (0%) for NEXT week,
    # but the marked week used the PRIOR held weights (A for both at seed).
    state = json.loads((tmp_path / "nav_state.json").read_text())
    assert state["books"]["hold_initial"]["held_weights"] == {"A": 1.0}
    assert state["books"]["our_model"]["held_weights"] == {"B": 1.0}   # rotated


def test_hold_built_seeds_with_full_move_weights_and_stays_fixed(tmp_path, monkeypatch):
    """hold_built seeds from the full-conviction build and never reweights."""
    monkeypatch.setattr(nt, "GarchVolBenchmark", _PassThroughBenchmark)
    monkeypatch.setattr(nt, "fetch_asset_returns", lambda tk, p, c: {"A": 0.10, "B": 0.0})
    tr = NavTracker(tmp_path, ticker_types={"A": "equity", "B": "equity", "CASH": "cash"})
    # our_model seeds at {A:0.9, B:0.1}; the full-move book is a DIFFERENT mix.
    full_build = {"A": 0.6, "B": 0.4}
    tr.update("2026-06-15", {"A": 0.9, "B": 0.1},
              {"A": np.zeros(50), "B": np.zeros(50)},
              hold_built_weights=full_build)
    state = json.loads((tmp_path / "nav_state.json").read_text())
    assert state["books"]["hold_built"]["held_weights"] == full_build
    assert state["books"]["hold_built"]["nav"] == pytest.approx(1.0)
    # our_model rotates next week; hold_built is NEVER retraded — but it DOES
    # drift with realised returns (buy-and-hold float, not a frozen mix). With
    # A +10%, B flat: A's weight grows 0.6 → 0.6·1.1/(0.6·1.1+0.4) = 0.62264.
    tr.update("2026-06-22", {"B": 1.0},
              {"A": np.zeros(50), "B": np.zeros(50)},
              hold_built_weights={"A": 0.0, "B": 1.0})  # ignored after seed
    state = json.loads((tmp_path / "nav_state.json").read_text())
    hb = state["books"]["hold_built"]["held_weights"]
    assert hb["A"] == pytest.approx(0.66 / 1.06)        # drifted up (never traded)
    assert hb["B"] == pytest.approx(0.40 / 1.06)        # drifted down
    assert state["books"]["our_model"]["held_weights"] == {"B": 1.0}    # rotated to target


def test_hold_built_absent_when_not_supplied(tmp_path, no_network):
    """The focused run passes no hold_built_weights → no hold_built book."""
    tr = NavTracker(tmp_path, ticker_types={"A": "equity", "CASH": "cash"})
    tr.update("2026-06-15", {"A": 0.9, "CASH": 0.1}, {"A": np.zeros(50)})
    state = json.loads((tmp_path / "nav_state.json").read_text())
    assert "hold_built" not in state["books"]


def test_full_move_build_differs_from_damped_our_model():
    """The reconstructed full move != the cadence-damped target (cadence<1)."""
    from agentic.agents.portfolio.portfolio_manager import apply_changes, scale_changes
    start = {"A": 0.5, "B": 0.5}
    proposed = [{"ticker": "A", "action": "trim", "amount": 0.30, "explanation": "x"},
                {"ticker": "B", "action": "add", "amount": 0.30, "explanation": "y"}]
    full, _, _ = apply_changes(dict(start), proposed)
    damped, _, _ = apply_changes(dict(start), scale_changes(proposed, 0.5))
    assert full != damped
    assert full["A"] == pytest.approx(0.20)   # full move
    assert damped["A"] == pytest.approx(0.35)  # half move


def test_nav_jsonl_appends_one_row_per_run(tmp_path, no_network):
    tr = NavTracker(tmp_path, ticker_types={"A": "equity"})
    tr.update("2026-06-15", {"A": 1.0}, {"A": np.zeros(50)})
    tr.update("2026-06-22", {"A": 1.0}, {"A": np.zeros(50)})
    lines = (tmp_path / "nav.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["first"] is True


def test_vol_target_derived_from_trend_ulcer_and_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(nt, "GarchVolBenchmark", _PassThroughBenchmark)
    monkeypatch.setattr(nt, "fetch_asset_returns", lambda tk, p, c: {})
    import pandas as pd
    rng = np.random.default_rng(0)
    # a long aligned history so the trend-Ulcer path (needs >=60) fires
    idx = pd.bdate_range("2020-01-01", periods=1200)
    aligned = pd.DataFrame({"A": 0.012 * rng.standard_normal(1200)}, index=idx)
    rets = {"A": aligned["A"].values}
    tr = NavTracker(tmp_path, ticker_types={"A": "equity", "CASH": "cash"})
    tr.update("2026-06-15", {"A": 0.9, "CASH": 0.1}, rets, aligned_returns=aligned)
    state = json.loads((tmp_path / "nav_state.json").read_text())
    vt = state["vol_target"]
    assert vt["source"] == "trend_ulcer"
    assert vt["vol_target"] > 0 and "realized_annual_vol" in vt
    # second run must NOT change the frozen target
    tr2 = NavTracker(tmp_path, ticker_types={"A": "equity", "CASH": "cash"})
    tr2.update("2026-06-22", {"A": 0.9, "CASH": 0.1}, rets, aligned_returns=aligned)
    state2 = json.loads((tmp_path / "nav_state.json").read_text())
    assert state2["vol_target"]["vol_target"] == vt["vol_target"]


def test_vol_target_falls_back_without_history(tmp_path, no_network):
    tr = NavTracker(tmp_path, ticker_types={"A": "equity"}, annual_vol_target=0.12)
    tr.update("2026-06-15", {"A": 1.0}, {})          # no returns → fallback
    state = json.loads((tmp_path / "nav_state.json").read_text())
    assert state["vol_target"]["source"] == "fallback_default"
    assert state["vol_target"]["vol_target"] == 0.12
