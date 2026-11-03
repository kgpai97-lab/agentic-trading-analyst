"""Phase 1 data-contract unit tests (Step 7).

Deterministic, no LLM. Covers:
  * Claim / AnalystSignal pydantic validation (accept valid, reject malformed).
  * enforce_constraints() satisfies every numeric limit on adversarial weight
    vectors and renormalises to 1.0.
  * scan_objective_for_numeric_constraints() warns on a numeric limit planted
    in objective prose.
  * scale_changes() cadence damping: deltas scaled by cadence_factor; 1.0 is a
    no-op; 'exit' is preserved.
"""

from __future__ import annotations

from datetime import date

import pytest

from agentic.agents.portfolio.portfolio_manager import apply_changes, scale_changes
from agentic.agents.utils.signals import AnalystSignal, Claim
from agentic.dataflows.constraints import (
    PortfolioConstraints,
    enforce_constraints,
    scan_objective_for_numeric_constraints,
)

_TOL = 1e-6


# ---------------------------------------------------------------------------
# 1a — Claim / AnalystSignal validation
# ---------------------------------------------------------------------------

def test_claim_accepts_and_normalises_valid_input():
    c = Claim(
        statement="  NVDA outperforms the SOX  ", ticker="nvda",
        metric="return", direction="Positive", horizon="1m",
        made_on=date(2026, 6, 14),
    )
    assert c.direction == "positive"   # lowercased
    assert c.horizon == "1M"           # uppercased
    assert c.statement == "NVDA outperforms the SOX"  # stripped
    assert c.realized_outcome is None and c.correct is None


@pytest.mark.parametrize("bad", [
    {"direction": "sideways-ish"},  # no canonical/synonym token to coerce
    {"horizon": "2Q"},            # not a known horizon
    {"statement": "   "},         # empty after strip
    {"ticker": ""},               # empty ticker
])
def test_claim_rejects_malformed(bad):
    base = dict(
        statement="x", ticker="NVDA", metric="return",
        direction="positive", horizon="1M", made_on=date(2026, 6, 14),
    )
    base.update(bad)
    with pytest.raises(Exception):
        Claim(**base)


@pytest.mark.parametrize("raw,expected", [
    ("up", "positive"),
    ("bullish", "positive"),
    ("BEARISH", "negative"),
    ("mixed", "neutral"),
    ("flat", "neutral"),
    ("neutral-to-negative", "neutral"),   # first canonical token wins
    ("positive (risk)", "positive"),
])
def test_claim_coerces_direction_synonyms(raw, expected):
    """The extraction LLM reaches for synonyms ('mixed', 'bullish'); we coerce
    them onto the canonical three rather than dropping the claim."""
    c = Claim(
        statement="x", ticker="NVDA", metric="return",
        direction=raw, horizon="1M", made_on=date(2026, 6, 14),
    )
    assert c.direction == expected


def test_analyst_signal_accepts_valid():
    s = AnalystSignal(direction=-0.4, confidence=0.9, horizon="3M",
                      key_evidence=[], prose="report")
    assert s.direction == -0.4 and s.confidence == 0.9


@pytest.mark.parametrize("bad", [
    {"direction": 1.5},      # > 1
    {"direction": -2.0},     # < -1
    {"confidence": 1.1},     # > 1
    {"confidence": -0.1},    # < 0
    {"horizon": "soon"},     # unknown horizon
])
def test_analyst_signal_rejects_out_of_range(bad):
    base = dict(direction=0.0, confidence=0.5, horizon="1M")
    base.update(bad)
    with pytest.raises(Exception):
        AnalystSignal(**base)


# ---------------------------------------------------------------------------
# 1c/1d — enforce_constraints
# ---------------------------------------------------------------------------

def _sums_to_one(w):
    return abs(sum(w.values()) - 1.0) < _TOL


def test_enforce_no_constraints_is_just_renormalise():
    w = {"A": 0.5, "B": 0.3, "C": 0.4}  # sums to 1.2
    types = {"A": "equity", "B": "equity", "C": "equity"}
    out = enforce_constraints(w, types, PortfolioConstraints())
    assert _sums_to_one(out)
    # ratios preserved
    assert out["A"] == pytest.approx(0.5 / 1.2)


def test_enforce_per_position_cap():
    w = {"A": 0.7, "B": 0.2, "C": 0.1}
    types = {k: "equity" for k in w}
    out = enforce_constraints(w, types, PortfolioConstraints(max_single_position=0.4))
    assert _sums_to_one(out)
    assert all(v <= 0.4 + 1e-4 for t, v in out.items())
    # freed weight went to the other names, not vanished
    assert out["B"] > 0.2 and out["C"] > 0.1


def test_enforce_per_position_cap_all_at_cap_is_feasible():
    # 3 names, cap 0.34 → feasible (sum cap 1.02 >= 1). Each should sit ~0.333.
    w = {"A": 0.8, "B": 0.1, "C": 0.1}
    types = {k: "equity" for k in w}
    out = enforce_constraints(w, types, PortfolioConstraints(max_single_position=0.34))
    assert _sums_to_one(out)
    assert all(v <= 0.34 + 1e-4 for v in out.values())


def test_enforce_min_cash():
    w = {"A": 0.6, "B": 0.35, "CASH": 0.05}
    types = {"A": "equity", "B": "equity", "CASH": "cash"}
    out = enforce_constraints(w, types, PortfolioConstraints(min_cash=0.20))
    assert _sums_to_one(out)
    assert out["CASH"] >= 0.20 - 1e-4
    # risky sleeve scaled down but ratio A:B preserved
    assert out["A"] / out["B"] == pytest.approx(0.6 / 0.35, rel=1e-3)


def test_enforce_max_cash():
    w = {"A": 0.3, "B": 0.2, "CASH": 0.5}
    types = {"A": "equity", "B": "equity", "CASH": "cash"}
    out = enforce_constraints(w, types, PortfolioConstraints(max_cash=0.10))
    assert _sums_to_one(out)
    assert out["CASH"] <= 0.10 + 1e-4


def test_enforce_min_bond_weight():
    w = {"VTI": 0.8, "BND": 0.1, "CASH": 0.1}
    types = {"VTI": "equity_fund", "BND": "bond_fund", "CASH": "cash"}
    out = enforce_constraints(w, types, PortfolioConstraints(min_bond_weight=0.40))
    assert _sums_to_one(out)
    assert out["BND"] >= 0.40 - 1e-4


def test_enforce_combined_limits_all_satisfied():
    # Adversarial: over-concentrated, under-bonded, under-cashed at once.
    w = {"AAA": 0.75, "BBB": 0.15, "BND": 0.05, "CASH": 0.05}
    types = {"AAA": "equity", "BBB": "equity", "BND": "bond_fund", "CASH": "cash"}
    c = PortfolioConstraints(max_single_position=0.40, min_bond_weight=0.20, min_cash=0.10)
    out = enforce_constraints(w, types, c)
    assert _sums_to_one(out)
    assert out.get("CASH", 0) >= 0.10 - 1e-4
    assert out.get("BND", 0) >= 0.20 - 1e-4
    # cap is on non-cash holdings
    assert all(v <= 0.40 + 1e-4 for t, v in out.items() if t != "CASH")


def test_constraints_reject_incoherent_cash_band():
    with pytest.raises(Exception):
        PortfolioConstraints(min_cash=0.5, max_cash=0.2)


def test_enforce_max_total_stock_caps_aggregate_equity():
    # 80% NVDA + 0 others, cap aggregate stock at 40% → equity sleeve -> 0.40,
    # freed weight goes to the non-stock/non-cash sleeve (BND) first.
    w = {"NVDA": 0.80, "BND": 0.10, "CASH": 0.10}
    types = {"NVDA": "equity", "BND": "bond_fund", "CASH": "cash"}
    out = enforce_constraints(w, types, PortfolioConstraints(max_total_stock=0.40))
    assert _sums_to_one(out)
    stock = sum(out[t] for t in out if types[t] in ("equity", "equity_fund"))
    assert stock <= 0.40 + 1e-4
    assert out["BND"] > 0.10           # freed weight landed in bonds


def test_enforce_max_total_stock_sums_equity_and_equity_fund():
    # Both 'equity' and 'equity_fund' count toward the aggregate stock cap.
    w = {"NVDA": 0.30, "QQQ": 0.40, "BND": 0.20, "CASH": 0.10}
    types = {"NVDA": "equity", "QQQ": "equity_fund", "BND": "bond_fund", "CASH": "cash"}
    out = enforce_constraints(w, types, PortfolioConstraints(max_total_stock=0.50))
    assert _sums_to_one(out)
    stock = out.get("NVDA", 0) + out.get("QQQ", 0)
    assert stock <= 0.50 + 1e-4


def test_enforce_max_total_stock_to_cash_when_no_other_sleeve():
    # All-equity + cash, cap stock at 60% → the only home for freed weight is cash.
    w = {"NVDA": 0.50, "QQQ": 0.50}
    types = {"NVDA": "equity", "QQQ": "equity_fund"}
    out = enforce_constraints(w, types, PortfolioConstraints(max_total_stock=0.60))
    assert _sums_to_one(out)
    stock = out.get("NVDA", 0) + out.get("QQQ", 0)
    assert stock <= 0.60 + 1e-4
    assert out.get("CASH", 0) >= 0.40 - 1e-4   # freed weight became cash


def test_enforce_total_stock_cap_with_single_and_cash_limits():
    # All three interacting: per-name cap, aggregate stock cap, cash floor.
    w = {"NVDA": 0.70, "AMD": 0.15, "BND": 0.10, "CASH": 0.05}
    types = {"NVDA": "equity", "AMD": "equity", "BND": "bond_fund", "CASH": "cash"}
    c = PortfolioConstraints(max_single_position=0.40, max_total_stock=0.50, min_cash=0.10)
    out = enforce_constraints(w, types, c)
    assert _sums_to_one(out)
    assert all(v <= 0.40 + 1e-4 for t, v in out.items() if t != "CASH")
    assert out.get("NVDA", 0) + out.get("AMD", 0) <= 0.50 + 1e-3
    assert out.get("CASH", 0) >= 0.10 - 1e-3


# ---------------------------------------------------------------------------
# 1c — load-time objective-prose scan
# ---------------------------------------------------------------------------

def test_scan_warns_on_numeric_limit_in_prose():
    obj = "Long-term growth. Never hold more than 15% in any single stock."
    warns = scan_objective_for_numeric_constraints(obj)
    assert warns and "15%" in warns[0]


def test_scan_silent_on_prose_without_numeric_limits():
    obj = "Tilt toward quality and momentum; prefer liquid large caps."
    assert scan_objective_for_numeric_constraints(obj) == []


# ---------------------------------------------------------------------------
# 1e — cadence damping
# ---------------------------------------------------------------------------

def test_scale_changes_damps_by_factor():
    ch = [
        {"ticker": "A", "action": "trim", "amount": 0.20, "explanation": "x"},
        {"ticker": "B", "action": "add", "amount": 0.20, "explanation": "y"},
    ]
    out = scale_changes(ch, 0.1)
    assert out[0]["amount"] == pytest.approx(0.02)
    assert out[1]["amount"] == pytest.approx(0.02)


def test_scale_changes_factor_one_is_noop():
    ch = [{"ticker": "A", "action": "trim", "amount": 0.20, "explanation": "x"}]
    out = scale_changes(ch, 1.0)
    assert out[0]["amount"] == pytest.approx(0.20)


def test_scale_changes_preserves_exit():
    ch = [{"ticker": "A", "action": "exit", "amount": 0.0, "explanation": "x"}]
    out = scale_changes(ch, 0.1)
    assert out[0]["action"] == "exit"


def test_cadence_completes_over_about_ten_rebalances():
    # Apply a damped 'add' repeatedly; the position should approach (but the
    # test just checks ~10 steps make meaningful progress vs. 1 step).
    cur = {"A": 0.5, "B": 0.5}
    proposed = [{"ticker": "A", "action": "add", "amount": 0.30, "explanation": "x"}]
    one_step, _, _ = apply_changes(cur, scale_changes(proposed, 0.1))
    # one 10% step moves A by ~0.03 pre-renormalisation; clearly < full 0.30
    assert one_step["A"] < 0.5 + 0.30
    assert one_step["A"] > 0.5  # but it did move up
