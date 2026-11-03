"""Phase 3 — forward-test runner unit tests (no LLM, PortfolioGraph stubbed).

Covers the runner's deterministic logic:
  * weekday mapping (weekends → Friday track);
  * first run seeds the track from the seed file;
  * each weekday track is INDEPENDENT (a run on one doesn't touch another);
  * the run appends a history record and writes the new weights back as the
    track's current holdings (preserving holding types);
  * analyst signals are serialised into the history record.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from agentic.agents.utils.signals import AnalystSignal, Claim
from agentic.forward import runner as fwd
from agentic.forward.runner import ForwardTest, weekday_name

_SEED = {
    "objective": "test",
    "constraints": {"max_single_position": 0.5},
    "holdings": {
        "AAA": {"weight": 0.6, "type": "equity"},
        "BND": {"weight": 0.3, "type": "bond_fund"},
        "CASH": {"weight": 0.1, "type": "cash"},
    },
}


def _make_signal():
    c = Claim(statement="AAA rises", ticker="AAA", metric="return",
              direction="positive", horizon="1M", made_on=date(2026, 6, 15))
    return AnalystSignal(direction=0.5, confidence=0.7, horizon="1M",
                         key_evidence=[c], prose="p")


class _StubPG:
    """Stand-in for PortfolioGraph: returns a canned rebalance + signals."""

    def __init__(self, config=None, debug=False):
        self.config = config or {}

    def propagate(self, main_ticker, portfolio_path, date_str, **kw):
        # A simple deterministic "decision": shift a little AAA→BND.
        state = {
            "final_decision": {
                "target_weights": {"AAA": 0.5, "BND": 0.4, "CASH": 0.1},
                "changes": [{"ticker": "AAA", "action": "trim", "amount": 0.1,
                             "explanation": "trim concentration"}],
                "narrative": "rebalanced",
                "constraints_applied": {"limits": {"max_single_position": 0.5}, "adjusted": {}},
            },
            "per_ticker_reports": {
                "AAA": {"signals": {"market": _make_signal()}},
                "BND": {"signals": {}},
            },
        }
        return state, "report-md"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    # The runner's NAV step uses a shared ReturnCache; stub its downloader so
    # these structural tests never touch yfinance (NAV just marks flat).
    from agentic.quant.return_cache import ReturnCache
    monkeypatch.setattr(ReturnCache, "_download_closes",
                        staticmethod(lambda ticker, start, end: {}))


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setattr(fwd, "PortfolioGraph", _StubPG)
    # Two named portfolios sharing the same seed shape (enough to test the
    # portfolio × weekday matrix without three real fixtures).
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps(_SEED), encoding="utf-8")
    ft = ForwardTest(
        portfolios={"pa": str(seed), "pb": str(seed)},
        forward_dir=tmp_path / "forward",
    )
    return ft, tmp_path


def test_weekday_mapping_weekend_maps_to_friday():
    assert weekday_name(date(2026, 6, 15)) == "monday"
    assert weekday_name(date(2026, 6, 19)) == "friday"
    assert weekday_name(date(2026, 6, 20)) == "friday"   # Saturday
    assert weekday_name(date(2026, 6, 21)) == "friday"   # Sunday


def test_first_run_seeds_and_records(seeded):
    ft, tmp = seeded
    rec = ft.run_one("pa", run_date="2026-06-15")  # a Monday
    assert rec["weekday"] == "monday" and rec["portfolio"] == "pa"
    track = tmp / "forward" / "pa" / "monday"
    assert (track / "portfolio.json").exists()
    assert (track / "history.jsonl").exists()
    lines = (track / "history.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["target_weights"]["BND"] == 0.4


def test_weights_written_back_with_types(seeded):
    ft, tmp = seeded
    ft.run_one("pa", run_date="2026-06-15")
    pf = json.loads((tmp / "forward" / "pa" / "monday" / "portfolio.json").read_text())
    assert pf["holdings"]["BND"]["weight"] == 0.4
    assert pf["holdings"]["BND"]["type"] == "bond_fund"
    assert pf["constraints"]["max_single_position"] == 0.5


def test_seed_portfolio_preserved_immutably(seeded):
    ft, tmp = seeded
    ft.run_one("pa", run_date="2026-06-15")
    track = tmp / "forward" / "pa" / "monday"
    seed = json.loads((track / "seed_portfolio.json").read_text())
    # seed holds the ORIGINAL allocation (AAA 0.6), not the post-rebalance one
    assert seed["holdings"]["AAA"]["weight"] == 0.6
    # a second run must NOT change the immutable seed
    ft.run_one("pa", run_date="2026-06-22")
    seed2 = json.loads((track / "seed_portfolio.json").read_text())
    assert seed2 == seed


def test_dated_portfolio_snapshots_per_run(seeded):
    ft, tmp = seeded
    ft.run_one("pa", run_date="2026-06-15")
    ft.run_one("pa", run_date="2026-06-22")
    snaps = sorted((tmp / "forward" / "pa" / "monday" / "portfolios").glob("*.json"))
    assert [p.name for p in snaps] == [
        "portfolio_2026-06-15.json", "portfolio_2026-06-22.json"]
    # each snapshot records the post-rebalance book for THAT date
    s1 = json.loads(snaps[0].read_text())
    assert s1["as_of"] == "2026-06-15"


def test_history_records_start_weights(seeded):
    ft, tmp = seeded
    r1 = ft.run_one("pa", run_date="2026-06-15")
    # first run starts from the seed (AAA 0.6)
    assert r1["first_run"] is True
    assert r1["start_weights"]["AAA"] == 0.6
    r2 = ft.run_one("pa", run_date="2026-06-22")
    # Second run starts from run-1's target (AAA 0.5) DRIFTED by the week's real
    # price moves — buy-and-hold drift, so it is near 0.5 but not exactly 0.5
    # (the book floats; it is not silently rebalanced back to target for free).
    assert r2["first_run"] is False
    assert r2["start_weights"]["AAA"] == pytest.approx(0.5, abs=0.05)
    # weights still normalise to ~1.0 after drift
    assert sum(r2["start_weights"].values()) == pytest.approx(1.0, abs=1e-6)


def test_signals_serialised_into_history(seeded):
    ft, tmp = seeded
    rec = ft.run_one("pa", run_date="2026-06-15")
    sig = rec["signals"]["AAA"]["market"]
    assert sig["direction"] == 0.5
    assert sig["key_evidence"][0]["ticker"] == "AAA"
    json.dumps(rec, default=str)


def test_run_all_runs_every_portfolio_for_one_weekday(seeded):
    ft, tmp = seeded
    recs = ft.run_all(run_date="2026-06-15")   # Monday, both portfolios
    assert {r["portfolio"] for r in recs} == {"pa", "pb"}
    assert (tmp / "forward" / "pa" / "monday" / "history.jsonl").exists()
    assert (tmp / "forward" / "pb" / "monday" / "history.jsonl").exists()


def test_tracks_are_independent_across_portfolio_and_weekday(seeded):
    ft, tmp = seeded
    ft.run_one("pa", run_date="2026-06-15")  # pa/monday
    ft.run_one("pa", run_date="2026-06-16")  # pa/tuesday
    ft.run_one("pb", run_date="2026-06-15")  # pb/monday
    # each (portfolio, weekday) is its own track
    for p in ("pa", "pb"):
        for wd in ("monday",):
            assert (tmp / "forward" / p / wd / "history.jsonl").exists()
    # a repeat pa/monday appends rather than clobbering
    ft.run_one("pa", run_date="2026-06-22")  # next monday
    pa_mon = (tmp / "forward" / "pa" / "monday" / "history.jsonl").read_text().strip().splitlines()
    pa_tue = (tmp / "forward" / "pa" / "tuesday" / "history.jsonl").read_text().strip().splitlines()
    assert len(pa_mon) == 2 and len(pa_tue) == 1


def test_two_stage_cadence_first_run_uses_initial_factor(tmp_path, monkeypatch):
    # First run of a track → cadence_factor_initial (0.5); second run → 0.1.
    seen = {}

    class _CfgCapturePG(_StubPG):
        def __init__(self, config=None, debug=False):
            super().__init__(config, debug)
            # record the cadence in effect for whichever run is happening now
            seen.setdefault("seq", []).append(config.get("cadence_factor"))

    monkeypatch.setattr(fwd, "PortfolioGraph", _CfgCapturePG)
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps(_SEED), encoding="utf-8")
    ft = ForwardTest(portfolios={"p": str(seed)}, forward_dir=tmp_path / "forward")
    ft.run_one("p", run_date="2026-06-15")   # first → initial factor
    ft.run_one("p", run_date="2026-06-22")   # second → normal factor
    assert seen["seq"][0] == pytest.approx(0.5)    # cadence_factor_initial default
    assert seen["seq"][1] == pytest.approx(0.1)    # cadence_factor default


def test_run_all_isolates_per_portfolio_failure(tmp_path, monkeypatch):
    # If one portfolio raises, the others still run and the failure is recorded.
    class _FlakyPG(_StubPG):
        def propagate(self, main_ticker, portfolio_path, date_str, **kw):
            if "boom" in str(portfolio_path):
                raise RuntimeError("kaboom")
            return super().propagate(main_ticker, portfolio_path, date_str, **kw)

    monkeypatch.setattr(fwd, "PortfolioGraph", _FlakyPG)
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps(_SEED), encoding="utf-8")
    boom = tmp_path / "boom.json"
    boom.write_text(json.dumps(_SEED), encoding="utf-8")
    ft = ForwardTest(portfolios={"ok": str(seed), "boom": str(boom)},
                     forward_dir=tmp_path / "forward")
    recs = ft.run_all(run_date="2026-06-15")
    by = {r["portfolio"]: r for r in recs}
    assert "error" in by["boom"] and "error" not in by["ok"]
