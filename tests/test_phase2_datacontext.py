"""Phase 2 — DataContext / snapshot / manifest unit tests (no network).

Covers:
  * DataContext ContextVar isolation across threads (the concurrency-clobber fix)
    and the back-compat set_run_context / current_* readers.
  * _fetch_history builds an as_of-bounded request (end exclusive) and falls back
    to period= when as_of is empty/malformed — verified by monkeypatching yfinance
    so no network is touched.
  * write_snapshot writes the expected path + record and is a no-op when disabled.
  * manifest hashes prompts, redacts secrets, and is JSON-serialisable.
"""

from __future__ import annotations

import json
import threading

from agentic.dataflows import y_finance
from agentic.dataflows.data_context import (
    RunContext,
    current_as_of_date,
    current_fund_subclass,
    current_lookback_days,
    get_run_context,
    set_run_context,
)
from agentic.dataflows.manifest import build_manifest, hash_prompts
from agentic.dataflows.result import ToolResult
from agentic.dataflows.snapshot import write_snapshot


# ---------------------------------------------------------------------------
# DataContext
# ---------------------------------------------------------------------------

def test_set_and_read_run_context():
    set_run_context(trade_date="2024-01-15", lookback_days=20, fund_subclass="bond_fund")
    assert current_as_of_date() == "2024-01-15"
    assert current_lookback_days() == 20
    assert current_fund_subclass() == "bond_fund"
    assert get_run_context() == RunContext("2024-01-15", 20, "bond_fund")


def test_context_is_isolated_across_threads():
    # The whole point of the ContextVar: a second thread's set must NOT clobber
    # this thread's value (the old module-dict singleton failed this).
    set_run_context(trade_date="2020-01-01", lookback_days=10)
    seen = {}

    def worker():
        set_run_context(trade_date="2099-12-31", lookback_days=99)
        seen["worker"] = current_as_of_date()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert seen["worker"] == "2099-12-31"
    # This thread is unchanged despite the worker overwriting in its own context.
    assert current_as_of_date() == "2020-01-01"


def test_default_context_is_live():
    # A fresh ContextVar default (no set) is "live": empty as_of, no bound.
    assert RunContext().as_of_date == ""


def test_update_run_context_preserves_ticker():
    from agentic.dataflows.data_context import current_ticker, update_run_context
    set_run_context(trade_date="2024-01-15", lookback_days=30)
    update_run_context(ticker="TQQQ")
    assert current_ticker() == "TQQQ"
    update_run_context(trade_date="2024-01-16")   # re-publish date must not wipe ticker
    assert current_ticker() == "TQQQ"


def test_update_run_context_preserves_other_fields():
    # The analyst-node bug: re-setting trade_date must NOT wipe lookback /
    # snapshot_dir that TradingGraph.propagate set up front.
    from agentic.dataflows.data_context import current_snapshot_dir, update_run_context
    set_run_context(trade_date="2024-01-15", lookback_days=45,
                    fund_subclass="bond_fund", snapshot_dir="outputs/snap")
    update_run_context(trade_date="2024-02-01")   # analyst node re-publishes date
    assert current_as_of_date() == "2024-02-01"
    assert current_lookback_days() == 45            # preserved
    assert current_fund_subclass() == "bond_fund"   # preserved
    assert current_snapshot_dir() == "outputs/snap" # preserved


def test_route_to_vendor_snapshots_from_context(tmp_path, monkeypatch):
    # snapshot_dir on the DataContext (not config) must trigger a snapshot write,
    # since the @tool wrappers call route_to_vendor without a config.
    import agentic.dataflows.interface as iface
    monkeypatch.setattr(iface.y_finance, "get_fundamentals_yfinance",
                        lambda ticker: ToolResult.success(f"funds for {ticker}"))
    set_run_context(trade_date="2024-01-15", snapshot_dir=str(tmp_path))
    iface.route_to_vendor("get_fundamentals", "AAPL")
    written = list(tmp_path.rglob("*.json"))
    assert written and written[0].parts[-3:] == ("2024-01-15", "AAPL", "get_fundamentals.json")


# ---------------------------------------------------------------------------
# _normalise_ticker — dict unwrap + typo snap
# ---------------------------------------------------------------------------

def test_normalise_ticker_plain_string():
    from agentic.agents.utils.agent_utils import _normalise_ticker
    set_run_context(trade_date="2024-01-15")
    update_run_context = __import__("agentic.dataflows.data_context", fromlist=["update_run_context"]).update_run_context
    update_run_context(ticker="TQQQ")
    assert _normalise_ticker("TQQQ") == "TQQQ"


def test_normalise_ticker_dict_unwrap():
    from agentic.agents.utils.agent_utils import _normalise_ticker
    assert _normalise_ticker({"ticker": "NVDA"}) == "NVDA"
    assert _normalise_ticker({"symbol": "AAPL"}) == "AAPL"


def test_normalise_ticker_snaps_one_char_typo(capsys):
    from agentic.agents.utils.agent_utils import _normalise_ticker
    from agentic.dataflows.data_context import update_run_context
    set_run_context(trade_date="2024-01-15")
    update_run_context(ticker="TQQQ")
    result = _normalise_ticker("TQQ3")           # last char wrong
    assert result == "TQQQ"
    assert "TQQ3" in capsys.readouterr().out     # warning was printed


def test_normalise_ticker_no_snap_two_char_diff():
    from agentic.agents.utils.agent_utils import _normalise_ticker
    from agentic.dataflows.data_context import update_run_context
    set_run_context(trade_date="2024-01-15")
    update_run_context(ticker="TQQQ")
    assert _normalise_ticker("SOXL") == "SOXL"  # 4 edits away — not snapped


def test_normalise_ticker_no_snap_without_context():
    from agentic.agents.utils.agent_utils import _normalise_ticker
    set_run_context()                             # clear — no ticker set
    assert _normalise_ticker("TQQ3") == "TQQ3"  # nothing to snap to


def test_normalise_ticker_snaps_to_universe(capsys):
    # The hallucinated typo is for a DIFFERENT holding than the one being
    # analysed — caught via the run universe, not the context ticker.
    from agentic.agents.utils.agent_utils import _normalise_ticker
    from agentic.dataflows.data_context import update_run_context
    set_run_context(trade_date="2024-01-15", universe=("NVDA", "TQQQ", "BITO"))
    update_run_context(ticker="NVDA")             # analysing NVDA, not TQQQ
    result = _normalise_ticker("TQQ3")            # cross-holding typo for TQQQ
    assert result == "TQQQ"
    assert "TQQ3" in capsys.readouterr().out      # warning was printed


def test_normalise_ticker_exact_universe_match_not_snapped():
    # A valid universe ticker must pass through untouched (no spurious snap to a
    # 1-edit-distance sibling).
    from agentic.agents.utils.agent_utils import _normalise_ticker
    set_run_context(trade_date="2024-01-15", universe=("TQQQ", "QQQ"))
    assert _normalise_ticker("QQQ") == "QQQ"      # exact match short-circuits


def test_edit_distance():
    from agentic.agents.utils.agent_utils import _edit_distance
    assert _edit_distance("TQQQ", "TQQQ") == 0
    assert _edit_distance("TQQ3", "TQQQ") == 1   # substitution
    assert _edit_distance("TQQQQ", "TQQQ") == 1  # insertion
    assert _edit_distance("TQQ", "TQQQ") == 1    # deletion
    assert _edit_distance("SOXL", "TQQQ") == 4


# ---------------------------------------------------------------------------
# _fetch_history — as_of bounding (yfinance monkeypatched; no network)
# ---------------------------------------------------------------------------

class _FakeTicker:
    last_call: dict = {}

    def __init__(self, ticker):
        self.ticker = ticker

    def history(self, **kwargs):
        _FakeTicker.last_call = dict(kwargs)
        import pandas as pd  # noqa: PLC0415
        return pd.DataFrame()  # empty is fine; we only inspect the call args


def test_fetch_history_uses_end_bound_for_as_of(monkeypatch):
    import yfinance as yf  # noqa: PLC0415
    monkeypatch.setattr(yf, "Ticker", _FakeTicker)

    y_finance._fetch_history("AAPL", lookback_days=30, as_of_date="2024-01-15")
    call = _FakeTicker.last_call
    # as_of present → start/end window, NOT period
    assert "period" not in call
    assert "start" in call and "end" in call
    # end is EXCLUSIVE = as_of + 1 day
    assert call["end"] == "2024-01-16"
    assert call["start"] < "2024-01-15"


def test_fetch_history_uses_period_when_live(monkeypatch):
    import yfinance as yf  # noqa: PLC0415
    monkeypatch.setattr(yf, "Ticker", _FakeTicker)

    y_finance._fetch_history("AAPL", lookback_days=30, as_of_date="")
    call = _FakeTicker.last_call
    assert "period" in call and "start" not in call


def test_fetch_history_malformed_as_of_falls_back_to_period(monkeypatch):
    import yfinance as yf  # noqa: PLC0415
    monkeypatch.setattr(yf, "Ticker", _FakeTicker)

    y_finance._fetch_history("AAPL", lookback_days=30, as_of_date="not-a-date")
    assert "period" in _FakeTicker.last_call


# ---------------------------------------------------------------------------
# Snapshot persistence
# ---------------------------------------------------------------------------

def test_write_snapshot_creates_dated_path(tmp_path):
    res = ToolResult.success("payload-text", row_count=3)
    path = write_snapshot(tmp_path, "2024-01-15", "AAPL", "get_stock_data", res)
    assert path is not None and path.exists()
    assert path.parts[-3:] == ("2024-01-15", "AAPL", "get_stock_data.json")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["tool"] == "get_stock_data"
    assert rec["result"]["ok"] is True
    assert rec["result"]["payload"] == "payload-text"


def test_write_snapshot_disabled_is_noop(tmp_path):
    res = ToolResult.success("x")
    assert write_snapshot("", "2024-01-15", "AAPL", "get_stock_data", res) is None


def test_write_snapshot_sanitises_ticker(tmp_path):
    res = ToolResult.success("x")
    path = write_snapshot(tmp_path, "2024-01-15", "^VIX", "get_indicators", res)
    assert path is not None and path.exists()  # no path-traversal / illegal chars


def test_write_snapshot_uses_today_when_as_of_empty(tmp_path):
    # Live runs (empty as_of) still get a dated folder = today.
    res = ToolResult.success("x")
    path = write_snapshot(tmp_path, "", "AAPL", "get_news", res)
    assert path is not None and path.exists()
    # parent dir is a YYYY-MM-DD date string
    date_dir = path.parts[-3]
    assert len(date_dir) == 10 and date_dir[4] == "-"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_hash_prompts_returns_hashes(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("nope", encoding="utf-8")
    h = hash_prompts(tmp_path)
    assert set(h.keys()) == {"a.txt", "b.txt"}     # only *.txt
    assert all(len(v) == 64 for v in h.values())   # sha256 hex


def test_manifest_redacts_secrets_and_serialises(tmp_path):
    (tmp_path / "p.txt").write_text("prompt", encoding="utf-8")
    cfg = {
        "deep_think_llm": "gpt-oss:120b", "quick_think_llm": "gpt-oss:120b",
        "deep_reasoning": True, "quick_reasoning": True, "llm_provider": "ollama",
        "ollama_base_url": "https://secret.example", "snapshot_dir": "outputs/snap",
    }
    m = build_manifest(cfg, tmp_path, "2026-06-15", label="FWD")
    assert m["config"]["ollama_base_url"] == "<redacted>"
    assert m["models"]["deep_think_llm"] == "gpt-oss:120b"
    assert m["prompt_hashes"]["p.txt"]
    # fully JSON-serialisable
    json.dumps(m, default=str)
