"""Snapshot persistence — write-only provenance of every tool fetch.

Purpose (Phase 2): when a run fetches data, we save the exact raw tool output
that informed the decision, keyed by the run's as_of_date. This is the
provenance trail the forward test (Phase 3) and the claim checker (Phase 5)
rely on — "here is precisely what the model saw on 2026-06-15 when it picked
NVDA." It is **write-only during analysis**: nothing in the live pipeline reads
snapshots back to feed a new run (the as_of-aware replay reader is a deferred
Phase 8 / backtest concern). yfinance cannot backdate fundamentals/holdings, so
there is no "historical snapshot fetch" — only fetch-live-then-store.

Layout:
    <snapshot_dir>/<as_of_date>/<TICKER>/<tool>.json

Each file holds the ToolResult (ok / payload / error / meta) plus a captured
timestamp. as_of_date is the run's analysis date ("live" when unset, so live
forward runs still get a dated folder = today via the caller).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from agentic.dataflows.result import ToolResult

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(part: str) -> str:
    """Make a path component filesystem-safe (tickers like 'BRK.B', '^VIX')."""
    cleaned = _SAFE.sub("_", str(part)).strip("_")
    return cleaned or "unknown"


def write_snapshot(
    snapshot_dir: str | Path,
    as_of_date: str,
    ticker: str,
    tool: str,
    result: ToolResult,
) -> Path | None:
    """Persist one tool result. Returns the written path (or None if disabled).

    Best-effort: any I/O error is swallowed (provenance must never break a run).
    """
    if not snapshot_dir:
        return None
    try:
        date_key = as_of_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = Path(snapshot_dir) / _safe(date_key) / _safe(ticker) / f"{_safe(tool)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "tool":       tool,
            "ticker":     ticker,
            "as_of_date": as_of_date,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "result": {
                "ok":      result.ok,
                "payload": result.payload,
                "error":   result.error,
                "meta":    result.meta,
            },
        }
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return path
    except Exception as exc:  # noqa: BLE001 — provenance is best-effort
        print(f"[snapshot] WARNING: failed to write {tool} for {ticker}: {exc}")
        return None
