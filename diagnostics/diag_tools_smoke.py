"""Smoke-test every data tool for every ticker in portfolio_test.json.

Calls each tool that would actually fire for a given holding's asset type,
prints the full tagged result, and summarises pass/fail at the end.
No LLM involved.

Usage:
    .\\venv\\Scripts\\python.exe diagnostics\\diag_tools_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NamedTuple

import _bootstrap  # noqa: F401  — puts project root on sys.path

ROOT = _bootstrap.ROOT

from agentic.dataflows.interface import route_to_vendor  # noqa: E402


# ---------------------------------------------------------------------------
# Which tools fire for each asset type
# ---------------------------------------------------------------------------

# Shared across all analysable asset types
_COMMON_TOOLS = ["get_news", "get_global_news", "get_stock_data", "get_indicators"]

TOOLS_BY_TYPE: dict[str, list[str]] = {
    "equity":           _COMMON_TOOLS + ["get_fundamentals", "get_income_statement"],
    "equity_fund":      _COMMON_TOOLS + ["get_etf_holdings"],
    "bond_fund":        _COMMON_TOOLS + ["get_etf_holdings"],
    "alternative_fund": _COMMON_TOOLS + ["get_etf_holdings"],
    "cash":             [],  # non-analysable
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Result(NamedTuple):
    ticker: str
    tool: str
    tag: str       # OK | EMPTY | ERROR
    preview: str   # first few lines


def _call(ticker: str, tool: str) -> Result:
    try:
        tr = route_to_vendor(tool, ticker)
        text = tr.to_llm_string()
    except Exception as exc:  # noqa: BLE001
        text = f"[ERROR] Exception: {exc}"

    if text.startswith("[OK]"):
        tag = "OK"
    elif text.startswith("[EMPTY]"):
        tag = "EMPTY"
    else:
        tag = "ERROR"

    # Strip non-ASCII for Windows console
    preview = text.encode("ascii", errors="replace").decode("ascii")
    return Result(ticker, tool, tag, preview)


def _colour(tag: str) -> str:
    codes = {"OK": "\033[32m", "EMPTY": "\033[33m", "ERROR": "\033[31m"}
    return f"{codes.get(tag, '')}[{tag}]\033[0m"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    portfolio_path = ROOT / "portfolio_test.json"
    portfolio = json.loads(portfolio_path.read_text())
    holdings: dict[str, dict] = portfolio["holdings"]

    results: list[Result] = []

    for ticker, meta in holdings.items():
        asset_type = meta.get("type", "equity")
        tools = TOOLS_BY_TYPE.get(asset_type, [])

        if not tools:
            print(f"\n{'-'*60}")
            print(f"  {ticker:<8}  ({asset_type})  -- skipped (non-analysable)")
            continue

        print(f"\n{'='*60}")
        print(f"  {ticker}  ({asset_type})  -- {len(tools)} tools")
        print(f"{'='*60}")

        for tool in tools:
            r = _call(ticker, tool)
            results.append(r)
            print(f"\n  {tool}")
            print(f"  {_colour(r.tag)}")
            for line in r.preview.splitlines():
                print(f"    {line}")

    # Summary table
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    total = len(results)
    by_tag: dict[str, list[Result]] = {"OK": [], "EMPTY": [], "ERROR": []}
    for r in results:
        by_tag.setdefault(r.tag, []).append(r)

    print(f"  Total calls : {total}")
    print(f"  OK          : {len(by_tag['OK'])}")
    print(f"  EMPTY       : {len(by_tag['EMPTY'])}")
    print(f"  ERROR       : {len(by_tag['ERROR'])}")

    if by_tag["ERROR"]:
        print("\n  Failures:")
        for r in by_tag["ERROR"]:
            print(f"    {r.ticker:<8}  {r.tool}")
    if by_tag["EMPTY"]:
        print("\n  Empty (expected for some funds/crypto):")
        for r in by_tag["EMPTY"]:
            print(f"    {r.ticker:<8}  {r.tool}")

    sys.exit(1 if by_tag["ERROR"] else 0)


if __name__ == "__main__":
    main()
