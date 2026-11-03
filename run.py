"""Interactive entry point for the agentic trading analyst pipeline.

Usage examples:
    # Single-ticker briefing (Step 5b):
    python run.py --ticker XLK --asset-type etf --date 2024-01-15 --window 30
    python run.py --ticker NVDA --date 2024-01-15 --window 7

    # Portfolio-aware rebalance (Step 6) — --portfolio activates Step 6 mode.
    # --ticker (optional) is the main / candidate ticker against the portfolio.
    python run.py --ticker NVDA --main-type equity --portfolio portfolio_test.json --date 2024-01-15

    # Pure rebalance (no candidate) — omit --ticker:
    python run.py --portfolio portfolio_test.json --date 2024-01-15

    # Fully interactive (prompts for missing values):
    python run.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

# Reports contain non-ASCII glyphs (e.g. the ★ focus marker, ─ rules). On
# Windows the console defaults to cp1252, which crashes on print(report); force
# UTF-8 so the final echo matches the UTF-8 report file written to disk.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
    pass

from dotenv import load_dotenv

load_dotenv()

from agentic.graph.portfolio_graph import PortfolioGraph
from agentic.graph.trading_graph import TradingGraph

VALID_WINDOWS = {1, 7, 30}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Agentic trading analyst — generate an investment briefing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-t", "--ticker",
                   help="Ticker symbol, e.g. NVDA, XLK, SPY")
    p.add_argument("-a", "--asset-type",
                   default=None, choices=["stock", "etf"],
                   dest="asset_type",
                   help="Asset type (default: stock)")
    p.add_argument("-d", "--date",
                   help="Analysis date YYYY-MM-DD")
    p.add_argument("-w", "--window",
                   type=int, default=None,
                   choices=sorted(VALID_WINDOWS),
                   metavar="{1,7,30}",
                   help="Lookback window in days: 1, 7, or 30 (default: 30)")
    p.add_argument("-p", "--portfolio",
                   default=None,
                   help="Path to a portfolio JSON file. When set, activates "
                        "Step 6 portfolio-rebalance mode. --ticker is OPTIONAL "
                        "here: present = candidate add; absent = pure rebalance.")
    p.add_argument("--main-type",
                   default=None,
                   choices=["equity", "equity_fund", "bond_fund", "alternative_fund"],
                   dest="main_type",
                   help="Asset class of the main/candidate ticker in portfolio "
                        "mode (default: equity, or the JSON main_ticker_type).")
    p.add_argument("--priority-threshold",
                   type=float, default=None,
                   dest="priority_threshold",
                   help="Cumulative-priority cutoff for Junior PA selection "
                        "(0..1, default 0.5). Only used with --portfolio.")
    return p.parse_args()


def prompt_choice(prompt: str, choices: list[str], default: str) -> str:
    choices_str = "/".join(choices)
    while True:
        raw = input(f"{prompt} [{choices_str}] (default: {default}): ").strip().lower()
        if not raw:
            return default
        if raw in choices:
            return raw
        print(f"  Please enter one of: {choices_str}")


def main() -> None:
    args = parse_args()

    portfolio_path = args.portfolio

    # Ticker handling differs by mode:
    #   - Single-ticker (no --portfolio): ticker is REQUIRED.
    #   - Portfolio mode: ticker is OPTIONAL (absent = pure rebalance).
    ticker = (args.ticker or "").upper()
    if not portfolio_path and not ticker:
        ticker = input("Ticker symbol (e.g. NVDA, XLK): ").strip().upper()
        if not ticker:
            print("Error: ticker is required.", file=sys.stderr)
            sys.exit(1)

    asset_type = args.asset_type
    if asset_type is None:
        asset_type = prompt_choice("Asset type", ["stock", "etf"], "stock")

    date_str = args.date
    if not date_str:
        date_str = input("Analysis date (YYYY-MM-DD, default: today): ").strip()
        if not date_str:
            date_str = date.today().isoformat()

    window = args.window
    if window is None:
        raw = input("Lookback window in days [1/7/30] (default: 7): ").strip()
        if not raw:
            window = 7
        else:
            try:
                window = int(raw)
            except ValueError:
                window = 7
            if window not in VALID_WINDOWS:
                print(f"  Invalid window {window!r}, defaulting to 7.")
                window = 7

    # Validate date format.
    try:
        date.fromisoformat(date_str)
    except ValueError:
        print(f"Error: invalid date format {date_str!r} — expected YYYY-MM-DD.", file=sys.stderr)
        sys.exit(1)

    priority_threshold = args.priority_threshold

    print(f"\n{'='*72}")
    if portfolio_path:
        print(f"  Main ticker: {ticker or '(none — pure rebalance)'}")
        if ticker and args.main_type:
            print(f"  Main type  : {args.main_type}")
    else:
        print(f"  Ticker     : {ticker}")
        print(f"  Asset type : {asset_type}")
    print(f"  Date       : {date_str}")
    print(f"  Window     : {window} day{'s' if window != 1 else ''}")
    if portfolio_path:
        print(f"  Portfolio  : {portfolio_path}")
        if priority_threshold is not None:
            print(f"  Priority threshold: {priority_threshold}")
    print(f"{'='*72}\n")
    print("Building graph and running analysis — this may take a minute...\n")

    if portfolio_path:
        # asset_type here is only the DEFAULT for tickers lacking an explicit
        # class; per-ticker routing comes from the portfolio JSON `type` field.
        cfg = {"asset_type": asset_type, "lookback_days": window}
        if priority_threshold is not None:
            cfg["priority_threshold"] = priority_threshold
        pg = PortfolioGraph(config=cfg)
        _, report = pg.propagate(
            ticker or None, portfolio_path, date_str, main_ticker_type=args.main_type
        )
        print(report)
    else:
        ta = TradingGraph(config={"asset_type": asset_type, "lookback_days": window})
        _, briefing = ta.propagate(ticker, date_str)
        print(briefing)


if __name__ == "__main__":
    main()
