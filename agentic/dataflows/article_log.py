"""Human-readable article provenance log.

Agentic concept: tool outputs that reach the LLM are intentionally lossy —
URLs and other metadata are stripped to keep the prompt focused. But for
human review (and later, for auditing what the agent actually saw on a given
date), we want the full record. This module writes that record to a JSON
file outside the LLM's view.

Layout:
    logs/<TICKER>/<YYYY-MM-DD>.json

One file per (ticker, trade_date) run. Re-running overwrites — the file
always reflects the most recent fetch for that pair, which matches how we
think about a briefing being a snapshot for a date.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Project root = three levels up from this file (agentic/dataflows/article_log.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOGS_DIR = _PROJECT_ROOT / "logs"


def write_article_log(
    ticker: str,
    trade_date: str,
    articles: list[dict[str, Any]],
    source_tool: str = "get_news",
) -> Path:
    """Persist the structured article list for one run to logs/<ticker>/<date>.json.

    Args:
        ticker:      Symbol the news was fetched for (used in the path).
        trade_date:  The trade date this briefing is for (used in the filename).
        articles:    List of {title, publisher, url, date} dicts.
        source_tool: Which tool produced these records (e.g. "get_news").

    Returns:
        Path to the file that was written.
    """
    ticker_dir = _LOGS_DIR / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)

    out_path = ticker_dir / f"{trade_date}.json"

    payload = {
        "ticker":      ticker.upper(),
        "trade_date":  trade_date,
        "source_tool": source_tool,
        "fetched_at":  datetime.now().isoformat(timespec="seconds"),
        "n_articles":  len(articles),
        "articles":    articles,
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
