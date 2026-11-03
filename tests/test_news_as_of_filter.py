"""Unit tests for the news no-look-ahead filter — no network required.

yfinance's news feed is not point-in-time (it always returns "recent"
articles); `_filter_articles_by_as_of` is the client-side enforcement point
for `as_of_date`. These tests operate on plain dicts, not live fetches.
"""

from __future__ import annotations

from agentic.dataflows.y_finance import _filter_articles_by_as_of


def _article(date: str) -> dict:
    return {"title": f"headline {date}", "summary": "", "publisher": "x",
            "url": "", "date": date}


def test_no_as_of_date_is_noop():
    articles = [_article("2026-07-20"), _article("2026-07-25"), _article("unknown date")]
    assert _filter_articles_by_as_of(articles, "") == articles


def test_drops_articles_after_as_of_date():
    articles = [_article("2026-07-20"), _article("2026-07-23"), _article("2026-07-25")]
    result = _filter_articles_by_as_of(articles, "2026-07-23")
    assert [a["date"] for a in result] == ["2026-07-20", "2026-07-23"]


def test_keeps_article_dated_exactly_on_as_of_date():
    articles = [_article("2026-07-23")]
    assert _filter_articles_by_as_of(articles, "2026-07-23") == articles


def test_drops_unparseable_date_when_filtering_active():
    articles = [_article("2026-07-20"), _article("unknown date")]
    result = _filter_articles_by_as_of(articles, "2026-07-23")
    assert [a["date"] for a in result] == ["2026-07-20"]


def test_empty_input_returns_empty():
    assert _filter_articles_by_as_of([], "2026-07-23") == []
