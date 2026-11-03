"""yfinance implementations of all data-fetch functions.

Agentic concept: the data layer is intentionally dumb — plain functions that
fetch data and return ToolResult. They know nothing about LangGraph, agents,
or state. This separation means you can unit-test data fetching without
running an LLM, and swap implementations without touching agent code.

Each function returns a ToolResult (see dataflows/result.py). The @tool
wrappers in agent_utils.py render it to a tagged string ([OK]/[EMPTY]/[ERROR])
that the LLM sees. Analyst prompts have a matching handling block.

News functions are also split into a structured layer:
    fetch_news_articles → list[dict] (structured; for future provenance logs)
    format_news_for_llm → str        (the prompt-facing rendering)
    get_news_yfinance   → ToolResult (vendor entry point)
The structured layer is exported but currently unused by tool wrappers.
"""

from __future__ import annotations

import math
from datetime import datetime
from functools import lru_cache
from typing import Any

from agentic.dataflows.result import ToolResult


# ===========================================================================
# Ticker existence — cached, used as the first line of every get_* function.
# ===========================================================================

@lru_cache(maxsize=256)
def _ticker_resolves(ticker: str) -> bool:
    """True if yfinance has any data for this symbol.

    Checks both .info (non-empty, with a real shortName/longName/symbol key)
    AND .history (non-empty). yfinance returns a near-empty dict for unknown
    symbols, so .info alone is unreliable.
    """
    try:
        import yfinance as yf  # noqa: PLC0415

        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:  # noqa: BLE001
            info = {}
        if info.get("symbol") or info.get("shortName") or info.get("longName"):
            return True
        # Fall back to a tiny history probe.
        hist = t.history(period="5d", interval="1d")
        return not hist.empty
    except Exception:  # noqa: BLE001
        return False


def _verify_ticker(ticker: str) -> ToolResult | None:
    """Return None if ticker resolves; an [EMPTY] ToolResult otherwise."""
    ticker = (ticker or "").strip()
    if not ticker:
        return ToolResult.empty(f"Empty ticker symbol.")
    if not _ticker_resolves(ticker):
        return ToolResult.empty(f"Unknown ticker {ticker!r} — no yfinance data.")
    return None


# ===========================================================================
# News
# ===========================================================================

def _extract_url(content: dict) -> str:
    """Pull the canonical article URL out of a yfinance article record."""
    candidates = [
        content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else None,
        content.get("clickThroughUrl", {}).get("url") if isinstance(content.get("clickThroughUrl"), dict) else None,
        content.get("link"),
        content.get("url"),
    ]
    for url in candidates:
        if url:
            return str(url)
    return ""


# ---------------------------------------------------------------------------
# Underlying-index "look-through" for fund tickers.
#
# A fund ticker (SPY, BND, GLD…) has thin wrapper-level news — flows, expense
# ratios, distributions — while the market-moving stories live in the index or
# asset it tracks. For known funds we additionally fetch news for a free-text
# proxy query (e.g. SPY → "S&P 500 index") via yfinance's Search endpoint and
# merge it into get_news, so the News/Sentiment analysts read the underlying's
# news, not just the wrapper's.
#
# Each entry is (search_query, human_label). The label heads the underlying
# section in the merged output so the LLM knows the source. Keep queries short
# and unambiguous — Search matches news headlines, not tickers.
# ---------------------------------------------------------------------------
_FUND_UNDERLYING_PROXY: dict[str, tuple[str, str]] = {
    # Queries are kept SHORT — yfinance's Search endpoint returns 0 results for
    # long multi-term queries (verified: "US bond market Treasury yields" → 0,
    # "Treasury bond yields" → 10). Two to three salient words is the sweet spot.
    #
    # NOTE: for a fund ticker we return ONLY this underlying news (the wrapper
    # feed — AUM milestones, fee wars, vehicle-vs-vehicle comparisons — is
    # dropped in get_news_yfinance). So the query must capture the market-moving
    # driver of the fund, not the fund vehicle itself.
    #
    # US broad equity
    "SPY":  ("S&P 500 index", "S&P 500 (underlying index)"),
    "VOO":  ("S&P 500 index", "S&P 500 (underlying index)"),
    "IVV":  ("S&P 500 index", "S&P 500 (underlying index)"),
    "SPLG": ("S&P 500 index", "S&P 500 (underlying index)"),
    "VTI":  ("US stock market", "US total stock market (underlying)"),
    "ITOT": ("US stock market", "US total stock market (underlying)"),
    "SCHB": ("US stock market", "US total stock market (underlying)"),
    "QQQ":  ("Nasdaq 100", "Nasdaq 100 (underlying index)"),
    "QQQM": ("Nasdaq 100", "Nasdaq 100 (underlying index)"),
    "TQQQ": ("Nasdaq 100", "Nasdaq 100 3x leveraged (underlying index)"),
    "DIA":  ("Dow Jones", "Dow Jones (underlying index)"),
    "IWM":  ("Russell 2000 small cap", "Russell 2000 (underlying index)"),
    "IJR":  ("small cap stocks", "US small caps (underlying)"),
    "IJH":  ("mid cap stocks", "US mid caps (underlying)"),
    "MDY":  ("mid cap stocks", "US mid caps (underlying)"),
    "VUG":  ("growth stocks", "US large-cap growth (underlying)"),
    "VTV":  ("value stocks", "US large-cap value (underlying)"),
    "RSP":  ("S&P 500 index", "S&P 500 equal-weight (underlying)"),
    # US equity sectors (SPDR Select Sector + popular peers)
    "XLK":  ("technology stocks", "Technology sector (underlying)"),
    "VGT":  ("technology stocks", "Technology sector (underlying)"),
    "SMH":  ("semiconductor stocks", "Semiconductors (underlying)"),
    "SOXX": ("semiconductor stocks", "Semiconductors (underlying)"),
    "SOXL": ("semiconductor stocks", "Semiconductors 3x leveraged (underlying)"),
    "XLF":  ("bank stocks", "Financial sector (underlying)"),
    "XLE":  ("energy stocks oil", "Energy sector (underlying)"),
    "XLV":  ("healthcare stocks", "Healthcare sector (underlying)"),
    "XLY":  ("consumer discretionary", "Consumer discretionary (underlying)"),
    "XLP":  ("consumer staples", "Consumer staples (underlying)"),
    "XLI":  ("industrial stocks", "Industrial sector (underlying)"),
    "XLB":  ("materials stocks", "Materials sector (underlying)"),
    "XLU":  ("utility stocks", "Utilities sector (underlying)"),
    "XLRE": ("real estate REITs", "Real estate sector (underlying)"),
    "XLC":  ("communication stocks", "Communication services (underlying)"),
    "XBI":  ("biotech stocks", "Biotech (underlying)"),
    "ARKK": ("innovation stocks disruptive", "ARK Innovation (underlying theme)"),
    # International equity
    "VXUS": ("international stock markets", "International ex-US equities (underlying)"),
    "VEU":  ("international stock markets", "International ex-US equities (underlying)"),
    "ACWX": ("global stock markets", "Global ex-US equities (underlying)"),
    "EFA":  ("developed markets stocks", "EAFE developed markets (underlying)"),
    "VEA":  ("developed markets stocks", "Developed ex-US markets (underlying)"),
    "IEFA": ("developed markets stocks", "Developed ex-US markets (underlying)"),
    "EEM":  ("emerging markets stocks", "Emerging markets (underlying)"),
    "VWO":  ("emerging markets stocks", "Emerging markets (underlying)"),
    "IEMG": ("emerging markets stocks", "Emerging markets (underlying)"),
    "MCHI": ("China stocks", "China equities (underlying)"),
    "FXI":  ("China stocks", "China large-caps (underlying)"),
    "EWJ":  ("Japan stocks Nikkei", "Japan equities (underlying)"),
    "EWG":  ("Germany stocks DAX", "Germany equities (underlying)"),
    "INDA": ("India stocks", "India equities (underlying)"),
    "EWZ":  ("Brazil stocks", "Brazil equities (underlying)"),
    # Bonds / rates
    "BND":  ("Treasury bond yields", "US aggregate bond market (underlying)"),
    "AGG":  ("Treasury bond yields", "US aggregate bond market (underlying)"),
    "BNDX": ("global bond yields", "International bonds (underlying)"),
    "LQD":  ("corporate bonds", "IG corporate bonds (underlying)"),
    "VCIT": ("corporate bonds", "IG corporate bonds (underlying)"),
    "HYG":  ("high yield bonds", "High-yield bonds (underlying)"),
    "JNK":  ("high yield bonds", "High-yield bonds (underlying)"),
    "TLT":  ("Treasury bond yields", "Long-dated Treasuries (underlying)"),
    "IEF":  ("Treasury bond yields", "Intermediate Treasuries (underlying)"),
    "SHY":  ("short term Treasury", "Short-dated Treasuries (underlying)"),
    "BIL":  ("Treasury bills rates", "T-bills / front-end rates (underlying)"),
    "TIP":  ("inflation bonds TIPS", "TIPS / inflation (underlying)"),
    "VTIP": ("inflation bonds TIPS", "Short-term TIPS / inflation (underlying)"),
    "MUB":  ("municipal bonds", "Municipal bonds (underlying)"),
    "EMB":  ("emerging market bonds", "EM sovereign bonds (underlying)"),
    # Commodities / alternatives
    "GLD":  ("gold price", "Gold price (underlying)"),
    "IAU":  ("gold price", "Gold price (underlying)"),
    "SLV":  ("silver price", "Silver price (underlying)"),
    "GSG":  ("commodities prices", "Broad commodities (underlying)"),
    "DBC":  ("commodities prices", "Broad commodities (underlying)"),
    "PDBC": ("commodities prices", "Broad commodities (underlying)"),
    "USO":  ("crude oil price", "Crude oil (underlying)"),
    "UNG":  ("natural gas price", "Natural gas (underlying)"),
    "DBA":  ("agriculture commodities", "Agriculture commodities (underlying)"),
    "CPER": ("copper price", "Copper (underlying)"),
    "BITO": ("Bitcoin price crypto", "Bitcoin futures ETF (underlying)"),
    "VNQ":  ("real estate REITs", "US REITs / real estate (underlying)"),
    "IYR":  ("real estate REITs", "US REITs / real estate (underlying)"),
    "SCHH": ("real estate REITs", "US REITs / real estate (underlying)"),
    "VNQI": ("global real estate", "International REITs (underlying)"),
}


def underlying_proxy_for(ticker: str) -> tuple[str, str] | None:
    """Return (search_query, label) for a known fund ticker, else None.

    Case-insensitive. Returning None means "treat as a direct security" — no
    look-through fetch happens, so direct equities (AAPL, MSFT) are unaffected.
    """
    return _FUND_UNDERLYING_PROXY.get((ticker or "").strip().upper())


def _filter_articles_by_as_of(
    articles: list[dict[str, Any]], as_of_date: str = ""
) -> list[dict[str, Any]]:
    """Drop any article dated after `as_of_date` (no-look-ahead for news).

    yfinance's news feed is NOT point-in-time — it always returns "recent"
    articles regardless of what date we're analysing as of. When `as_of_date`
    is set, this is the enforcement point: articles with a parsed `date` after
    it are dropped, and articles whose date couldn't be parsed ("unknown
    date") are ALSO dropped rather than kept — an unparseable date can't be
    proven to be on-or-before `as_of_date`, and silently including it would
    reopen the exact look-ahead leak this filter exists to close. Empty
    `as_of_date` (live/current-day runs) is a no-op — every article passes
    through unchanged, matching the pre-existing behaviour.
    """
    if not as_of_date:
        return articles
    return [a for a in articles if a.get("date", "unknown date") <= as_of_date]


def fetch_search_news(
    query: str, max_articles: int = 8, as_of_date: str = ""
) -> list[dict[str, Any]]:
    """Fetch news for a free-text query via yfinance's Search endpoint.

    Used for fund look-through (see _FUND_UNDERLYING_PROXY). Search news has a
    flatter shape than Ticker.news — title/publisher/link/providerPublishTime
    live at the top level with no `content` wrapper and no `summary`. We map it
    onto the same dict shape fetch_news_articles produces so the formatter and
    dedupe logic are shared.

    `as_of_date` ("YYYY-MM-DD"), when given, drops any article published after
    that date (see `_filter_articles_by_as_of`) — yfinance itself has no
    server-side date filter, so this is applied client-side after the fetch.
    """
    try:
        import yfinance as yf  # noqa: PLC0415

        raw = yf.Search(query, news_count=max_articles).news or []
    except Exception:  # noqa: BLE001
        return []

    articles: list[dict[str, Any]] = []
    for art in raw[:max_articles]:
        pub_time = art.get("providerPublishTime") or art.get("pubDate")
        if isinstance(pub_time, int):
            date_str = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d")
        elif isinstance(pub_time, str):
            date_str = pub_time[:10]
        else:
            date_str = "unknown date"
        articles.append({
            "title":     art.get("title", "No title"),
            "summary":   (art.get("summary") or "").strip(),  # Search rarely provides one
            "publisher": art.get("publisher") or "Unknown",
            "url":       _extract_url(art),
            "date":      date_str,
        })
    return _filter_articles_by_as_of(articles, as_of_date)


def fetch_news_articles(
    ticker: str, max_articles: int = 20, as_of_date: str = ""
) -> list[dict[str, Any]]:
    """Fetch recent news articles for a ticker as a list of structured dicts.

    `as_of_date` ("YYYY-MM-DD"), when given, drops any article published after
    that date (see `_filter_articles_by_as_of`) — yfinance's `.news` has no
    server-side date filter (it's always "recent", not point-in-time), so this
    is applied client-side after the fetch.
    """
    try:
        import yfinance as yf  # noqa: PLC0415

        stock = yf.Ticker(ticker)
        raw_news = stock.news or []
    except Exception:  # noqa: BLE001
        return []

    articles: list[dict[str, Any]] = []
    for article in raw_news[:max_articles]:
        content = article.get("content", article)
        title     = content.get("title", "No title")
        summary   = (content.get("summary") or "").strip()
        publisher = (
            content.get("provider", {}).get("displayName", "")
            or content.get("publisher", "")
            or "Unknown"
        )
        pub_time = content.get("pubDate") or content.get("providerPublishTime")
        if isinstance(pub_time, int):
            date_str = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d")
        elif isinstance(pub_time, str):
            date_str = pub_time[:10]
        else:
            date_str = "unknown date"

        articles.append({
            "title":     title,
            "summary":   summary,
            "publisher": publisher,
            "url":       _extract_url(content),
            "date":      date_str,
        })

    return _filter_articles_by_as_of(articles, as_of_date)


def format_news_for_llm(articles: list[dict[str, Any]], ticker: str) -> str:
    if not articles:
        return f"No recent news found for {ticker}."
    lines = [f"Recent news for {ticker} ({len(articles)} articles):\n"]
    for i, art in enumerate(articles, start=1):
        lines.append(f"{i}. [{art['date']}] {art['title']}  —  {art['publisher']}")
        if art.get("summary"):
            lines.append(f"   {art['summary']}")
    return "\n".join(lines)


def get_news_yfinance(
    ticker: str, max_articles: int = 20, as_of_date: str = ""
) -> ToolResult:
    """Vendor entry point: fetch + format ticker-specific news.

    Fund look-through: if `ticker` is a known fund (see _FUND_UNDERLYING_PROXY),
    we return ONLY news for the underlying index/asset it tracks (e.g.
    SPY → "S&P 500 index"). The wrapper feed (AUM milestones, fee wars,
    vehicle-vs-vehicle comparisons) is DROPPED entirely — for a passive holder
    those headlines are noise that crowds out the market-moving underlying news.
    Direct equities have no proxy → their own feed is returned unchanged.

    `as_of_date` ("YYYY-MM-DD"), when given, drops any article published after
    that date — a client-side no-look-ahead filter (yfinance's own feed is
    always "recent", not point-in-time; see `_filter_articles_by_as_of`).
    """
    bad = _verify_ticker(ticker)
    if bad is not None:
        return bad
    try:
        proxy = underlying_proxy_for(ticker)

        # Fund ticker → underlying-only. Skip the wrapper feed completely.
        if proxy is not None:
            query, underlying_label = proxy
            underlying = fetch_search_news(
                query, max_articles=max_articles, as_of_date=as_of_date)
            if not underlying:
                return ToolResult.empty(
                    f"No underlying news found for {ticker} "
                    f"(tracked exposure: {underlying_label})."
                )
            section = (
                f"--- Underlying look-through: {underlying_label} "
                f"({len(underlying)} articles) ---\n"
                f"(Wrapper-level fund news for {ticker} is intentionally "
                f"omitted; treat the underlying exposure as primary.)\n\n"
                + format_news_for_llm(underlying, underlying_label)
            )
            return ToolResult.success(section, article_count=len(underlying))

        # Direct security → its own feed, unchanged.
        articles = fetch_news_articles(
            ticker, max_articles=max_articles, as_of_date=as_of_date)
        if not articles:
            return ToolResult.empty(f"No recent news found for {ticker}.")
        return ToolResult.success(
            format_news_for_llm(articles, ticker),
            article_count=len(articles),
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"Error fetching news for {ticker}: {exc}")


def get_global_news_yfinance(
    ticker: str, max_articles: int = 20, as_of_date: str = ""
) -> ToolResult:
    """Fetch broader macro news, deduped against the ticker's own headlines.

    yfinance has no clean "global news" endpoint, so we pull news for two
    macro proxies (SPY for US equities, ^VIX for risk sentiment) and merge.
    We then subtract any titles already returned by get_news(ticker) so the
    News Analyst sees genuinely different content from each tool call.

    We do NOT verify `ticker` here — it's only used as context labeling.
    Empty-merge falls through to an [EMPTY] result.

    `as_of_date` ("YYYY-MM-DD"), when given, drops any article published
    after that date from every underlying fetch (ticker feed + both macro
    proxies) before dedup/merge.
    """
    try:
        # Build the set of titles the ticker's own feed already covers. For a
        # fund ticker this is its underlying-index news (wrapper feed is dropped
        # everywhere now); for a direct equity it's the wrapper-free .news feed.
        proxy = underlying_proxy_for(ticker)
        if proxy is not None:
            ticker_articles = fetch_search_news(
                proxy[0], max_articles=max_articles, as_of_date=as_of_date)
        else:
            ticker_articles = fetch_news_articles(
                ticker, max_articles=max_articles, as_of_date=as_of_date)
        ticker_titles: set[str] = {a["title"] for a in ticker_articles}

        # Macro context: broad-market mood (S&P 500) + risk gauge (VIX). Route
        # the S&P portion through the underlying-index search so it's true market
        # news, not SPY-the-vehicle headlines.
        spy_articles = fetch_search_news(
            "S&P 500 index", max_articles=max_articles, as_of_date=as_of_date)
        vix_articles = fetch_news_articles(
            "^VIX", max_articles=max_articles // 2, as_of_date=as_of_date)

        seen_titles: set[str] = set(ticker_titles)  # seed with ticker titles to exclude them
        merged: list[dict[str, Any]] = []
        for art in spy_articles + vix_articles:
            if art["title"] in seen_titles:
                continue
            seen_titles.add(art["title"])
            merged.append(art)

        if not merged:
            return ToolResult.empty(
                f"No additional global / macro news found beyond {ticker}'s own headlines."
            )

        lines = [f"Global / macro news ({len(merged)} articles, context: {ticker}):\n"]
        for i, art in enumerate(merged[:max_articles], start=1):
            lines.append(f"{i}. [{art['date']}] {art['title']}  —  {art['publisher']}")
            if art.get("summary"):
                lines.append(f"   {art['summary']}")
        return ToolResult.success("\n".join(lines), article_count=len(merged))
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"Error fetching global news: {exc}")


# ===========================================================================
# Market — prices + indicators
# ===========================================================================

def _fetch_history(ticker: str, lookback_days: int, as_of_date: str = "",
                   buffer_days: int = 5):
    """Fetch daily OHLCV history, honouring an optional as_of upper bound.

    No-look-ahead (Phase 2): when `as_of_date` ("YYYY-MM-DD") is given, we bound
    the fetch with start/end so NO bar printed after as_of_date is returned —
    `end` is as_of_date + 1 day (yfinance's `end` is exclusive) and `start` is
    far enough back to cover lookback_days of *trading* days (we pad calendar
    days generously for weekends/holidays). When `as_of_date` is empty (live
    forward runs), we keep the original `period=` fetch — current behaviour.

    Returns a DataFrame (possibly empty). Callers tail() to lookback_days.
    """
    import yfinance as yf  # noqa: PLC0415

    t = yf.Ticker(ticker)
    if as_of_date:
        from datetime import timedelta  # noqa: PLC0415
        try:
            end_d = datetime.strptime(as_of_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            # Malformed date → fall back to live period fetch rather than crash.
            return t.history(period=f"{lookback_days + buffer_days}d", interval="1d")
        # ~1.6 calendar days per trading day covers weekends + holidays.
        span = int((lookback_days + buffer_days) * 1.6) + 7
        start_d = end_d - timedelta(days=span)
        end_excl = end_d + timedelta(days=1)   # yfinance end is exclusive
        return t.history(start=start_d.isoformat(), end=end_excl.isoformat(),
                         interval="1d")
    return t.history(period=f"{lookback_days + buffer_days}d", interval="1d")


def get_stock_data_yfinance(ticker: str, lookback_days: int = 60,
                            as_of_date: str = "") -> ToolResult:
    """Return a text table of the last `lookback_days` of OHLCV data.

    `as_of_date` (no-look-ahead): when set, no bar after that date is included.
    """
    bad = _verify_ticker(ticker)
    if bad is not None:
        return bad
    try:
        # Fetch a bit more than requested so we can compute % change vs prior close
        # without losing the first row. as_of_date bounds the upper edge.
        hist = _fetch_history(ticker, lookback_days, as_of_date)
        if hist.empty:
            return ToolResult.empty(f"No price history available for {ticker}.")

        hist = hist.tail(lookback_days).copy()
        hist["PctChg"] = hist["Close"].pct_change() * 100.0

        # Compact summary first, then a sample of recent rows.
        first_close = hist["Close"].iloc[0]
        last_close  = hist["Close"].iloc[-1]
        period_pct  = (last_close / first_close - 1.0) * 100.0
        hi  = hist["High"].max()
        lo  = hist["Low"].min()

        lines = [
            f"OHLCV for {ticker} — last {len(hist)} trading days ({hist.index[0].date()} → {hist.index[-1].date()}):",
            f"  Period return : {period_pct:+.2f}%",
            f"  Period high   : {hi:,.2f}",
            f"  Period low    : {lo:,.2f}",
            f"  Last close    : {last_close:,.2f}",
            "",
            "Recent rows (most recent last):",
            f"{'Date':<12}{'Open':>10}{'High':>10}{'Low':>10}{'Close':>10}{'Volume':>14}{'%Chg':>8}",
        ]
        # Show last 10 rows in the table — full lookback is summarised above.
        for idx, row in hist.tail(10).iterrows():
            pct = row["PctChg"]
            pct_str = f"{pct:+.2f}" if pct == pct else "  —  "  # NaN check
            lines.append(
                f"{idx.date()!s:<12}"
                f"{row['Open']:>10.2f}{row['High']:>10.2f}{row['Low']:>10.2f}"
                f"{row['Close']:>10.2f}{int(row['Volume']):>14,}{pct_str:>8}"
            )
        return ToolResult.success("\n".join(lines), row_count=len(hist))
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"Error fetching price data for {ticker}: {exc}")


def _rsi(close, period: int = 14):
    """Wilder's RSI on a pandas Series of closes."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    # Wilder smoothing ≡ EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_indicators_yfinance(ticker: str, lookback_days: int = 365,
                            as_of_date: str = "") -> ToolResult:
    """Compute SMA20/50/200, RSI14, MACD on daily closes and return a summary.

    `as_of_date` (no-look-ahead): when set, indicators are computed only from
    bars on/before that date (the last close in-window becomes the "as of" row).
    """
    bad = _verify_ticker(ticker)
    if bad is not None:
        return bad
    try:
        # Need at least ~250 trading days for the 200-day SMA; clamp upward.
        # For short windows (1d / 7d), still fetch enough history to compute
        # the indicators — but report them "as of" the last available close
        # within the window. The SMA200 will be NaN for windows < ~250 days.
        # _fetch_history bounds the upper edge at as_of_date when provided.
        fetch_days = max(lookback_days, 250)
        hist = _fetch_history(ticker, fetch_days, as_of_date, buffer_days=0)
        if hist.empty:
            return ToolResult.empty(
                f"No price history available for {ticker} — cannot compute indicators."
            )

        close = hist["Close"]
        sma20  = close.rolling(20).mean()
        sma50  = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist   = macd_line - signal_line

        rsi = _rsi(close, 14)

        last_close = close.iloc[-1]
        last_sma20, last_sma50, last_sma200 = sma20.iloc[-1], sma50.iloc[-1], sma200.iloc[-1]
        last_rsi = rsi.iloc[-1]
        last_macd, last_sig, last_hist = macd_line.iloc[-1], signal_line.iloc[-1], macd_hist.iloc[-1]

        def _trend_vs(name: str, ma: float) -> str:
            if ma != ma:  # NaN
                return f"{name}: insufficient history"
            side = "above" if last_close > ma else "below"
            return f"{name}: {ma:,.2f}  (price is {side})"

        rsi_note = (
            "overbought (>70)" if last_rsi > 70
            else "oversold (<30)" if last_rsi < 30
            else "neutral"
        )
        macd_note = "bullish crossover" if last_hist > 0 else "bearish crossover"

        lines = [
            f"Technical indicators for {ticker} (as of {hist.index[-1].date()}):",
            f"  Last close    : {last_close:,.2f}",
            f"  {_trend_vs('SMA20', last_sma20)}",
            f"  {_trend_vs('SMA50', last_sma50)}",
            f"  {_trend_vs('SMA200', last_sma200)}",
            f"  RSI14         : {last_rsi:.1f}  ({rsi_note})",
            f"  MACD          : {last_macd:+.3f}   signal: {last_sig:+.3f}   hist: {last_hist:+.3f}  ({macd_note})",
        ]
        return ToolResult.success("\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"Error computing indicators for {ticker}: {exc}")


# ===========================================================================
# Fundamentals
# ===========================================================================

def _fmt_large(n) -> str:
    """Render large numbers in human form: 3_456_000_000 → '3.46B'."""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(n):
        return "n/a"
    sign = "-" if n < 0 else ""
    n = abs(n)
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            return f"{sign}{n / scale:.2f}{unit}"
    return f"{sign}{n:.2f}"


def _fmt_pct(x) -> str:
    try:
        v = float(x)
        if math.isnan(v):
            return "n/a"
        return f"{v * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(x, digits: int = 2) -> str:
    try:
        v = float(x)
        if math.isnan(v):
            return "n/a"
        return f"{v:.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _is_present(v) -> bool:
    """True if a value from .info is worth rendering (not None, not NaN, not zero-ish)."""
    if v is None or v == "":
        return False
    try:
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return bool(v)


def _fmt_dividend_yield(info: dict) -> str:
    """Return a human-readable dividend yield string.

    yfinance uses two inconsistent scales:
      trailingAnnualDividendYield — proper fraction (0.00334 = 0.33%)
      dividendYield               — already in percentage points (3.94 = 3.94%)
    Prefer the trailing annual value; fall back to dividendYield rendered as-is.
    """
    tay = info.get("trailingAnnualDividendYield")
    if tay is not None:
        return _fmt_pct(tay)  # multiply by 100
    dy = info.get("dividendYield")
    if dy is not None:
        try:
            return f"{float(dy):.2f}%"  # already in %
        except (TypeError, ValueError):
            pass
    return "n/a"


_KEY_FUNDAMENTAL_FIELDS = (
    "marketCap", "trailingPE", "forwardPE", "trailingEps",
    "profitMargins", "revenueGrowth", "returnOnEquity",
    "freeCashflow", "earningsGrowth",
)


def get_fundamentals_yfinance(ticker: str) -> ToolResult:
    """Pull a fundamentals snapshot from yfinance .info."""
    bad = _verify_ticker(ticker)
    if bad is not None:
        return bad
    try:
        import yfinance as yf  # noqa: PLC0415

        info = yf.Ticker(ticker).info or {}
        if not info:
            return ToolResult.empty(f"No fundamentals data available for {ticker}.")

        # If every key fundamental is missing, this is effectively empty
        # (common for crypto and some small-caps). Signal that to the LLM
        # rather than rendering "n/a" across the board.
        if not any(info.get(k) not in (None, "") for k in _KEY_FUNDAMENTAL_FIELDS):
            return ToolResult.empty(
                f"Fundamentals for {ticker} are sparse — none of "
                "market cap, P/E, EPS, margins, growth, or ROE are reported."
            )

        lines = [
            f"Fundamentals snapshot for {ticker} ({info.get('shortName') or info.get('longName') or '—'}):",
            f"  Sector / Industry : {info.get('sector', 'n/a')} / {info.get('industry', 'n/a')}",
            "",
            "Valuation:",
            f"  Market cap        : {_fmt_large(info.get('marketCap'))}",
            f"  Trailing P/E      : {_fmt_num(info.get('trailingPE'))}",
            f"  Forward P/E       : {_fmt_num(info.get('forwardPE'))}",
            f"  Trailing EPS      : {_fmt_num(info.get('trailingEps'))}",
            f"  Dividend yield    : {_fmt_dividend_yield(info)}",
            f"  52-week range     : {_fmt_num(info.get('fiftyTwoWeekLow'))} – {_fmt_num(info.get('fiftyTwoWeekHigh'))}",
            f"  Beta              : {_fmt_num(info.get('beta'))}",
        ]

        # Extended valuation — only add lines where data exists
        ext_val = []
        if _is_present(info.get("pegRatio")):
            ext_val.append(f"  PEG ratio         : {_fmt_num(info.get('pegRatio'))}")
        if _is_present(info.get("enterpriseValue")):
            ext_val.append(f"  Enterprise value  : {_fmt_large(info.get('enterpriseValue'))}")
        if _is_present(info.get("enterpriseToEbitda")):
            ext_val.append(f"  EV / EBITDA       : {_fmt_num(info.get('enterpriseToEbitda'))}")
        if _is_present(info.get("enterpriseToRevenue")):
            ext_val.append(f"  EV / Revenue      : {_fmt_num(info.get('enterpriseToRevenue'))}")
        if _is_present(info.get("ebitdaMargins")):
            ext_val.append(f"  EBITDA margin     : {_fmt_pct(info.get('ebitdaMargins'))}")
        lines.extend(ext_val)

        lines += [
            "",
            "Profitability & growth:",
            f"  Profit margin     : {_fmt_pct(info.get('profitMargins'))}",
            f"  Revenue growth YoY: {_fmt_pct(info.get('revenueGrowth'))}",
            f"  Return on equity  : {_fmt_pct(info.get('returnOnEquity'))}",
            f"  Debt / equity     : {_fmt_num(info.get('debtToEquity'))}",
        ]
        if _is_present(info.get("earningsGrowth")):
            lines.append(f"  Earnings growth YoY    : {_fmt_pct(info.get('earningsGrowth'))}")
        if _is_present(info.get("earningsQuarterlyGrowth")):
            lines.append(f"  Earnings growth (qtr)  : {_fmt_pct(info.get('earningsQuarterlyGrowth'))}")

        lines += [
            "",
            "Cash flow & balance sheet:",
        ]
        if _is_present(info.get("freeCashflow")):
            lines.append(f"  Free cash flow    : {_fmt_large(info.get('freeCashflow'))}")
        if _is_present(info.get("operatingCashflow")):
            lines.append(f"  Operating CF      : {_fmt_large(info.get('operatingCashflow'))}")
        if _is_present(info.get("totalCash")):
            lines.append(f"  Cash & equivalents: {_fmt_large(info.get('totalCash'))}")
        if _is_present(info.get("totalDebt")):
            lines.append(f"  Total debt        : {_fmt_large(info.get('totalDebt'))}")
        if _is_present(info.get("currentRatio")):
            lines.append(f"  Current ratio     : {_fmt_num(info.get('currentRatio'))}")
        if _is_present(info.get("quickRatio")):
            lines.append(f"  Quick ratio       : {_fmt_num(info.get('quickRatio'))}")

        return ToolResult.success("\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"Error fetching fundamentals for {ticker}: {exc}")


def get_income_statement_yfinance(ticker: str) -> ToolResult:
    """Pull annual + most-recent-quarter income statement from yfinance."""
    bad = _verify_ticker(ticker)
    if bad is not None:
        return bad
    try:
        import yfinance as yf  # noqa: PLC0415

        t = yf.Ticker(ticker)
        stmt = t.income_stmt
        if stmt is None or stmt.empty:
            return ToolResult.empty(f"No income statement available for {ticker}.")

        # Full P&L row set in logical order; only rows present in stmt are rendered.
        wanted = [
            "Total Revenue",
            "Cost Of Revenue",
            "Gross Profit",
            "Research And Development",
            "Selling General And Administration",
            "Operating Income",
            "EBITDA",
            "Interest Expense Non Operating",
            "Pretax Income",
            "Tax Provision",
            "Net Income",
            "Diluted EPS",
            "Basic EPS",
        ]

        # yfinance returns columns = period-ending dates (newest first)
        cols = list(stmt.columns)[:4]  # up to 4 most-recent fiscal years

        lines = [f"Annual income statement for {ticker} (most recent {len(cols)} fiscal years):"]
        header = "  " + " " * 28 + "".join(f"{c.date()!s:>16}" for c in cols)
        lines.append(header)

        rendered_rows = 0
        for item in wanted:
            if item not in stmt.index:
                continue
            row = stmt.loc[item]
            cells = "".join(f"{_fmt_large(row[c]):>16}" for c in cols)
            lines.append(f"  {item:<28}{cells}")
            rendered_rows += 1

        if rendered_rows == 0:
            return ToolResult.empty(f"Income statement for {ticker} had no recognisable line items.")

        # Most-recent quarter — best-effort, never breaks the annual output
        try:
            q = t.quarterly_income_stmt
            if q is not None and not q.empty:
                qcol = q.columns[0]
                lines += ["", f"Most recent quarter ({qcol.date()}):"]
                for item in wanted:
                    if item not in q.index:
                        continue
                    val = _fmt_large(q.loc[item, qcol])
                    lines.append(f"  {item:<28}{val:>16}")
        except Exception:  # noqa: BLE001
            pass

        return ToolResult.success("\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"Error fetching income statement for {ticker}: {exc}")


# ===========================================================================
# ETF composition
# ===========================================================================

def _humanise_sector(key: str) -> str:
    """yfinance sector keys are lower_snake — return Title Case for the LLM."""
    return key.replace("_", " ").title()


def get_etf_holdings_yfinance(ticker: str, top_n: int = 10) -> ToolResult:
    """Top holdings, sector weights, asset-class breakdown, and fund profile for an ETF."""
    bad = _verify_ticker(ticker)
    if bad is not None:
        return bad
    try:
        import yfinance as yf  # noqa: PLC0415

        t = yf.Ticker(ticker)
        try:
            fd = t.funds_data
        except Exception:  # noqa: BLE001
            return ToolResult.empty(
                f"{ticker} does not appear to be a fund/ETF (no funds_data available)."
            )

        if fd is None:
            return ToolResult.empty(
                f"{ticker} does not appear to be a fund/ETF (no funds_data available)."
            )

        # --- Fund overview (category / family / type) ----------------------
        overview = fd.fund_overview or {}
        category   = overview.get("categoryName", "n/a")
        family     = overview.get("family", "n/a")
        legal_type = overview.get("legalType", "n/a")

        # --- Asset classes --------------------------------------------------
        ac = fd.asset_classes or {}
        stock_pct = float(ac.get("stockPosition", 0.0)) * 100.0
        bond_pct  = float(ac.get("bondPosition",  0.0)) * 100.0
        cash_pct  = float(ac.get("cashPosition",  0.0)) * 100.0

        # --- Top holdings ---------------------------------------------------
        holdings_df = fd.top_holdings
        no_holdings = holdings_df is None or holdings_df.empty
        if no_holdings:
            # Diagnose why holdings are absent so the analyst can reason correctly.
            if stock_pct == 0.0 and bond_pct == 0.0:
                holdings_note = (
                    "  Holdings breakdown unavailable. All three yfinance asset-class "
                    "positions (stocks/bonds/cash) are 0% — this fund likely holds a "
                    "physical asset (e.g. gold bullion, commodity futures) or another "
                    "instrument not classified by yfinance. Base your analysis on the "
                    "category name and sector weightings below."
                )
            elif bond_pct > 50.0:
                holdings_note = (
                    "  Individual bond holdings are not provided by yfinance for this "
                    "fund. The asset mix confirms it is primarily a bond fund "
                    f"({bond_pct:.1f}% bonds). Use category, family, and sector "
                    "weightings below to characterise duration/credit tilt."
                )
            else:
                holdings_note = (
                    "  Holdings breakdown unavailable from yfinance for this fund."
                )
            holdings_lines = [holdings_note]
            top_weight_sum = 0.0
        else:
            df = holdings_df.head(top_n)
            holdings_lines = []
            for symbol, row in df.iterrows():
                name = row.get("Name", "")
                pct  = float(row.get("Holding Percent", 0.0)) * 100.0
                holdings_lines.append(f"  {symbol:<8} {pct:>6.2f}%   {name}")
            top_weight_sum = float(df["Holding Percent"].sum()) * 100.0

        # --- Sector weightings (drop zeros) ---------------------------------
        sw = fd.sector_weightings or {}
        nonzero_sectors = [
            (_humanise_sector(k), float(v) * 100.0)
            for k, v in sw.items()
            if v and float(v) > 0.0
        ]
        nonzero_sectors.sort(key=lambda kv: -kv[1])
        if nonzero_sectors:
            sector_lines = [f"  {name:<22} {pct:>6.2f}%" for name, pct in nonzero_sectors]
        else:
            sector_lines = ["  (no sector weightings available)"]

        holdings_header = (
            f"Top {min(top_n, len(holdings_lines))} holdings (sum of weights: {top_weight_sum:.2f}%):"
            if not no_holdings else "Holdings:"
        )
        lines = [
            f"ETF composition for {ticker}:",
            f"  Category   : {category}",
            f"  Family     : {family}",
            f"  Type       : {legal_type}",
            f"  Asset mix  : stocks {stock_pct:.1f}%   bonds {bond_pct:.1f}%   cash {cash_pct:.1f}%",
            "",
            holdings_header,
            *holdings_lines,
            "",
            "Sector weightings (non-zero):",
            *sector_lines,
        ]

        # --- Mandate description --------------------------------------------
        try:
            desc = (fd.description or "").strip()
            if desc:
                lines += ["", "Mandate:", f"  {desc[:300]}{'...' if len(desc) > 300 else ''}"]
        except Exception:  # noqa: BLE001
            pass

        # --- Bond profile (bond funds only) ---------------------------------
        if bond_pct > 50.0:
            try:
                bh = fd.bond_holdings
                ratings = fd.bond_ratings or {}

                def _bval(df, label):
                    try:
                        v = df.loc[label].iloc[0]
                        return "n/a" if (v is None or (hasattr(v, '__float__') and math.isnan(float(v)))) else f"{float(v):.2f} years"
                    except Exception:  # noqa: BLE001
                        return "n/a"

                maturity = _bval(bh, "Maturity")
                duration = _bval(bh, "Duration")

                bond_lines = [
                    "",
                    "Bond profile:",
                    f"  Average maturity : {maturity}",
                    f"  Average duration : {duration}",
                ]

                # Credit quality breakdown — ordered from highest to lowest quality
                rating_order = [
                    ("aaa",         "AAA"),
                    ("aa",          "AA"),
                    ("a",           "A"),
                    ("bbb",         "BBB"),
                    ("bb",          "BB"),
                    ("b",           "B"),
                    ("below_b",     "Below B"),
                    ("us_government","US Government"),
                    ("other",       "Other"),
                ]
                nonzero_ratings = [
                    (label, float(ratings[key]) * 100.0)
                    for key, label in rating_order
                    if ratings.get(key, 0.0) and float(ratings[key]) > 0.0
                ]
                if nonzero_ratings:
                    bond_lines.append("")
                    bond_lines.append("Credit quality breakdown:")
                    for label, pct in nonzero_ratings:
                        bond_lines.append(f"  {label:<16} {pct:>6.2f}%")

                lines.extend(bond_lines)
            except Exception:  # noqa: BLE001
                pass

        return ToolResult.success("\n".join(lines))

    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"Error fetching ETF composition for {ticker}: {exc}")
