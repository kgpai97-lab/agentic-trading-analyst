"""Vendor routing layer — the single place that dispatches data requests.

Agentic concept: the vendor routing layer decouples *what* data an agent
needs from *where* it comes from. An agent calls `route_to_vendor("get_news",
ticker)`. This function looks up the configured vendor for news data and calls
the matching implementation. Switching from yfinance to Alpha Vantage later
is a one-line config change, not a code change.

VENDOR_METHODS maps:
    method_name → vendor_name → implementation_function

To add a new data source (e.g. "alpha_vantage"):
    1. Add the implementation function to a new alpha_vantage.py file.
    2. Add an "alpha_vantage" key under the method in VENDOR_METHODS.
    3. Set the vendor in DEFAULT_CONFIG["data_vendors"].
"""

from __future__ import annotations

from agentic.dataflows import y_finance
from agentic.dataflows.result import ToolResult

# ---------------------------------------------------------------------------
# Registry: method → vendor → implementation
# ---------------------------------------------------------------------------
VENDOR_METHODS: dict[str, dict[str, callable]] = {
    "get_news":             {"yfinance": y_finance.get_news_yfinance},
    "get_global_news":      {"yfinance": y_finance.get_global_news_yfinance},
    "get_stock_data":       {"yfinance": y_finance.get_stock_data_yfinance},
    "get_indicators":       {"yfinance": y_finance.get_indicators_yfinance},
    "get_fundamentals":     {"yfinance": y_finance.get_fundamentals_yfinance},
    "get_income_statement": {"yfinance": y_finance.get_income_statement_yfinance},
    "get_etf_holdings":     {"yfinance": y_finance.get_etf_holdings_yfinance},
}

# Maps method names to the config key that controls their vendor selection.
_METHOD_TO_CONFIG_KEY: dict[str, str] = {
    "get_news":             "news_data",
    "get_global_news":      "news_data",
    "get_stock_data":       "core_stock_apis",
    "get_indicators":       "technical_indicators",
    "get_fundamentals":     "fundamental_data",
    "get_income_statement": "fundamental_data",
    "get_etf_holdings":     "fundamental_data",
}


def route_to_vendor(method: str, *args, config: dict | None = None, **kwargs) -> ToolResult:
    """Dispatch a data-fetch call to the configured vendor implementation.

    Always returns a ToolResult — vendor functions that still return strings
    are coerced into ToolResult.success(...) at the boundary so callers do
    not have to special-case.
    """
    if config is None:
        from agentic.default_config import get_config  # noqa: PLC0415
        config = get_config()

    config_key = _METHOD_TO_CONFIG_KEY.get(method, "yfinance")
    vendor = config.get("data_vendors", {}).get(config_key, "yfinance")

    try:
        impl = VENDOR_METHODS[method][vendor]
    except KeyError:
        raise KeyError(
            f"No implementation for method={method!r} vendor={vendor!r}. "
            f"Registered: {list(VENDOR_METHODS.get(method, {}).keys())}"
        )

    result = impl(*args, **kwargs)
    if not isinstance(result, ToolResult):
        result = ToolResult.success(str(result))

    # Phase 2: write-only snapshot of the raw fetch for provenance/reproducibility.
    # snapshot_dir comes from the run-scoped DataContext first (the @tool wrappers
    # call route_to_vendor WITHOUT a config, so config["snapshot_dir"] is the
    # default and would be empty); a direct caller passing config still works as
    # a fallback. Keyed by the run's as_of_date so provenance is dated. Never
    # read back during analysis — see dataflows/snapshot.py.
    if args:
        from agentic.dataflows.data_context import (  # noqa: PLC0415
            current_as_of_date,
            current_snapshot_dir,
        )
        snapshot_dir = current_snapshot_dir() or config.get("snapshot_dir", "")
        if snapshot_dir:
            from agentic.dataflows.snapshot import write_snapshot  # noqa: PLC0415
            write_snapshot(snapshot_dir, current_as_of_date(), str(args[0]), method, result)

    return result
