"""DataContext — run-scoped fetch metadata, the no-look-ahead choke point.

Every data fetch reads from a single run-scoped context: the `as_of_date` the
run is analysing "as of", plus the lookback window and the income_alt
fund_subclass hint. This module owns that context.

Two things changed from the old module-level `_run_context` dict in
`agent_utils.py` (a documented limitation — see README "Known limitations"):

  1. **ContextVar, not a plain dict.** A `contextvars.ContextVar` is isolated per
     async task / thread, so running `propagate()` concurrently for several
     tickers no longer clobbers a shared `trade_date`. (The old dict was a
     process-global singleton; `analyst_concurrency_limit=1` was the only thing
     keeping it safe.)

  2. **`as_of_date` is first-class.** Price fetches use it as the hard upper
     bound (`end=as_of_date`) so a run dated 2024-01-15 never pulls bars printed
     after that day. This is the one cheap, exact no-look-ahead win available
     from yfinance. News / fundamentals / holdings remain "recent/current"
     snapshots (yfinance can't backdate them — see README), so for those the
     as_of_date is recorded for provenance but not enforced.

Design note (why a context, not function arguments): the LLM emits tool calls
with only the arguments in each tool's schema (e.g. `ticker`). It cannot pass
run metadata like the trade date. So the analyst node publishes that metadata
into this context before invoking the LLM, and the tool wrappers read it back.
A ContextVar survives the LangGraph node→ToolNode hop within one task.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RunContext:
    """Immutable snapshot of the current run's fetch metadata.

    as_of_date: "YYYY-MM-DD" the run analyses as-of. Empty string means "live /
        today" (no upper bound enforced — current behavior for forward runs).
    lookback_days: history window the price/indicator tools request.
    fund_subclass: income_alt only — "bond_fund" | "alternative_fund" | "".
    ticker: the ticker the current analyst node is studying (e.g. "TQQQ").
        Published by the analyst node so _normalise_ticker can snap LLM typos.
    universe: the known tradeable tickers for this run (the portfolio holdings).
        Published by PortfolioGraph so _normalise_ticker can snap a hallucinated
        cross-holding typo (e.g. "TQQ3"→"TQQQ") even when the current analyst is
        studying a DIFFERENT name. Empty tuple = no universe known.
    """

    as_of_date: str = ""
    lookback_days: int = 30
    fund_subclass: str = ""
    snapshot_dir: str = ""
    ticker: str = ""
    universe: tuple[str, ...] = ()


# The single run-scoped slot. Default is a "live" context (no as_of bound).
_ctx: contextvars.ContextVar[RunContext] = contextvars.ContextVar(
    "agentic_run_context", default=RunContext()
)


def set_run_context(
    trade_date: str = "",
    lookback_days: int = 30,
    fund_subclass: str = "",
    snapshot_dir: str = "",
    universe: tuple[str, ...] = (),
) -> None:
    """Publish run metadata for the tool wrappers to read.

    Called by analyst nodes (and TradingGraph.propagate) before invoking the
    LLM. `trade_date` is the run's as_of_date — the no-look-ahead upper bound for
    dated runs (empty = live). Kept named `trade_date` for backward-compat with
    existing callers.

    `snapshot_dir` rides the context (not config) because the @tool wrappers
    call route_to_vendor WITHOUT a config arg — the DataContext is the only
    run-scoped channel they read. Empty = snapshots off.
    """
    _ctx.set(RunContext(
        as_of_date=trade_date or "",
        lookback_days=int(lookback_days),
        fund_subclass=fund_subclass or "",
        snapshot_dir=snapshot_dir or "",
        universe=tuple(universe or ()),
    ))


def update_run_context(**changes) -> None:
    """Merge `changes` into the current context, preserving unset fields.

    The analyst node re-publishes `trade_date` on every invocation; using this
    (instead of set_run_context) means it does NOT clobber `lookback_days` /
    `fund_subclass` / `snapshot_dir` that TradingGraph.propagate set up front.
    `trade_date` is accepted as an alias for `as_of_date`.
    """
    if "trade_date" in changes:
        changes["as_of_date"] = changes.pop("trade_date") or ""
    _ctx.set(replace(_ctx.get(), **changes))


def current_snapshot_dir() -> str:
    """The run's snapshot directory ('' = snapshots disabled)."""
    return _ctx.get().snapshot_dir


def get_run_context() -> RunContext:
    """Return the current run context (a live default if none was set)."""
    return _ctx.get()


def current_as_of_date() -> str:
    """The run's as_of_date ('' = live/today, no upper bound enforced)."""
    return _ctx.get().as_of_date


def current_lookback_days() -> int:
    return _ctx.get().lookback_days


def current_fund_subclass() -> str:
    """Public reader for the income_alt analyst's task instruction."""
    return _ctx.get().fund_subclass


def current_ticker() -> str:
    """The ticker the current analyst node is studying ('' if not set)."""
    return _ctx.get().ticker


def current_universe() -> tuple[str, ...]:
    """Known tradeable tickers for this run (portfolio holdings); () if unset."""
    return _ctx.get().universe
