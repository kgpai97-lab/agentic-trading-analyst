"""Shared factory for tool-loop analyst nodes.

All four analysts (sentiment, market, news, fundamentals) follow the same shape:

    system prompt + human task → LLM (with tools bound) → either:
        (a) tool_calls → ToolNode runs tool → loop back
        (b) plain text → msg-clear node writes the report field → next analyst

The only differences between them are:
  • which prompt file they load
  • which tool list they bind
  • what the human task message says (which mentions which tools to call)

This module factors that out so each analyst file stays a thin three-liner.
"""

from __future__ import annotations

from datetime import date as _date
from pathlib import Path
from typing import Callable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from agentic.agents.portfolio._common import extract_json_block
from agentic.agents.utils.agent_states import AgentState
from agentic.agents.utils.signals import AnalystSignal, AnalystSignalExtract, Claim

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"


def load_prompt(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 1b — two-call structured extraction (gpt-oss-safe)
#
# Call 1 is the analyst's normal free-form prose run (untouched, full reasoning
# quality). Call 2 is THIS: a cheap, constrained call whose only job is
# prose → AnalystSignal. Splitting the two keeps the reasoning-first gpt-oss
# model from hitting its empty-content failure mode under a hard structured
# constraint, and isolates the fragile part into a small retryable call.
#
# We extract into AnalystSignalExtract (no `made_on`, no checker fields), then
# stamp `made_on = trade_date` and attach the verbatim prose in code — the model
# never invents a date, and the prose is never paraphrased.
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = (
    "You convert a financial analyst's written report into a compact structured "
    "signal. Do NOT add new analysis — only distill what the report already says.\n"
    "- direction: a single number in [-1, 1]. -1 = strongly bearish on {ticker}, "
    "0 = neutral/mixed, +1 = strongly bullish. Reflect the report's overall stance.\n"
    "- confidence: a single number in [0, 1] for how strongly/clearly the report "
    "commits to that stance (hedged/uncertain = low, emphatic = high).\n"
    "- horizon: the report's time horizon as one of 1W, 2W, 1M, 3M, 6M, 1Y "
    "(default 1M if unstated).\n"
    "- key_evidence: a list of the report's individually checkable claims. For "
    "each: a one-sentence falsifiable `statement`, the `ticker` it concerns "
    "(default {ticker}), the `metric` it bears on (e.g. return, vol, earnings, "
    "margin), a `direction` of positive|negative|neutral, and a `horizon`. "
    "Include only claims the report actually makes; omit vague filler."
)

_EXTRACTION_HUMAN = (
    "Report on {ticker} (as of {trade_date}):\n\n{prose}\n\n"
    "Return the structured signal."
)

# Used by the JSON-mode fallback when with_structured_output fails to produce
# parseable JSON (gpt-oss tends to answer with markdown tables/prose instead).
# We re-ask with an explicit shape and parse it ourselves with extract_json_block.
_JSON_FALLBACK_INSTRUCTION = (
    "Return ONLY a single JSON object, no prose, no markdown, no code fences. "
    "Shape:\n"
    '{{"direction": <number -1..1>, "confidence": <number 0..1>, '
    '"horizon": "1W|2W|1M|3M|6M|1Y", "key_evidence": '
    '[{{"statement": "...", "ticker": "{ticker}", "metric": "...", '
    '"direction": "positive|negative|neutral", "horizon": "1M"}}]}}'
)


def _build_signal(
    ext: AnalystSignalExtract, prose: str, ticker: str, made_on: _date,
) -> AnalystSignal:
    """Map a validated AnalystSignalExtract onto a full AnalystSignal."""
    claims = [
        Claim(
            statement=c.statement, ticker=c.ticker or ticker,
            metric=c.metric, direction=c.direction,
            horizon=c.horizon, made_on=made_on,
        )
        for c in ext.key_evidence
    ]
    return AnalystSignal(
        direction=ext.direction, confidence=ext.confidence,
        horizon=ext.horizon, key_evidence=claims, prose=prose,
    )


def _neutral_signal(prose: str, horizon: str = "1M") -> AnalystSignal:
    """Fallback when extraction fails after retry: a no-information signal.

    direction=0 / confidence=0 means "no usable directional view"; the ensemble
    (Phase 6) weights by confidence, so a failed extraction contributes nothing
    rather than corrupting the combined signal. The prose is preserved so the
    human report is never lost.
    """
    return AnalystSignal(direction=0.0, confidence=0.0, horizon=horizon,
                         key_evidence=[], prose=prose)


def extract_signal(
    llm: BaseChatModel,
    prose: str,
    ticker: str,
    trade_date: str,
    node_name: str,
) -> AnalystSignal:
    """Call 2 of the two-call pattern: prose → validated AnalystSignal.

    A single bounded retry. On total failure returns a neutral signal (never
    raises) so a flaky extraction can't take down an analyst's main output.

    `made_on` is stamped from `trade_date` (parsed; falls back to today if the
    date string is malformed) and the verbatim prose is attached — neither is
    asked of the model.
    """
    try:
        made_on = _date.fromisoformat(trade_date)
    except (ValueError, TypeError):
        made_on = _date.today()

    messages = [
        SystemMessage(content=_EXTRACTION_SYSTEM.format(ticker=ticker)),
        HumanMessage(content=_EXTRACTION_HUMAN.format(
            ticker=ticker, trade_date=trade_date, prose=prose)),
    ]
    cfg = {"tags": [f"{node_name}:extract"],
           "metadata": {"analyst": node_name, "phase": "extract"}}

    last_err: Exception | None = None

    # Attempt 1 + retry: native structured output (cleanest when it works).
    for _attempt in range(2):  # one initial try + one retry
        try:
            # Inside the try so a model lacking with_structured_output (e.g. a
            # test double) degrades to the JSON fallback rather than raising.
            structured = llm.with_structured_output(AnalystSignalExtract)
            ext: AnalystSignalExtract = structured.invoke(messages, config=cfg)
            return _build_signal(ext, prose, ticker, made_on)
        except Exception as e:  # noqa: BLE001 — extraction is best-effort
            last_err = e
            continue

    # Fallback: gpt-oss often ignores the structured-output constraint and
    # answers with a markdown table / prose, which the JSON parser rejects.
    # Re-ask in plain mode with an explicit JSON shape and parse it ourselves —
    # this recovers most of those cases instead of discarding real signal.
    fallback_messages = [
        SystemMessage(content=_EXTRACTION_SYSTEM.format(ticker=ticker)),
        HumanMessage(content=_EXTRACTION_HUMAN.format(
            ticker=ticker, trade_date=trade_date, prose=prose)),
        HumanMessage(content=_JSON_FALLBACK_INSTRUCTION.format(ticker=ticker)),
    ]
    try:
        resp = llm.invoke(fallback_messages, config=cfg)
        data = extract_json_block(getattr(resp, "content", "") or "")
        ext = AnalystSignalExtract.model_validate(data)
        return _build_signal(ext, prose, ticker, made_on)
    except Exception as e:  # noqa: BLE001
        last_err = e

    # Everything failed (empty content, unparseable, schema violation, or
    # transport error). Degrade to a neutral signal rather than crash the analyst.
    print(f"[{node_name}] signal extraction failed after retry + JSON fallback: {last_err}")
    return _neutral_signal(prose)


def make_analyst_node(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
    task_instruction: str,
    node_name: str,
) -> Callable[[AgentState], dict]:
    """Build a LangGraph node function for a tool-loop analyst.

    Args:
        llm:              Pre-built LangChain chat model.
        tools:            The TOOLS_<NAME> list to bind for this analyst.
        system_prompt:    Loaded prompt text (analyst persona + report structure).
        task_instruction: One-line phrasing of the per-run task. Receives no .format()
                          arguments here — formatting happens inside the node so
                          per-state fields like ticker and date can be injected.
                          Use "{ticker}" and "{trade_date}" placeholders.
        node_name:        String written to state["sender"] for traceability.
    """
    # Lazy import to dodge a circular import (agent_utils imports nothing from
    # analysts, but analyst -> agent_utils -> ... could chain back here through
    # tools.) Importing inside the factory keeps the wiring simple.
    from agentic.dataflows.data_context import update_run_context  # noqa: PLC0415

    llm_with_tools = llm.bind_tools(list(tools))

    def node_fn(state: AgentState) -> dict:
        # Merge (not replace) so lookback_days / fund_subclass / snapshot_dir
        # that TradingGraph.propagate published up front survive this re-set.
        # Also publish the ticker so _normalise_ticker can snap LLM typos back.
        update_run_context(trade_date=state["trade_date"],
                           ticker=state.get("company_of_interest", ""))

        messages = list(state["messages"])

        if not messages:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=task_instruction.format(
                        ticker=state["company_of_interest"],
                        trade_date=state["trade_date"],
                        # fund_subclass is only present for the income_alt
                        # analyst; default to "" so other analysts whose
                        # task_instruction lacks the placeholder are unaffected.
                        fund_subclass=state.get("fund_subclass", "") or "fund",
                    )
                ),
            ]

        # Tag this invocation so the UsageCallback (wired in trading_graph.py)
        # can bucket tokens / latency / tool counts by analyst. The tag also
        # propagates to any ToolNode the LLM triggers, so tool calls land in
        # the same bucket as the analyst that requested them.
        response = llm_with_tools.invoke(
            messages,
            config={"tags": [node_name], "metadata": {"analyst": node_name}},
        )

        if not state["messages"]:
            # Seed-message run: persist the system+human messages so subsequent
            # re-entries see the full conversation history.
            return {"messages": messages + [response], "sender": node_name}

        return {"messages": [response], "sender": node_name}

    return node_fn
