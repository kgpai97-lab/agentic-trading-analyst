"""GraphSetup — builds and compiles the LangGraph StateGraph.

Step 5 graph — all four analysts running sequentially, each with its own
tool loop:

    sentiment_analyst ─tool?─► tools_sentiment ─┐
            │                                    │ loop back
            │ (plain text)                       │
            ▼                       ◄────────────┘
    clear_sentiment ─► market_analyst ─tool?─► tools_market ─┐
                              │                              │ loop back
                              │ (plain text)                 │
                              ▼                ◄─────────────┘
                       clear_market ─► news_analyst ─tool?─► tools_news ─┐
                                              │                          │ loop back
                                              │ (plain text)             │
                                              ▼            ◄─────────────┘
                                       clear_news ─► fundamentals_analyst ─tool?─► tools_fundamentals ─┐
                                                              │                                        │ loop back
                                                              │ (plain text)                           │
                                                              ▼                       ◄────────────────┘
                                                       clear_fundamentals ─► END

Each analyst:
  • binds its TOOLS_<NAME> list and emits tool_calls when it wants data
  • the paired ToolNode executes the tool and appends a ToolMessage
  • the paired clear_<name> node copies the final AIMessage into its report
    field and wipes state["messages"] so the next analyst starts fresh.
"""

from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import RemoveMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agentic.agents.analysts.composition import create_composition_analyst
from agentic.agents.analysts.fixed_income_alt import create_fixed_income_alt_analyst
from agentic.agents.analysts.fundamentals import create_fundamentals_analyst
from agentic.agents.analysts.market import create_market_analyst
from agentic.agents.analysts.news import create_news_analyst
from agentic.agents.analysts.sentiment import create_sentiment_analyst
from agentic.agents.utils.agent_states import AgentState
from agentic.agents.utils.agent_utils import (
    TOOLS_COMPOSITION,
    TOOLS_FUNDAMENTALS,
    TOOLS_INCOME_ALT,
    TOOLS_MARKET,
    TOOLS_NEWS,
    TOOLS_SENTIMENT,
)
from agentic.graph.conditional_logic import ConditionalLogic


# News tool outputs render one numbered line per article (plus an indented
# summary line). They're the bulkiest tool result — up to 20 articles × 2 tools.
# When we forward raw tool output downstream (Step 6), cap the article count so
# the payload stays bounded; numeric/holdings outputs are passed verbatim.
_MAX_NEWS_ARTICLES = 8
_NEWS_HEADER_RE = re.compile(r"news for|global / macro|additional global", re.IGNORECASE)
_NEWS_ITEM_RE = re.compile(r"^\s*(\d+)\.\s")


def _truncate_news_output(text: str) -> str:
    """If `text` is a news-tool dump, keep only the first _MAX_NEWS_ARTICLES
    articles (headline + summary lines); otherwise return it unchanged.

    Detection is intentionally loose: a news dump has a "news for …" header and
    numbered article lines. Non-news tool outputs (OHLCV tables, indicator
    summaries, ETF holdings) have neither and pass through verbatim.
    """
    first_line = text.split("\n", 1)[0]
    if not _NEWS_HEADER_RE.search(first_line):
        return text

    out_lines: list[str] = []
    article_count = 0
    for line in text.split("\n"):
        if _NEWS_ITEM_RE.match(line):
            article_count += 1
            if article_count > _MAX_NEWS_ARTICLES:
                out_lines.append(
                    f"   … (+ remaining articles truncated; showing first "
                    f"{_MAX_NEWS_ARTICLES})"
                )
                break
        out_lines.append(line)
    return "\n".join(out_lines)


def _make_msg_clear_node(report_field: str, analyst_name: str, extract_llm=None):
    """Node that copies the last AI message into a report field, captures the
    raw tool output, runs structured signal extraction, and wipes the buffer.

    Important — MessagesState uses an `add_messages` reducer that *appends*
    rather than replaces. Returning {"messages": []} is a no-op (append-empty
    does nothing). To actually delete prior messages we emit a RemoveMessage
    for each one; the reducer handles the deletion.

    The raw ToolMessage contents are captured into `tool_outputs[analyst_name]`
    BEFORE the wipe so the Step-6 portfolio layer can forward the underlying
    data (not just the analyst's prose) downstream. News dumps are truncated to
    keep the payload bounded (see _truncate_news_output).

    Phase 1b (Step 7): if `extract_llm` is supplied, this node also runs Call 2
    of the two-call extraction — the finalized prose report is distilled into a
    validated AnalystSignal written to `signals[analyst_name]`. The clear node
    is the right home for this because the prose is final here (the tool loop
    has converged) and it runs exactly once per analyst. Extraction is
    best-effort (never raises — see extract_signal), so a flaky extraction
    can't break the analyst's main report output.
    """
    def node_fn(state: AgentState) -> dict:
        messages = state["messages"]
        last_message = messages[-1]

        tool_chunks = [
            _truncate_news_output(str(m.content))
            for m in messages
            if isinstance(m, ToolMessage) and str(m.content).strip()
        ]
        captured = "\n\n".join(tool_chunks)

        removals = [RemoveMessage(id=m.id) for m in messages]
        update: dict = {
            "messages":   removals,
            report_field: last_message.content,
        }
        if captured:
            update["tool_outputs"] = {analyst_name: captured}

        if extract_llm is not None:
            # Lazy import: avoids a circular import at module load
            # (_common imports agent_states which is imported widely).
            from agentic.agents.analysts._common import extract_signal  # noqa: PLC0415
            signal = extract_signal(
                extract_llm,
                prose=str(last_message.content),
                ticker=state["company_of_interest"],
                trade_date=state["trade_date"],
                node_name=analyst_name,
            )
            update["signals"] = {analyst_name: signal}

        return update
    return node_fn


class GraphSetup:
    """Builds the LangGraph StateGraph for the current build step.

    Step 3: sentiment node + clear node, unconditional edge
    Step 4: adds ToolNode + conditional edge (tool loop)
    Step 5: adds market, news, fundamentals analysts (sequential)   ← current
    Step 6: adds debate loops, managers, trader
    """

    def __init__(
        self,
        quick_llm: BaseChatModel,
        deep_llm: BaseChatModel,
        config: dict,
    ) -> None:
        self.quick_llm = quick_llm
        self.deep_llm  = deep_llm
        self.config    = config

    def build(self):
        graph = StateGraph(AgentState)

        # ── Analyst pipeline definition ─────────────────────────────────────
        # Each tuple: (analyst_name, factory, tools_list, report_field).
        # Order here defines run order in the graph.
        #
        # The fourth slot is asset-type-dependent:
        #   stock      → Fundamentals Analyst (P/E, EPS, margins, income statement)
        #   etf        → Composition Analyst  (equity-fund holdings, sector tilt)
        #   income_alt → Fixed-Income/Alternatives Analyst (bond & non-equity funds)
        # All write to the same `fundamentals_report` field so that downstream
        # Step-6 nodes can read "the fourth analyst's view" without branching.
        asset_type = self.config.get("asset_type")
        if asset_type == "etf":
            fourth = ("composition", create_composition_analyst, TOOLS_COMPOSITION, "fundamentals_report")
        elif asset_type == "income_alt":
            fourth = ("fixed_income_alt", create_fixed_income_alt_analyst, TOOLS_INCOME_ALT, "fundamentals_report")
        else:
            fourth = ("fundamentals", create_fundamentals_analyst, TOOLS_FUNDAMENTALS, "fundamentals_report")

        analysts = [
            ("sentiment", create_sentiment_analyst, TOOLS_SENTIMENT, "sentiment_report"),
            ("market",    create_market_analyst,    TOOLS_MARKET,    "market_report"),
            ("news",      create_news_analyst,      TOOLS_NEWS,      "news_report"),
            fourth,
        ]

        # ── Analyst subset (Step 6 cost control) ────────────────────────────
        # The portfolio layer runs the analyst graph in two passes: the
        # composition/fundamentals (fourth-slot) analyst for ALL holdings, then
        # market/news/sentiment only for the focus list. `analyst_subset`
        # selects which analysts this build wires:
        #   "composition" → only the fourth-slot analyst
        #   "mns"         → sentiment + market + news (no fourth slot)
        #   "all"/absent  → all four (standalone single-ticker TradingGraph)
        subset = self.config.get("analyst_subset", "all")
        if subset == "composition":
            analysts = [fourth]
        elif subset == "mns":
            analysts = [a for a in analysts if a[0] in ("sentiment", "market", "news")]
        elif subset not in ("all", None):
            raise ValueError(
                f"unknown analyst_subset {subset!r}; expected "
                "'composition', 'mns', 'all', or None"
            )

        routers = {
            "sentiment":        ConditionalLogic.should_continue_sentiment,
            "market":           ConditionalLogic.should_continue_market,
            "news":             ConditionalLogic.should_continue_news,
            "fundamentals":     ConditionalLogic.should_continue_fundamentals,
            "composition":      ConditionalLogic.should_continue_composition,
            "fixed_income_alt": ConditionalLogic.should_continue_fixed_income_alt,
        }

        # ── Add nodes ────────────────────────────────────────────────────────
        # Phase 1b: the clear node runs the cheap structured-extraction second
        # call on the quick tier when `extract_signals` is enabled (default).
        # Set extract_signals=False to skip it (e.g. fast smoke runs).
        extract_llm = self.quick_llm if self.config.get("extract_signals", True) else None
        for name, factory, tools, report_field in analysts:
            graph.add_node(f"{name}_analyst",  factory(self.quick_llm))
            graph.add_node(f"tools_{name}",    ToolNode(list(tools)))
            graph.add_node(f"clear_{name}",    _make_msg_clear_node(report_field, name, extract_llm))

        # ── Entry point ──────────────────────────────────────────────────────
        graph.set_entry_point(f"{analysts[0][0]}_analyst")

        # ── Wire each analyst's tool loop + sequential hop ──────────────────
        for i, (name, _, _, _) in enumerate(analysts):
            graph.add_conditional_edges(
                f"{name}_analyst",
                routers[name],
                {
                    f"tools_{name}":  f"tools_{name}",
                    f"clear_{name}":  f"clear_{name}",
                },
            )
            # After tools run, loop back to the analyst so it sees the result.
            graph.add_edge(f"tools_{name}", f"{name}_analyst")

            # After the analyst writes its report, hand off to the next one
            # (or END if this is the last analyst in the pipeline).
            if i + 1 < len(analysts):
                next_name = analysts[i + 1][0]
                graph.add_edge(f"clear_{name}", f"{next_name}_analyst")
            else:
                graph.add_edge(f"clear_{name}", END)

        return graph.compile()
