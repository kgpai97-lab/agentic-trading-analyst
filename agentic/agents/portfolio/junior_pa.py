"""Junior Portfolio Analyst — assesses portfolio composition and selects the
tickers the Portfolio Manager should focus on while rebalancing.

Runs AFTER the analyst layer has covered every holding, so it reads each
ticker's four-section briefing plus the raw tool output (ETF holdings, sector
weights) and judges composition: cross-holding overlap (look-through from fund
top holdings), single-name / sector concentration, and each asset's fit to the
stated OBJECTIVE.

The LLM judges a raw 0-100 FOCUS score per holding (HIGH = needs more PM
attention this rebalance: poor objective fit, high overlap, concentration); the
priority blend, normalization, and cumulative-threshold cutoff are computed here
in code (small models do not reliably execute the arithmetic in one shot).

LLM output (JSON, see prompts/junior_pa_system.txt):
  {
    "focus_scores":       {ticker: int 0-100},   # HIGH = needs PM focus
    "priority_rationale": str                      # composition assessment
  }

Node output (PortfolioState keys — unchanged contract):
  {
    "priority_scores":    {ticker: float, sums to 1.0 over non-cash holdings},
    "priority_rationale": str,
    "selected_tickers":   [str, ...]   # main_ticker first (if any), then by
                                       # focus priority desc until threshold
  }
"""

from __future__ import annotations

import re
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic.agents.portfolio._common import extract_json_block, load_prompt


_WEIGHT_BLEND = 0.3   # weight component of the priority blend
_FOCUS_BLEND = 0.7    # composition-judgment component (overlap/concentration/fit)


def _format_portfolio(weights: dict[str, float]) -> str:
    rows = [f"  {t:<8s} {w:.4f}" for t, w in weights.items()]
    return "\n".join(rows) if rows else "  (empty)"


# Only these analysts' tool output is composition-relevant. The fund analysts
# (composition / fixed_income_alt) emit holdings + sector weights + asset mix —
# exactly the look-through data the Junior PA needs. The fundamentals analyst
# (direct equities) emits valuation/growth noise; only its Sector / Industry
# line matters for composition. market / news / sentiment are pure timing and
# are dropped — the Junior PA prompt explicitly ignores price action and news.
_COMPOSITION_LABELS = {"composition", "fixed_income_alt"}
_FUNDAMENTALS_LABEL = "fundamentals"

# Sections in the flattened tool-output block are delimited by this header
# (see PortfolioGraph._format_tool_outputs): "--- <analyst> tools ---".
_TOOL_SECTION_RE = re.compile(r"^--- (\w+) tools ---$", re.MULTILINE)


def _split_tool_sections(tool_out: str) -> dict[str, str]:
    """Split a flattened tool-output block into {analyst_label: body}."""
    sections: dict[str, str] = {}
    matches = list(_TOOL_SECTION_RE.finditer(tool_out))
    for i, m in enumerate(matches):
        label = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(tool_out)
        sections[label] = tool_out[start:end].strip()
    return sections


def _composition_slice(tool_out: str) -> str:
    """Keep only the composition-relevant slice of a ticker's tool output:
    fund holdings/sectors verbatim, plus the Sector/Industry line from a direct
    equity's fundamentals. Drop market / news / sentiment entirely.
    """
    sections = _split_tool_sections(tool_out)
    parts: list[str] = []
    for label, body in sections.items():
        if label in _COMPOSITION_LABELS:
            parts.append(body)
        elif label == _FUNDAMENTALS_LABEL:
            # Direct equity: only its sector matters for the look-through tilt.
            sector_lines = [
                ln.strip() for ln in body.splitlines()
                if "Sector / Industry" in ln or "Sector/Industry" in ln
            ]
            if sector_lines:
                parts.append("\n".join(sector_lines))
    return "\n\n".join(p for p in parts if p.strip())


def _format_composition_inputs(per_ticker_reports: dict[str, dict]) -> str:
    """Render each holding's COMPOSITION data only (fund holdings + sector
    weights + asset mix; direct-equity sector) so the Junior PA can judge
    overlap / concentration / fit without the timing-and-news noise.
    """
    if not per_ticker_reports:
        return "  (no analyst coverage available)"
    blocks: list[str] = []
    for ticker, payload in per_ticker_reports.items():
        tool_out = (payload.get("tool_outputs") or "").strip()
        comp = _composition_slice(tool_out) if tool_out else ""
        header = f"===== {ticker} ====="
        if comp:
            blocks.append(f"{header}\n{comp}")
        else:
            blocks.append(
                f"{header}\n  (no composition/sector data available for {ticker})"
            )
    return "\n\n".join(blocks)


def _compute_priority(
    weights: dict[str, float],
    focus: dict[str, float],
) -> dict[str, float]:
    """Blend normalized weight and normalized focus score into priority scores
    that sum to 1.0 over the scored (non-cash) holdings.

        priority = 0.3 * normalized_weight + 0.7 * normalized_focus

    The focus score is the Junior PA's judgment of how much a holding needs PM
    attention this rebalance (overlap / concentration / objective fit). Weight
    is folded in so a large mis-fit position outranks a tiny one at the same
    focus level. `weights` is restricted to the non-cash holdings that were
    scored; empty or all-zero components degrade gracefully to the other.
    """
    tickers = list(focus.keys())
    if not tickers:
        return {}

    weight_sum = sum(weights.get(t, 0.0) for t in tickers)
    focus_sum = sum(max(focus.get(t, 0.0), 0.0) for t in tickers)

    priority: dict[str, float] = {}
    for t in tickers:
        nw = (weights.get(t, 0.0) / weight_sum) if weight_sum > 0 else 0.0
        nf = (max(focus.get(t, 0.0), 0.0) / focus_sum) if focus_sum > 0 else 0.0
        priority[t] = round(_WEIGHT_BLEND * nw + _FOCUS_BLEND * nf, 6)
    return priority


def _select_by_threshold(
    priority: dict[str, float],
    threshold: float,
) -> list[str]:
    """Sort by priority desc, accumulate until the running sum >= threshold.

    Priority sums to ~1.0, so `threshold` is a direct fraction of total
    priority mass. Always selects at least one ticker when any exist.
    """
    ordered = sorted(priority.items(), key=lambda kv: kv[1], reverse=True)
    selected: list[str] = []
    running = 0.0
    for ticker, score in ordered:
        selected.append(ticker)
        running += score
        if running >= threshold:
            break
    return selected


def create_junior_pa(deep_llm: BaseChatModel) -> Callable[[dict], dict]:
    """Return a node that mutates PortfolioState in place."""
    system_text = load_prompt("junior_pa_system.txt")

    def node(state: dict) -> dict:
        objective = state["objective"]
        weights   = state["current_weights"]
        main      = state["main_ticker"]
        threshold = state["priority_threshold"]

        per_ticker = state.get("per_ticker_reports", {})

        main_line = (
            f"MAIN TICKER (candidate buy/add): {main}\n"
            if main
            else "MAIN TICKER: none — this is a pure rebalance of the existing holdings.\n"
        )
        human_text = (
            f"OBJECTIVE:\n{objective}\n\n"
            f"CURRENT PORTFOLIO (ticker  weight):\n{_format_portfolio(weights)}\n\n"
            f"{main_line}\n"
            "FUND COMPOSITION — each holding's composition data follows: for "
            "funds, the top holdings, sector weights, and asset mix; for direct "
            "equities, the holding's sector. Use it to judge cross-holding "
            "overlap (look-through), single-name/sector concentration, and fit "
            "to the OBJECTIVE:\n\n"
            f"{_format_composition_inputs(per_ticker)}\n\n"
            "Now assess composition and assign a raw 0-100 FOCUS score to every "
            "non-cash holding (HIGH = needs more PM attention this rebalance), "
            "then return the JSON object."
        )

        resp = deep_llm.invoke(
            [
                SystemMessage(content=system_text),
                HumanMessage(content=human_text),
            ],
            config={"tags": ["junior_pa"]},
        )
        parsed = extract_json_block(resp.content)

        # Keep only real non-cash holdings; coerce focus scores to floats.
        non_cash = {t for t in weights if t.upper() != "CASH"}
        raw_focus = parsed.get("focus_scores", {})
        focus: dict[str, float] = {}
        for t in non_cash:
            v = raw_focus.get(t)
            if isinstance(v, (int, float)):
                focus[t] = float(v)
            else:
                focus[t] = 0.0  # unscored holding → no focus signal

        # Code owns the math: blend → normalized priority (sums to ~1.0).
        scores = _compute_priority(weights, focus)

        # Code owns selection: cumulative-threshold cutoff over priority mass.
        selected = _select_by_threshold(scores, threshold)

        # Main ticker (candidate add) is always analysed and listed first.
        if main:
            if main in selected:
                selected.remove(main)
            selected.insert(0, main)

        return {
            "priority_scores":    scores,
            "priority_rationale": parsed.get("priority_rationale", ""),
            "selected_tickers":   selected,
        }

    return node
