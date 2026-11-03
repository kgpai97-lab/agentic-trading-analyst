"""Head Analyst — consolidates the four per-ticker analyst reports and
flags inconsistencies. Tool-less; single llm.invoke().
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic.agents.portfolio._common import load_prompt


def create_head_analyst(deep_llm: BaseChatModel) -> Callable[[str, str, str], str]:
    """Return a callable: (ticker, briefing_md, raw_tool_output) -> assessment_md.

    The raw tool output (OHLCV, indicators, news, ETF holdings) is forwarded
    alongside the analysts' prose so the Head can verify their claims and read
    fund holdings directly (e.g. for look-through / overlap reasoning).
    """
    system_text = load_prompt("head_analyst_system.txt")

    def run(ticker: str, briefing: str, tool_output: str = "") -> str:
        raw_block = (
            f"RAW TOOL OUTPUT (ground truth the analysts read):\n\n{tool_output}\n\n"
            if str(tool_output).strip()
            else ""
        )
        human_text = (
            f"TICKER: {ticker}\n\n"
            f"FOUR-ANALYST BRIEFING:\n\n{briefing}\n\n"
            f"{raw_block}"
            "Now write your consolidated assessment in the structure specified."
        )
        resp = deep_llm.invoke(
            [
                SystemMessage(content=system_text),
                HumanMessage(content=human_text),
            ],
            config={"tags": [f"head_analyst:{ticker}"]},
        )
        return resp.content.strip()

    return run
