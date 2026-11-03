"""Researcher — writes a combined bull + bear report from a Head Analyst
assessment. Tool-less; single llm.invoke().
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic.agents.portfolio._common import load_prompt


def create_researcher(deep_llm: BaseChatModel) -> Callable[[str, str, str], str]:
    """Return a callable: (ticker, head_assessment_md, raw_tool_output)
    -> bull_bear_report_md.

    The raw tool output is forwarded so catalyst claims can be sized against the
    underlying data (e.g. a holding's actual weight in a fund) rather than
    inflated from prose alone.
    """
    system_text = load_prompt("researcher_system.txt")

    def run(ticker: str, head_assessment: str, tool_output: str = "") -> str:
        raw_block = (
            f"RAW TOOL OUTPUT (ground truth for sizing claims):\n\n{tool_output}\n\n"
            if str(tool_output).strip()
            else ""
        )
        human_text = (
            f"TICKER: {ticker}\n\n"
            f"HEAD ANALYST ASSESSMENT:\n\n{head_assessment}\n\n"
            f"{raw_block}"
            "Now write the bull + bear research note in the structure specified."
        )
        resp = deep_llm.invoke(
            [
                SystemMessage(content=system_text),
                HumanMessage(content=human_text),
            ],
            config={"tags": [f"researcher:{ticker}"]},
        )
        return resp.content.strip()

    return run
