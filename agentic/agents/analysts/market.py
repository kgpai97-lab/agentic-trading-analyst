"""Market Analyst — graph node factory.

Tools: get_stock_data, get_indicators. Writes to state["market_report"].
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agentic.agents.analysts._common import load_prompt, make_analyst_node
from agentic.agents.utils.agent_states import AgentState


def create_market_analyst(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    from agentic.agents.utils.agent_utils import TOOLS_MARKET  # noqa: PLC0415

    return make_analyst_node(
        llm=llm,
        tools=TOOLS_MARKET,
        system_prompt=load_prompt("market_analyst_system.txt"),
        task_instruction=(
            "Analyse the recent price action and technical posture of "
            "**{ticker}** as of {trade_date}. Call get_stock_data and "
            "get_indicators each once, then write your report."
        ),
        node_name="market_analyst",
    )
