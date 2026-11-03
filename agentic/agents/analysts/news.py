"""News Analyst — graph node factory.

Tools: get_news, get_global_news. Writes to state["news_report"].
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agentic.agents.analysts._common import load_prompt, make_analyst_node
from agentic.agents.utils.agent_states import AgentState


def create_news_analyst(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    from agentic.agents.utils.agent_utils import TOOLS_NEWS  # noqa: PLC0415

    return make_analyst_node(
        llm=llm,
        tools=TOOLS_NEWS,
        system_prompt=load_prompt("news_analyst_system.txt"),
        task_instruction=(
            "Analyse the news flow for **{ticker}** as of {trade_date}. "
            "Call get_news and get_global_news each once, then write your report."
        ),
        node_name="news_analyst",
    )
