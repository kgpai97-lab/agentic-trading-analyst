"""Sentiment Analyst — graph node factory.

See agentic/agents/analysts/_common.py for the shared tool-loop pattern. This
file is intentionally a three-liner: pick the prompt, pick the tools, name
the node.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agentic.agents.analysts._common import load_prompt, make_analyst_node
from agentic.agents.utils.agent_states import AgentState


def create_sentiment_analyst(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    from agentic.agents.utils.agent_utils import TOOLS_SENTIMENT  # noqa: PLC0415

    return make_analyst_node(
        llm=llm,
        tools=TOOLS_SENTIMENT,
        system_prompt=load_prompt("sentiment_analyst_system.txt"),
        task_instruction=(
            "Analyse sentiment for **{ticker}** as of {trade_date}. "
            "Start by calling get_news, then write your report."
        ),
        node_name="sentiment_analyst",
    )
