"""Fundamentals Analyst — graph node factory.

Tools: get_fundamentals, get_income_statement. Writes to state["fundamentals_report"].
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agentic.agents.analysts._common import load_prompt, make_analyst_node
from agentic.agents.utils.agent_states import AgentState


def create_fundamentals_analyst(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    from agentic.agents.utils.agent_utils import TOOLS_FUNDAMENTALS  # noqa: PLC0415

    return make_analyst_node(
        llm=llm,
        tools=TOOLS_FUNDAMENTALS,
        system_prompt=load_prompt("fundamentals_analyst_system.txt"),
        task_instruction=(
            "Analyse the fundamentals of **{ticker}** as of {trade_date}. "
            "Call get_fundamentals and get_income_statement each once, then "
            "write your report."
        ),
        node_name="fundamentals_analyst",
    )
