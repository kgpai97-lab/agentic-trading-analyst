"""Composition Analyst — graph node factory.

Replaces the Fundamentals Analyst in the fourth pipeline slot when
asset_type == "etf". Tool: get_etf_holdings. Writes to state["fundamentals_report"]
(same slot as fundamentals, so downstream nodes don't need to branch).
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agentic.agents.analysts._common import load_prompt, make_analyst_node
from agentic.agents.utils.agent_states import AgentState


def create_composition_analyst(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    from agentic.agents.utils.agent_utils import TOOLS_COMPOSITION  # noqa: PLC0415

    return make_analyst_node(
        llm=llm,
        tools=TOOLS_COMPOSITION,
        system_prompt=load_prompt("composition_analyst_system.txt"),
        task_instruction=(
            "Analyse the composition of the ETF **{ticker}** as of {trade_date}. "
            "Call get_etf_holdings once, then write your report."
        ),
        node_name="composition_analyst",
    )
