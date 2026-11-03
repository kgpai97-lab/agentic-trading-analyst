"""Fixed-Income / Alternatives Analyst — graph node factory.

Fourth pipeline slot when asset_type == "income_alt" (a bond_fund or
alternative_fund holding). Tool: get_etf_holdings (same as composition).
Writes to state["fundamentals_report"] (same slot as fundamentals /
composition, so downstream nodes don't branch).

One node serves both subclasses. The {fund_subclass} placeholder in the task
instruction is filled from state["fund_subclass"] ("bond_fund" |
"alternative_fund") by make_analyst_node, so the LLM knows which lens to use.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agentic.agents.analysts._common import load_prompt, make_analyst_node
from agentic.agents.utils.agent_states import AgentState


def create_fixed_income_alt_analyst(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    from agentic.agents.utils.agent_utils import TOOLS_INCOME_ALT  # noqa: PLC0415

    return make_analyst_node(
        llm=llm,
        tools=TOOLS_INCOME_ALT,
        system_prompt=load_prompt("fixed_income_alt_analyst_system.txt"),
        task_instruction=(
            "Analyse the {fund_subclass} **{ticker}** as of {trade_date}. "
            "Call get_etf_holdings once, then write your report."
        ),
        node_name="fixed_income_alt_analyst",
    )
