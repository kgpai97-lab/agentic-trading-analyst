"""Pytest fixtures — fake chat models for graph-shape tests.

These tests do not call Ollama or any real LLM. They drive the StateGraph
with scripted message sequences so we can verify routing, tool loops, and
message clearing in isolation from model latency / non-determinism.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """A BaseChatModel that returns a pre-scripted sequence of AIMessages.

    Each .invoke() / ._generate() call consumes one AIMessage from the script.
    bind_tools() is a no-op (returns self) so analyst nodes work unchanged.
    """

    # Pydantic v2 — keep this lenient so we can stash a list of messages.
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    def __init__(self, responses: Sequence[AIMessage], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Iterator over the scripted responses. _generate() advances it.
        self._responses_iter: Iterator[AIMessage] = iter(list(responses))

    @property
    def _llm_type(self) -> str:  # required abstract
        return "scripted-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            msg = next(self._responses_iter)
        except StopIteration as e:  # noqa: F841
            raise RuntimeError(
                "ScriptedChatModel ran out of responses — your test asked the "
                "fake LLM to speak more times than the script provided."
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        # Analyst nodes call this; we ignore the tool list and keep replaying
        # the script. The scripted AIMessages already carry tool_calls when
        # the test wants a tool loop.
        return self


@pytest.fixture
def scripted_llm():
    """Factory fixture: scripted_llm([AIMessage(...), AIMessage(...)])."""

    def _make(responses: Sequence[AIMessage]) -> ScriptedChatModel:
        return ScriptedChatModel(responses=responses)

    return _make
