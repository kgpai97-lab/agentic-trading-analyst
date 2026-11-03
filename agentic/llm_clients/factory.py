"""LLM client factory — the single place that knows about providers.

Agentic concept: agents should be provider-agnostic. Each agent node receives
an already-constructed LangChain chat-model object (a BaseChatModel). It calls
.invoke() or .ainvoke() on that object without knowing whether the underlying
model is Ollama, OpenAI, or Anthropic. This lets us swap providers by changing
one config value, not by touching agent code.

The no-raw-imports rule: agent files must NEVER import
ChatOllama, ChatOpenAI, etc. directly. All provider imports live here.

Supported providers:
    "ollama"    — local Ollama server (default; free, no API key needed)
    "openai"    — OpenAI or any OpenAI-compatible endpoint
    "anthropic" — Anthropic Claude
    "google"    — Google Gemini
"""

from __future__ import annotations

import os

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel


def create_llm_client(
    provider: str,
    model: str,
    base_url: str | None = None,
    temperature: float = 0.2,
    callbacks: list[BaseCallbackHandler] | None = None,
    reasoning: bool | None = None,
    **kwargs,
) -> BaseChatModel:
    """Instantiate and return a LangChain chat model for the given provider.

    Args:
        provider:    One of "ollama", "openai", "anthropic", "google".
        model:       Model name as the provider expects it
                     (e.g. "qwen3.5:cloud" for Ollama, "gpt-4o" for OpenAI).
        base_url:    Override the API endpoint. Required for Ollama
                     ("http://localhost:11434"); ignored for cloud providers
                     unless you're hitting a proxy.
        temperature: Sampling temperature. 0.0 = deterministic, 1.0 = creative.
                     Analysts use ~0.2 to stay factual.
        reasoning:   Toggle the model's "thinking" mode (Ollama only — maps to
                     ChatOllama's `reasoning`, i.e. the server's `think` option).
                     True = think before answering (synthesis roles);
                     False = no thinking (fast analyst nodes);
                     None = leave the model's default untouched. Ignored by the
                     cloud providers, which expose thinking through their own
                     params.
        **kwargs:    Passed through to the underlying chat-model constructor
                     (e.g. format="json" for Ollama structured output).

    Returns:
        A LangChain BaseChatModel. Every agent calls .invoke() on this object.

    Raises:
        ValueError:  Unknown provider string.
        ImportError: Required provider package not installed
                     (e.g. langchain-openai missing when provider="openai").
    """
    provider = provider.lower().strip()

    # Ollama-only generation params. create_llm_pair() forwards num_predict /
    # num_ctx (the deep-tier overrides) through **kwargs; those are meaningful
    # to ChatOllama but would crash ChatAnthropic / ChatOpenAI / Gemini, which
    # don't accept them. Strip them for every non-Ollama provider so callers
    # can switch providers without also having to clear the deep_num_* config.
    if provider != "ollama":
        for _ollama_only in ("num_ctx", "num_predict", "client_kwargs"):
            kwargs.pop(_ollama_only, None)

    if provider == "ollama":
        # langchain-ollama talks to the Ollama HTTP server (not the CLI).
        # The server must already be running at base_url before Python starts.
        from langchain_ollama import ChatOllama  # noqa: PLC0415

        # Ollama defaults (num_ctx=2048, num_predict=128) are way too tight for
        # tool loops where the model has to ingest tool results and emit a
        # 150-250 word report afterwards. Bumping both. Caller-supplied kwargs
        # win if they want to override.
        # gpt-oss is a reasoning-first model: it ALWAYS emits an analysis chain
        # before its final answer. num_predict=1024 frequently truncates inside
        # that chain, returning EMPTY content (the thinking tokens are dropped
        # when reasoning is off, and never reach the final channel when on).
        # 4096 gives reliable headroom; the deep tier overrides higher still.
        ollama_defaults = {
            "num_ctx":     8192,
            "num_predict": 4096,
        }
        # reasoning maps to ChatOllama's `reasoning` field (the server `think`
        # option). Only forward it when the caller has an opinion; None means
        # "leave the model default alone".
        if reasoning is not None:
            ollama_defaults["reasoning"] = reasoning

        # Cloud models (e.g. qwen3.5:cloud, served from https://ollama.com)
        # authenticate via a Bearer token. Read it from the environment so the
        # secret never lands in the config dict / transcripts. Only inject when
        # present, so local-server usage is unaffected. A caller-supplied
        # client_kwargs in **kwargs wins.
        api_key = os.getenv("OLLAMA_API_KEY")
        if api_key and "client_kwargs" not in kwargs:
            ollama_defaults["client_kwargs"] = {
                "headers": {"Authorization": f"Bearer {api_key}"}
            }
        ollama_defaults.update(kwargs)

        return ChatOllama(
            model=model,
            base_url=base_url or "http://localhost:11434",
            temperature=temperature,
            callbacks=callbacks,
            **ollama_defaults,
        )

    if provider == "openai":
        # Requires: pip install langchain-openai; OPENAI_API_KEY in env.
        # base_url can point to any OpenAI-compatible server (e.g. vLLM, LM Studio).
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        init_kwargs = {"model": model, "temperature": temperature, **kwargs}
        if base_url:
            init_kwargs["base_url"] = base_url
        if callbacks is not None:
            init_kwargs["callbacks"] = callbacks
        return ChatOpenAI(**init_kwargs)

    if provider == "anthropic":
        # Requires: pip install langchain-anthropic; ANTHROPIC_API_KEY in env
        # (langchain-anthropic reads it automatically — never pass it in config).
        # Ollama-only num_ctx/num_predict/client_kwargs were already stripped
        # above, so a config carrying deep_num_* is safe here. See the
        # "Walkthrough — swapping Ollama for the Claude API" section in docs/SWITCHING_LLM_PROVIDERS.md.
        from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

        return ChatAnthropic(
            model=model, temperature=temperature, callbacks=callbacks, **kwargs
        )

    if provider == "google":
        # Requires: pip install langchain-google-genai; GOOGLE_API_KEY in env.
        from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

        return ChatGoogleGenerativeAI(
            model=model, temperature=temperature, callbacks=callbacks, **kwargs
        )

    raise ValueError(
        f"Unknown provider {provider!r}. "
        "Supported: 'ollama', 'openai', 'anthropic', 'google'."
    )


def create_llm_pair(
    config: dict,
    callbacks: list[BaseCallbackHandler] | None = None,
    deep_num_predict: int | None = None,
    deep_num_ctx: int | None = None,
) -> tuple[BaseChatModel, BaseChatModel]:
    """Create the (deep_think_llm, quick_think_llm) pair from a config dict.

    This is the standard way the graph creates both LLM tiers at startup.
    deep_think_llm  → high-stakes synthesis nodes (Head Analyst, Researcher,
                      Portfolio Manager) — thinking ON by default
    quick_think_llm → all analyst nodes + Junior PA — thinking OFF by default

    Both tiers point at the same model (qwen3.5:cloud) and differ only by the
    `reasoning` flag: the two tiers are now "thinking vs. not", not "big vs.
    small". `quick_reasoning` / `deep_reasoning` in config drive each tier;
    leave them None to use the model's own default.

    Args:
        deep_num_predict: Override num_predict for the deep-tier Ollama client.
                          Thinking models need >1024 to clear their reasoning
                          chain before emitting an answer.
        deep_num_ctx:     Override num_ctx for the deep-tier Ollama client.
                          Must exceed deep_num_predict + input length; Ollama
                          silently caps num_predict at (num_ctx - input_tokens).

    Usage:
        from agentic.default_config import get_config
        from agentic.llm_clients.factory import create_llm_pair

        deep_llm, quick_llm = create_llm_pair(get_config())
    """
    provider = config["llm_provider"]
    base_url = config.get("ollama_base_url")

    deep_kwargs: dict = {}
    if deep_num_predict is not None:
        deep_kwargs["num_predict"] = deep_num_predict
    if deep_num_ctx is not None:
        deep_kwargs["num_ctx"] = deep_num_ctx

    deep_llm = create_llm_client(
        provider, config["deep_think_llm"], base_url=base_url,
        callbacks=callbacks, reasoning=config.get("deep_reasoning"), **deep_kwargs
    )
    quick_llm = create_llm_client(
        provider, config["quick_think_llm"], base_url=base_url,
        callbacks=callbacks, reasoning=config.get("quick_reasoning"),
    )
    return deep_llm, quick_llm
