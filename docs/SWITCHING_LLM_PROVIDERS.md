# Switching LLM providers

This project is **provider-agnostic by design**: every agent receives a
pre-built LangChain `BaseChatModel` and never imports a provider directly.
Provider selection lives in ONE place — `agentic/llm_clients/factory.py` —
which already has working branches for `ollama`, `openai`, `anthropic`, and
`google`. Switching providers is **config + an API key + one package**, with
no changes to agent or graph code.

The default stack runs on **Ollama** (local or Ollama-cloud). Below is the
full walkthrough for moving any (or all) of the LLM work to a hosted API. The
**Claude (Anthropic) API** is used as the worked example; OpenAI and Google
follow the identical recipe (see the end).

---

## How model selection flows

```
default_config.py  ──►  create_llm_pair(config)  ──►  create_llm_client(provider, model, …)
   (llm_provider,          (builds the deep +            (the ONE place that
    *_think_llm,            quick tiers)                   imports ChatOllama /
    *_reasoning)                                           ChatAnthropic / …)
```

`create_llm_client()` switches on `provider` and returns the matching
`BaseChatModel`. Agents call `.invoke()` on it without knowing or caring which
provider is behind it.

There are **two LLM tiers**, both read from config:

| Tier | Config key | Used by |
|------|------------|---------|
| deep  | `deep_think_llm`  | Head Analyst, Researcher, Portfolio Manager (synthesis) |
| quick | `quick_think_llm` | All analyst nodes + Junior PA (fast path) |

You can point both tiers at the same hosted model, or mix (e.g. local quick +
hosted deep).

---

## Walkthrough — using the Claude (Anthropic) API

### 1. Install the Anthropic LangChain adapter

```powershell
.\venv\Scripts\python.exe -m pip install langchain-anthropic
```

Add `langchain-anthropic` to `requirements.txt` so it persists.

### 2. Put your API key in `.env`

`run.py` calls `load_dotenv()` at startup, so anything in `.env` is loaded into
the environment. Add:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`langchain-anthropic` reads `ANTHROPIC_API_KEY` from the environment
automatically — you do **not** pass it through config. (Keeping it out of the
config dict also keeps it out of the per-run transcript files.)

### 3. Point config at Anthropic + Claude models

**Option A — `.env` only (no code edits).** The keys below use the `agentic_`
override prefix wired in `default_config._ENV_OVERRIDES`; env vars win over the
defaults in `DEFAULT_CONFIG`:

```
agentic_LLM_PROVIDER=anthropic
agentic_DEEP_THINK_LLM=claude-opus-4-8
agentic_QUICK_THINK_LLM=claude-haiku-4-5-20251001
```

If your `.env` still has Ollama-cloud lines (`agentic_OLLAMA_BASE_URL`,
`OLLAMA_API_KEY`, Ollama model overrides), comment them out so they don't
shadow the above.

**Option B — make it the permanent default.** Edit `DEFAULT_CONFIG` in
`agentic/default_config.py`:

```python
"llm_provider":   "anthropic",
"deep_think_llm":  "claude-opus-4-8",
"quick_think_llm": "claude-haiku-4-5-20251001",
```

> Model IDs change over time — check Anthropic's current model list and use the
> latest you have access to. Pick a smaller/faster model for the quick tier and
> a stronger one for the deep tier if you want the tiers to differ.

### 4. Run as usual

```powershell
.\venv\Scripts\python.exe run.py --ticker NVDA --date 2024-01-15 --window 7
```

Nothing else changes — the four analysts, the portfolio roles, the per-agent
usage transcript, and the tool loops all work identically.

---

## What the factory handles for you

- **Ollama-only kwargs are stripped automatically.** `create_llm_pair()`
  forwards `deep_num_predict` / `deep_num_ctx` (and the Ollama branch injects a
  `client_kwargs` Bearer header for Ollama-cloud). `create_llm_client()` drops
  `num_ctx` / `num_predict` / `client_kwargs` for any non-Ollama provider, so
  you can leave those config keys in place — they're simply ignored for a
  hosted API. `ollama_base_url` is likewise ignored.
- **The `reasoning` flag is Ollama-specific.** `deep_reasoning` /
  `quick_reasoning` map to Ollama's `think` option and are a silent no-op for
  other providers. The quick/deep tiers still differ — they're just two
  separate client instances with whatever models you assigned. If you want a
  reasoning/thinking budget on a hosted model, pass it via a per-tier `**kwargs`
  extension in that provider's branch of the factory (not wired by default).

---

## Things to know before you switch

- **Tool-calling reliability usually improves.** Small local models have been a
  pain point for emitting `tool_calls` (see `prompts/llm_selection_prompt.md`).
  Most hosted models emit native tool calls, so the analyst tool loops become
  more reliable — this is a behaviour change, not just a swap.
- **Cost.** Local Ollama is free; hosted APIs are metered per token. The
  per-agent usage transcript written to `outputs/transcripts/` after every run
  gives you exact token counts per agent — use it to size spend before running
  a large sweep.
- **Offline requirement.** Going fully hosted removes the offline/local
  guarantee. A hybrid (local quick tier + hosted deep tier) keeps most calls
  local while upgrading only the synthesis roles.

---

## Other providers (OpenAI, Google)

Same recipe, different package + key + provider name:

| Provider | `pip install` | API key env var | `agentic_LLM_PROVIDER` |
|----------|---------------|-----------------|------------------------|
| Anthropic | `langchain-anthropic` | `ANTHROPIC_API_KEY` | `anthropic` |
| OpenAI    | `langchain-openai`    | `OPENAI_API_KEY`    | `openai`    |
| Google    | `langchain-google-genai` | `GOOGLE_API_KEY` | `google`   |

The `openai` branch also accepts a `base_url`, so it works against any
OpenAI-compatible endpoint (vLLM, LM Studio, a local gateway, etc.).

### Adding a brand-new provider

Add an `if provider == "...":` branch in `create_llm_client()` following the
existing pattern: lazy-import the LangChain adapter, build the client, return
it. Everything downstream consumes `BaseChatModel` generically, so no other
file needs to change.
