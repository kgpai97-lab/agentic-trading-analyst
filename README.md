# Agentic Trading Analyst

A multi-agent AI system that fetches live market data, analyses it through a
set of specialised LLM-backed agents, and produces a structured investment
briefing. Built as a learning project for LangGraph / agentic patterns.

> **Research tool only — not financial advice.**

---

## What it does

Given a ticker and a date, four analyst agents (Market, Sentiment, News, and a
fourth slot that adapts to the asset type — Fundamentals for stocks,
Composition for ETFs, Fixed-Income/Alternatives for bond/alt funds) each run a
tool-calling loop over live market data and write a prose report. The reports
are composed into a single briefing.

A portfolio-aware layer wraps that per-ticker pipeline: a Junior PA prioritises
which holdings to analyse, a Head Analyst consolidates each ticker's four
reports, a Researcher writes a bull/bear note, and a Portfolio Manager emits a
target-weight rebalance.

---

## Stack

| Layer | Choice |
|-------|--------|
| LLM provider | Ollama (local or Ollama-cloud) by default; pluggable — see below |
| Orchestration | LangGraph `StateGraph` (stateful graph with tool loops) |
| Market data | yfinance (free, no API key) |
| Package | `agentic` |

The LLM provider is **pluggable**. The default is Ollama, but the system is
provider-agnostic and ships with working branches for OpenAI, Anthropic
(Claude), and Google. To switch, see
**[docs/SWITCHING_LLM_PROVIDERS.md](docs/SWITCHING_LLM_PROVIDERS.md)**.

---

## Setup

```powershell
# 1. Create a virtualenv and install deps
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Configure the LLM provider
#    Default: a running Ollama server (https://ollama.com or http://localhost:11434).
#    Copy .env.example to .env and fill in your model / endpoint / key.
copy .env.example .env

# 3. (Ollama local) pull a model, e.g.
#    ollama pull llama3.2:3b
```

Configuration lives in `agentic/default_config.py` (`DEFAULT_CONFIG`). Any flat
key can be overridden from `.env` with the `agentic_` prefix
(e.g. `agentic_LLM_PROVIDER=openai`).

---

## Usage

### Single-ticker briefing

```powershell
.\venv\Scripts\python.exe run.py --ticker NVDA --date 2024-01-15 --window 7
.\venv\Scripts\python.exe run.py --ticker XLK --asset-type etf --date 2024-01-15
```

Or from Python:

```python
from dotenv import load_dotenv; load_dotenv()
from agentic.graph.trading_graph import TradingGraph

ta = TradingGraph(config={"asset_type": "stock", "lookback_days": 7})
state, briefing = ta.propagate("NVDA", "2024-01-15")
print(briefing)
```

### Portfolio rebalance

```powershell
# Pure rebalance of an existing portfolio:
.\venv\Scripts\python.exe run.py --portfolio portfolio_test.json --date 2024-01-15

# Rebalance with a candidate "main" ticker:
.\venv\Scripts\python.exe run.py --ticker NVDA --portfolio portfolio_test.json --date 2024-01-15
```

`portfolio_test.json` is an example holdings file; each holding declares a
`type` (`equity` / `equity_fund` / `bond_fund` / `alternative_fund` / `cash`)
that routes it to the right fourth analyst.

---

## Observability

After every run, a per-agent transcript is written to `outputs/transcripts/`
(timestamped). Each entry records what each agent read (raw input), what it
produced (raw output), token counts, and wall-clock timing — useful for
profiling and for sizing token spend before switching to a metered API.
Configure via `transcript_dir` / `transcript_format` (`jsonl` | `md`) in
`DEFAULT_CONFIG`; set `transcript_dir=""` to disable.

---

## Tests & diagnostics

```powershell
# Fast deterministic unit tests:
.\venv\Scripts\python.exe -m pytest tests

# Manual end-to-end checks that drive the real LLM (slow):
.\venv\Scripts\python.exe diagnostics\diag_step5_analysts.py
.\venv\Scripts\python.exe diagnostics\diag_step6_portfolio.py
```

See `diagnostics/README.md` for the full list.

---

## Repository layout

```
run.py                  Interactive CLI entry point
portfolio_test.json     Example portfolio fixture
docs/                   Guides (provider switching, …)
prompts/                System prompts, one per agent role
diagnostics/            Manual LLM-driven end-to-end checks
tests/                  Fast deterministic unit tests
agentic/                The package
├── default_config.py   DEFAULT_CONFIG + get_config()
├── agents/             Analyst + portfolio agent factories
├── graph/              TradingGraph (per-ticker) + PortfolioGraph (Step 6)
├── dataflows/          yfinance fetchers + tool registry
└── llm_clients/        Provider factory + usage/transcript callback
```
