"""Single source of truth for every configuration key in the system.

Agentic concept: a shared config dict is the simplest way to give every agent
and graph node the same runtime parameters without threading arguments everywhere.
LangGraph nodes receive state, not config — so config lives here and is read at
startup, not passed per-invocation.

To override any key without editing this file, set an env var named
agentic_<KEY> (e.g. agentic_LLM_PROVIDER=openai).
"""

import os
from copy import deepcopy

DEFAULT_CONFIG: dict = {
    # --- LLM provider --------------------------------------------------------
    # Which chat-model provider to use. The factory (llm_clients/factory.py)
    # reads this and returns the right LangChain chat-model object.
    # Supported: "ollama" | "openai" | "anthropic" | "google"
    "llm_provider": "ollama",

    # Model names for the two tiers used across the graph. Both tiers run
    # gpt-oss:120b (Ollama cloud, free tier).
    # deep_think_llm  → Head Analyst, Researcher, Portfolio Manager (synthesis)
    # quick_think_llm → all analyst nodes + Junior PA
    #
    # Was gpt-oss:20b until 2026-06-14, when the 20b cloud variant started
    # returning EMPTY content (tokens generated server-side but surfaced in
    # neither `content` nor `thinking`; done_reason="stop", so not a budget
    # issue). 120b returns content normally on the same key/endpoint. See .env.
    "deep_think_llm": "gpt-oss:120b",
    "quick_think_llm": "gpt-oss:120b",

    # Thinking ("reasoning") toggle per tier. Maps to ChatOllama's `reasoning`
    # field (the Ollama server `think` option): bool | "low"/"medium"/"high" | None.
    # gpt-oss:20b is a reasoning-FIRST model — must be True on both tiers.
    # reasoning=False causes langchain-ollama to discard the analysis channel → empty content.
    "deep_reasoning":  True,
    "quick_reasoning": True,

    "deep_num_predict": 16384,            # max tokens the deep tier may generate; must be < deep_num_ctx
    "deep_num_ctx":    32768,             # context window for deep-tier Ollama calls; must exceed deep_num_predict + input length

    # Base URL for Ollama's HTTP server (ignored for cloud providers).
    "ollama_base_url": "http://localhost:11434",

    # --- Asset type ---------------------------------------------------------
    # Controls which fourth-slot analyst the graph compiles.
    #   "stock" → Fundamentals Analyst (P/E, EPS, margins, income statement)
    #   "etf"   → Composition Analyst  (top holdings, sector tilt)
    # "crypto" is accepted but currently behaves like "stock".
    "asset_type": "stock",

    # --- Graph behaviour -----------------------------------------------------
    "max_debate_rounds": 1,         # bull/bear rounds (Step 6+)
    "max_risk_discuss_rounds": 1,   # risk debate rounds (Step 6+)
    "analyst_concurrency_limit": 1, # 1 = sequential, >1 = parallel (Step 5+)
    "checkpoint_enabled": False,    # SQLite resume on crash (deferred)

    # Step 7 Phase 1b: run the cheap second LLM call that distills each analyst's
    # prose into a validated AnalystSignal (numeric direction + falsifiable
    # claims), written to state["signals"]. True by default; set False to skip
    # the extra call on fast/smoke runs.
    "extract_signals": True,

    # --- Output ---------------------------------------------------------------
    "output_language": "English",
    "results_dir": "outputs/briefings",

    # If True, propagate() prints the UsageCallback report at the end of a run.
    # debug=True implies this. See agentic/llm_clients/usage_callback.py.
    "print_usage": False,

    # If set, propagate() writes a full per-agent transcript (raw input each
    # agent read, raw output it produced, timestamps + token/latency) to a
    # timestamped file under this directory after every run. Empty = disabled.
    # File format follows the extension of transcript_format ("jsonl" | "md").
    "transcript_dir":    "outputs/transcripts",
    "transcript_format": "jsonl",

    # If set, every data fetch (route_to_vendor) writes a write-only snapshot of
    # its raw output under <snapshot_dir>/<as_of_date>/<TICKER>/<tool>.json for
    # provenance + reproducibility (Phase 2). Empty = disabled (default; tests
    # and ad-hoc runs don't litter snapshots). The forward-test runner sets it.
    "snapshot_dir": "",

    # If set, PortfolioGraph.propagate() writes the rendered markdown report
    # AND the structured final_decision (as JSON) to a timestamped pair of
    # files under this directory after every run. Empty = disabled (report is
    # still returned/printed, just not persisted). Mirrors transcript_dir.
    "report_dir": "outputs/reports",

    # --- Data vendor selection ------------------------------------------------
    # Each category can be "yfinance" or "alpha_vantage" (Step 4+).
    "data_vendors": {
        "core_stock_apis":      "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data":     "yfinance",
        "news_data":            "yfinance",
    },

    # --- News limits ----------------------------------------------------------
    "news_article_limit":        20,
    "global_news_article_limit": 10,

    # --- Lookback window ------------------------------------------------------
    # Number of calendar days of historical data the market/indicator tools fetch.
    # Accepted values via --window in run.py: 1, 7, 30. Broader values are also
    # accepted programmatically (e.g. 60 for backtests).
    "lookback_days": 30,

    # --- Portfolio mode (Step 6) ---------------------------------------------
    # Cumulative-priority cutoff for the Junior Portfolio Analyst's selection
    # of holdings to deep-analyse. Higher = more tickers analysed = more cost.
    "priority_threshold": 0.5,

    # How many PM-uncertainty -> Researcher passes are allowed after the
    # initial PM decision. 0 disables the feedback loop entirely. The plan
    # caps this at a single pass; raise only with caution.
    "max_pm_uncertainty_rounds": 1,

    # --- Rebalance cadence & damping (Step 7, Phase 1e) ----------------------
    # Partial-adjustment factor: the fraction of the PM's full-conviction move
    # executed per rebalance. 0.1 → ~10 rebalances (≈10 weeks weekly) to
    # complete a full reallocation, damping over-trading / noise-chasing at a
    # weekly cadence. Set 1.0 to recover full-move behavior. Applied in code
    # (LLM-judges / code-does-math); the GARCH benchmark (Phase 4a) uses the
    # SAME factor on its exposure scalar so the comparison is fair.
    "cadence_factor": 0.1,

    # Two-stage cadence (forward test): the FIRST rebalance of a track is the
    # decisive initial repositioning (client's raw seed → an objective-meeting
    # book), so it applies this LARGER factor; subsequent rebalances use
    # `cadence_factor`. 1.0 = full first move. Only the forward runner uses this.
    "cadence_factor_initial": 0.5,

    # The rebalance period the forward test / GARCH module assume. Drives the
    # √t vol-horizon scaling in the GARCH benchmark (weekly → annual_target/√52).
    # One of "daily" | "weekly" | "monthly". The weekday forward-test runner
    # sets this to "weekly".
    "rebalance_period": "weekly",
}

# ---------------------------------------------------------------------------
# Env-var overrides (flat keys only — nested "data_vendors" not overridable
# via env for now; just edit the dict above for those).
# ---------------------------------------------------------------------------
_ENV_OVERRIDES: dict[str, str] = {
    "agentic_LLM_PROVIDER":            "llm_provider",
    "agentic_DEEP_THINK_LLM":          "deep_think_llm",
    "agentic_QUICK_THINK_LLM":         "quick_think_llm",
    "agentic_DEEP_REASONING":          "deep_reasoning",
    "agentic_QUICK_REASONING":         "quick_reasoning",
    "agentic_OLLAMA_BASE_URL":         "ollama_base_url",
    "agentic_MAX_DEBATE_ROUNDS":       "max_debate_rounds",
    "agentic_MAX_RISK_DISCUSS_ROUNDS": "max_risk_discuss_rounds",
    "agentic_OUTPUT_LANGUAGE":         "output_language",
    "agentic_ASSET_TYPE":              "asset_type",
    "agentic_LOOKBACK_DAYS":           "lookback_days",
    "agentic_PRIORITY_THRESHOLD":      "priority_threshold",
    "agentic_DEEP_NUM_PREDICT":        "deep_num_predict",
    "agentic_DEEP_NUM_CTX":           "deep_num_ctx",
    "agentic_CADENCE_FACTOR":          "cadence_factor",
    "agentic_REBALANCE_PERIOD":        "rebalance_period",
    "agentic_EXTRACT_SIGNALS":         "extract_signals",
}


def get_config(overrides: dict | None = None) -> dict:
    """Return a config dict with env-var and caller overrides applied.

    Usage:
        cfg = get_config({"llm_provider": "openai", "deep_think_llm": "gpt-4o"})
    """
    cfg = deepcopy(DEFAULT_CONFIG)

    for env_key, cfg_key in _ENV_OVERRIDES.items():
        val = os.getenv(env_key)
        if val is not None:
            # Coerce numeric strings back to the type of the default.
            # bool must come before int (bool is a subclass of int in Python).
            if isinstance(cfg[cfg_key], bool):
                val = val.lower() in ("1", "true", "yes")
            elif isinstance(cfg[cfg_key], int):
                val = int(val)
            elif isinstance(cfg[cfg_key], float):
                val = float(val)
            cfg[cfg_key] = val

    if overrides:
        cfg.update(overrides)

    return cfg
