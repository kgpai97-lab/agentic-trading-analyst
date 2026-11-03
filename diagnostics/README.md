# Diagnostics

Manual, LLM-driven end-to-end checks for the live graphs. These were the
old `verify_step*.py` / `profile_step5.py` gate scripts; they are kept here
as diagnostic tools, not part of any automated test suite (they hit Ollama
and take minutes to run).

Run from the project root with the venv interpreter:

```
.\venv\Scripts\python.exe diagnostics\diag_step4_sentiment.py    # sentiment-only tool loop
.\venv\Scripts\python.exe diagnostics\diag_step5_analysts.py     # all four analysts, stock
.\venv\Scripts\python.exe diagnostics\diag_step5b_stock_etf.py   # stock + ETF, both paths
.\venv\Scripts\python.exe diagnostics\diag_step6_portfolio.py    # portfolio rebalance (slow)
.\venv\Scripts\python.exe diagnostics\diag_profile_step5.py      # wall-clock + token profile
```

`_bootstrap.py` puts the project root on `sys.path` and exposes `ROOT` so the
scripts import `agentic` and locate `portfolio_test.json` regardless of the
working directory. Each script imports it first.

The fast, deterministic unit tests live in `../tests/` (run with `pytest`).
