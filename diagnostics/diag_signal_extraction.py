"""Phase 1b gate — does constrained structured extraction work on gpt-oss?

The stack has a documented history of "paper features" failing on the
reasoning-first gpt-oss model (the reasoning flag itself, num_predict truncation
-> empty content). The Step 7 plan therefore requires that we VERIFY
`with_structured_output(AnalystSignal)` actually returns a valid signal
**4/4 times under reasoning=True** before relying on the two-call extraction
pattern — exactly the way the reasoning flag was verified.

This runs Call 2 (extract_signal) over a fixed canned analyst report four times
and asserts every run yields a schema-valid AnalystSignal with at least one
claim. It is LLM-driven and hits Ollama cloud, so it needs .env credentials.

    .\\venv\\Scripts\\python.exe diagnostics\\diag_signal_extraction.py

Exits non-zero unless all four runs pass.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  — puts project root on sys.path
from dotenv import load_dotenv

load_dotenv()

from agentic.agents.analysts._common import extract_signal
from agentic.agents.utils.signals import AnalystSignal
from agentic.default_config import get_config
from agentic.llm_clients.factory import create_llm_pair

# A realistic, opinionated analyst report — directional, with several
# checkable claims, so a correct extraction should yield direction > 0 and
# multiple Claims.
_CANNED_REPORT = """\
## Market Analyst — NVDA (as of 2026-06-14)

NVDA closed the week up 4.2% on heavy volume, breaking above its 50-day moving
average for the first time since April. Momentum is clearly constructive: RSI
sits at 61 (rising, not yet overbought) and MACD crossed positive three sessions
ago. The 20-day realized volatility has compressed to 28% from 41% a month ago,
which historically precedes continuation moves in this name.

Demand signals remain strong — datacenter GPU backlog commentary from the last
earnings call points to revenue growth of roughly 35% year-over-year next
quarter. I expect the stock to outperform the semiconductor index over the next
month, with upside toward the prior high. The main risk is a broad tech
drawdown, but absent that I am bullish on a 1-month horizon.
"""

N_RUNS = 4


def main() -> int:
    cfg = get_config()
    # quick tier is what the analysts use; extraction rides the same tier.
    _deep, quick = create_llm_pair(
        cfg,
        deep_num_predict=cfg.get("deep_num_predict"),
        deep_num_ctx=cfg.get("deep_num_ctx"),
    )

    passes = 0
    for i in range(1, N_RUNS + 1):
        try:
            sig = extract_signal(
                quick, _CANNED_REPORT, ticker="NVDA",
                trade_date="2026-06-14", node_name="market",
            )
        except Exception as e:  # noqa: BLE001
            print(f"run {i}/{N_RUNS}: RAISED {e!r}")
            continue

        ok = (
            isinstance(sig, AnalystSignal)
            and -1.0 <= sig.direction <= 1.0
            and 0.0 <= sig.confidence <= 1.0
            and len(sig.key_evidence) >= 1
            # the canned report is clearly bullish; a working extraction should
            # not return a flat-zero (the neutral fallback) signal.
            and sig.confidence > 0.0
        )
        status = "PASS" if ok else "FAIL"
        print(
            f"run {i}/{N_RUNS}: {status}  "
            f"direction={sig.direction:+.2f} confidence={sig.confidence:.2f} "
            f"horizon={sig.horizon} claims={len(sig.key_evidence)}"
        )
        if ok:
            passes += 1
            if i == 1:  # show one full claim set for eyeballing
                for c in sig.key_evidence:
                    line = f"    - [{c.metric}/{c.direction}/{c.horizon}] {c.statement}"
                    # The model can emit non-ASCII punctuation (e.g. non-breaking
                    # hyphens); the Windows cp1252 console chokes on those. Strip
                    # to ASCII for display only — the stored Claim keeps the original.
                    print(line.encode("ascii", "replace").decode("ascii"))

    print(f"\n{passes}/{N_RUNS} runs produced a valid bullish signal.")
    if passes == N_RUNS:
        print("GATE PASS — constrained extraction is reliable on this stack.")
        return 0
    print("GATE FAIL — extraction is NOT reliable; revisit the two-call design "
          "(e.g. format='json' + manual parse, or prompt tweaks).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
