"""Real-data GARCH validation + (p,q) order selection.

Reproduces, on LIVE market data, the two things the unit tests only show on
SIMULATED data:
  1. The fit is sane on real returns (stationary persistence, sensible vol
     forecast vs the realised sample vol).
  2. Order selection: fit GARCH(p,q) over a grid + GJR(1,1) and rank by BIC and
     AIC, so we SEE which order each real asset prefers rather than assuming
     (1,1). Prints the full ranked table = the evidence behind the model choice.

This needs a network call (yfinance), so it lives in diagnostics/ (not the
offline unit-test suite). Run:

    .\\venv\\Scripts\\python.exe diagnostics\\diag_garch_realdata.py
    .\\venv\\Scripts\\python.exe diagnostics\\diag_garch_realdata.py --tickers SPY QQQ TLT --max-p 3 --max-q 3

Empirical expectation (a documented stylized fact, not an assumption): daily
equity vol almost always lands on a LOW order — (1,1) or an effective tie with
it — and equities prefer GJR over plain GARCH (the leverage effect). Bonds/FX
are more often symmetric. The table lets you confirm this per asset.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401 — puts project root on sys.path
import numpy as np

# Windows consoles default to cp1252 and choke on stray unicode; force UTF-8 so
# the table (and any model output) prints regardless of the active code page.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from agentic.dataflows.y_finance import _fetch_history
from agentic.quant.garch import order_search


def _log_returns(ticker: str, lookback_days: int) -> np.ndarray:
    h = _fetch_history(ticker, lookback_days=lookback_days)
    close = h["Close"].dropna()
    return np.log(close / close.shift(1)).dropna().values


def _label(fit) -> str:
    if fit.model == "gjr":
        return "GJR(1,1)"
    return f"GARCH({fit.p},{fit.q})"


def analyse(ticker: str, lookback_days: int, max_p: int, max_q: int) -> bool:
    try:
        rets = _log_returns(ticker, lookback_days)
    except Exception as exc:  # noqa: BLE001
        print(f"[{ticker}] FETCH ERROR: {exc}")
        return False
    if rets.size < 250:
        print(f"[{ticker}] only {rets.size} returns — skipping (need ~250+).")
        return False

    samp_daily = float(np.std(rets))
    print(f"\n=== {ticker} ===")
    print(f"  n returns       : {rets.size}")
    print(f"  sample daily vol: {samp_daily:.4f}  (~annual {samp_daily*np.sqrt(252)*100:.1f}%)")

    ranked_bic = order_search(rets, max_p=max_p, max_q=max_q, include_gjr=True, criterion="bic")
    ranked_aic = order_search(rets, max_p=max_p, max_q=max_q, include_gjr=True, criterion="aic")
    best_bic = ranked_bic[0]
    best_aic = ranked_aic[0]

    print(f"  {'model':<11}{'BIC':>12}{'AIC':>12}{'loglik':>12}{'persist':>9}  params")
    for f in ranked_bic:
        a = [round(x, 3) for x in f.params["alpha_vec"]]
        b = [round(x, 3) for x in f.params["beta_vec"]]
        g = f"  g={f.params['gamma']:.3f}" if f.model == "gjr" else ""
        print(f"  {_label(f):<11}{f.bic:>12.2f}{f.aic:>12.2f}{f.loglik:>12.2f}"
              f"{f.persistence:>9.4f}  a={a} b={b}{g}")

    fc = best_bic.forecast_vol(1)
    if len(ranked_bic) > 1:
        gap = ranked_bic[1].bic - best_bic.bic
        print(f"  -> BIC pick : {_label(best_bic)}  (dBIC vs runner-up {gap:+.2f})")
    else:
        print(f"  -> BIC pick : {_label(best_bic)}")
    print(f"  -> AIC pick : {_label(best_aic)}")
    print(f"  -> 1-step daily vol forecast: {fc:.4f}  (~annual {fc*np.sqrt(252)*100:.1f}%)")

    # Sanity: a stationary fit with a forecast in the right ballpark of sample vol.
    ok = (best_bic.persistence < 1.0) and (0.2 * samp_daily < fc < 5 * samp_daily)
    print(f"  -> sanity (stationary & forecast near sample vol): {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description="Real-data GARCH (p,q) order selection.")
    p.add_argument("--tickers", nargs="+",
                   default=["SPY", "AAPL", "QQQ", "TLT", "GLD"],
                   help="Tickers to fit (mix equities + bonds/commodities to see "
                        "GJR-vs-GARCH and order differ).")
    p.add_argument("--lookback-days", type=int, default=750,
                   help="Calendar days of history (~3y default).")
    p.add_argument("--max-p", type=int, default=2)
    p.add_argument("--max-q", type=int, default=2)
    args = p.parse_args()

    print("GARCH real-data order selection")
    print(f"grid: GARCH(1..{args.max_p}, 1..{args.max_q}) + GJR(1,1); "
          f"ranked by BIC (parsimony) and AIC.")

    results = [analyse(t, args.lookback_days, args.max_p, args.max_q) for t in args.tickers]
    n_ok = sum(1 for r in results if r)
    print(f"\n{n_ok}/{len(results)} tickers passed the sanity check.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
