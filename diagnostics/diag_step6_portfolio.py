"""Manual end-to-end test for Step 6 (portfolio-aware decision layer).

New order (Step 6 reorder): analysts run for ALL holdings → Junior PA composition
assessment + focus selection → Head/Researcher on the focus list → PM.

Gate:
  1. Junior PA produces a non-empty focus list (selected_tickers); when a main
     ticker is supplied it must be first (always analysed + selected).
  2. Every holding has a non-empty briefing; every FOCUS ticker additionally has
     head_assessment + researcher_report in state["per_ticker_reports"].
  3. PM final_decision has:
      - target_weights dict that sums to 1.0 ± 0.01
      - per_position_rationale covers every CHANGED position (held positions
        are intentionally omitted under the delta model)
      - narrative is non-empty

Slow: analysts now run on EVERY holding (not just the focus subset), so this is
slower than before. Plan for ~10-15 minutes depending on portfolio size.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import _bootstrap  # noqa: F401  -- puts project root on sys.path

from dotenv import load_dotenv
load_dotenv()

from agentic.graph.portfolio_graph import PortfolioGraph

PORTFOLIO_PATH = str(_bootstrap.ROOT / "portfolio_test.json")
MAIN_TICKER    = "DELL"
TRADE_DATE     = "2026-06-01"


def main() -> int:
    print(f"\n{'#'*72}\n# Step 6 gate: portfolio rebalance for {MAIN_TICKER}\n{'#'*72}")
    pg = PortfolioGraph(
        config={"asset_type": "stock", "lookback_days": 7, "print_usage": True},
        debug=True,
    )
    # VTIP is a TIPS bond fund; route it through the income_alt analyst.
    state, report = pg.propagate(
        MAIN_TICKER, PORTFOLIO_PATH, TRADE_DATE, main_ticker_type="equity"
    )

    ok = True

    # Expected fourth-section heading per asset class (Step 6b routing).
    HEADING_BY_CLASS = {
        "equity":           "Fundamentals",
        "equity_fund":      "Composition",
        "bond_fund":        "Fixed Income / Alternatives",
        "alternative_fund": "Fixed Income / Alternatives",
    }

    # 1. Junior PA selection
    selected = state.get("selected_tickers", [])
    print(f"\nselected_tickers: {selected}")
    if not selected:
        print("  FAIL: empty selected_tickers")
        ok = False
    elif not MAIN_TICKER:
        # Pure rebalance: no main ticker to anchor first.
        print("  OK: pure rebalance (no main ticker)")
    elif selected[0] != MAIN_TICKER:
        print(f"  FAIL: main ticker {MAIN_TICKER} not first in selected_tickers")
        ok = False
    else:
        print("  OK: main ticker is first")

    # 2. Per-ticker reports
    per_ticker = state.get("per_ticker_reports", {})
    ticker_types = state.get("ticker_types", {})

    # 2a. Every non-cash holding (+ main) must have a composition report
    # (composition pass runs for ALL holdings).
    current = state.get("current_weights", {})
    expected_analysed = {t for t in current if t.upper() != "CASH"}
    if MAIN_TICKER:
        expected_analysed.add(MAIN_TICKER)
    print("\nall holdings have composition pass (composition_report present):")
    for t in sorted(expected_analysed):
        c = per_ticker.get(t, {}).get("composition_report", "")
        c_ok = bool(c.strip())
        print(f"  {t:<8s} composition={len(c):>5}  {'OK' if c_ok else 'FAIL'}")
        if not c_ok:
            ok = False

    # 2b. Every FOCUS ticker additionally has MNS reports + briefing + head +
    # researcher. Non-focus tickers must NOT have MNS (cost-control split).
    print("\nfocus tickers have deep chain (mns + briefing + head + researcher):")
    for t in selected:
        payload = per_ticker.get(t, {})
        mns = payload.get("mns_reports", {})
        b = payload.get("briefing", "")
        h  = payload.get("head_assessment", "")
        r  = payload.get("researcher_report", "")
        mns_ok = bool(mns.get("market","").strip() and mns.get("news","").strip()
                      and mns.get("sentiment","").strip())
        all_ok = mns_ok and b.strip() and h.strip() and r.strip()
        print(f"  {t:<8s} mns={'Y' if mns_ok else 'N'} briefing={len(b):>5} "
              f"head={len(h):>5} researcher={len(r):>5}  {'OK' if all_ok else 'FAIL'}")
        if not all_ok:
            ok = False

    non_focus = sorted(expected_analysed - set(selected))
    print("\nnon-focus tickers must NOT have MNS/deep chain (cost split):")
    for t in non_focus:
        payload = per_ticker.get(t, {})
        leaked = bool(payload.get("mns_reports") or payload.get("head_assessment","").strip()
                      or payload.get("researcher_report","").strip())
        print(f"  {t:<8s} {'FAIL (has MNS/deep chain)' if leaked else 'OK (composition only)'}")
        if leaked:
            ok = False

    # 2c. Asset-class routing: each briefing's fourth section heading must
    # match the ticker's declared class.
    print("\nfourth-section heading per selected ticker:")
    for t in selected:
        cls = ticker_types.get(t, "equity")
        expected = HEADING_BY_CLASS.get(cls, "Fundamentals")
        briefing = per_ticker.get(t, {}).get("briefing", "")
        marker = f"# {expected} Analyst"
        if marker in briefing:
            print(f"  {t:<8s} class={cls:<16s} -> '{expected}'  OK")
        else:
            print(f"  {t:<8s} class={cls:<16s} -> expected '{expected}' heading, NOT FOUND  FAIL")
            ok = False

    # 3. Final decision shape
    decision = state.get("final_decision", {})
    weights  = decision.get("target_weights", {})
    rats     = decision.get("per_position_rationale", {})
    narrative = decision.get("narrative", "")

    total = sum(float(w) for w in weights.values()) if weights else 0.0
    print(f"\nfinal target_weights ({len(weights)} entries) sum = {total:.4f}")
    if not weights:
        print("  FAIL: empty target_weights"); ok = False
    elif abs(total - 1.0) > 0.01:
        print(f"  FAIL: weights sum {total:.4f} outside 1.0 ± 0.01"); ok = False
    else:
        print("  OK")

    # Delta model: rationale must cover every CHANGED position (held positions
    # are intentionally omitted). A change = target weight differs from current.
    current = state.get("current_weights", {})
    changed = {
        t for t in set(weights) | set(current)
        if abs(float(weights.get(t, 0.0)) - float(current.get(t, 0.0))) > 0.001
    }
    missing_rats = changed - set(rats)
    if missing_rats:
        print(f"  FAIL: changed positions missing per_position_rationale: {missing_rats}"); ok = False
    else:
        print(f"  OK: per_position_rationale covers all {len(changed)} changed positions")

    if not narrative.strip():
        print("  FAIL: empty narrative"); ok = False
    else:
        print(f"  OK: narrative is {len(narrative)} chars")

    print("\n" + "="*72)
    print(f"  {'PASS' if ok else 'FAIL'}")
    print("="*72 + "\n")

    if ok:
        print("--- RENDERED REPORT ---\n")
        print(report)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
