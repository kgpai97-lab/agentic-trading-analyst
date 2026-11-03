"""Forward-test CLI — advance all portfolios' weekday tracks once.

Three distinct portfolios (diversified / conservative / concentrated), each run
on all five weekdays → 15 independent tracks (Step 7 Phase 3). ONE invocation of
this script runs every portfolio for TODAY's weekday (weekends map to the Friday
track), so a single nightly scheduled task advances that weekday's tracks across
all portfolios. Records accumulate behind all later development.

Two VARIANTS of the desk run in parallel, seeded from the SAME fixtures so they
start identical and diverge only by ANALYSIS DEPTH:
  * prod  — priority_threshold 0.5 (production): the Junior PA deep-analyses only
            the holdings covering the top 50% of priority mass (a few names).
            Tracks: outputs/forward/<portfolio>/<weekday>/.
  * full  — priority_threshold 1.0 (full coverage): deep-analyses EVERY non-cash
            holding (the most expensive setting). The quality upper bound — if
            its decisions ≈ prod's, the cheaper 0.5 setting is justified.
            Tracks: outputs/forward_full/<portfolio>/<weekday>/.

Usage:
    python forward_run.py                       # prod variant, all portfolios, today
    python forward_run.py --variant both        # prod AND full (nightly default)
    python forward_run.py --variant full        # full-coverage only
    python forward_run.py --date 2026-06-15      # a specific date's weekday
    python forward_run.py --only concentrated    # just one portfolio
    python forward_run.py --variant both --debug

Outputs land under <dir>/<portfolio>/<weekday>/ (portfolio.json, history.jsonl,
reports/, transcripts/, snapshots/, nav.jsonl). First run of any track seeds it
from that portfolio's fixture.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()

from agentic.forward.runner import DEFAULT_PORTFOLIOS, ForwardTest

# Each variant: (label, forward_dir, config-overrides). prod = today's behaviour;
# full = priority_threshold 1.0 into a sibling dir. Both reuse the runner as-is.
_PROD_DIR = "outputs/forward"
_FULL_DIR = "outputs/forward_full"


def _run_variant(label: str, forward_dir: str, overrides: dict,
                 only: str | None, date: str | None, debug: bool,
                 seed_hold_built: bool = False) -> list[dict]:
    """Advance one variant's tracks and print a labelled summary."""
    ft = ForwardTest(forward_dir=forward_dir, config=overrides, debug=debug,
                     seed_hold_built=seed_hold_built)
    if only:
        records = [ft.run_one(only, run_date=date)]
    else:
        records = ft.run_all(run_date=date)

    print(f"\n=== Forward run summary [{label}] (dir={forward_dir}) ===")
    for rec in records:
        if "error" in rec:
            print(f"  {rec['portfolio']:<13} {rec.get('date','')}  ERROR: {rec['error']}")
            continue
        tw = rec["target_weights"]
        top = sorted(tw.items(), key=lambda kv: -kv[1])[:3]
        top_str = ", ".join(f"{t} {w:.2f}" for t, w in top)
        print(f"  {rec['portfolio']:<13} {rec['weekday']:<9} {rec['date']}  "
              f"{len(tw)} positions  | top: {top_str}")
    n_ok = sum(1 for r in records if "error" not in r)
    print(f"  {n_ok}/{len(records)} portfolios completed.")
    return records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Advance forward-test weekday tracks.")
    p.add_argument("--date", default=None,
                   help="Analysis date YYYY-MM-DD (default: today). Selects the weekday.")
    p.add_argument("--only", default=None, choices=list(DEFAULT_PORTFOLIOS),
                   help="Run only this one portfolio instead of all three.")
    p.add_argument("--variant", default="prod", choices=["prod", "full", "both"],
                   help="prod = 0.5 (default); full = 1.0 full coverage; both = run each.")
    p.add_argument("--dir", default=_PROD_DIR,
                   help="Root directory for the PROD (0.5) tracks.")
    p.add_argument("--full-dir", default=_FULL_DIR,
                   help="Root directory for the FULL (1.0) tracks.")
    p.add_argument("--debug", action="store_true", help="Verbose per-node logging.")
    args = p.parse_args(argv)

    # (label, forward_dir, config-overrides, seed_hold_built) per variant.
    # hold_built is derived from the FULL (1.0) run only — see _run_variant.
    plan: list[tuple[str, str, dict, bool]] = []
    if args.variant in ("prod", "both"):
        plan.append(("prod", args.dir, {}, False))
    if args.variant in ("full", "both"):
        plan.append(("full", args.full_dir, {"priority_threshold": 1.0}, True))

    # Run each variant; isolate a variant failure so the other still runs
    # (mirrors run_all's per-portfolio isolation).
    all_ok = True
    for label, fdir, overrides, hold_built in plan:
        try:
            records = _run_variant(label, fdir, overrides, args.only, args.date,
                                   args.debug, seed_hold_built=hold_built)
            all_ok = all_ok and all("error" not in r for r in records)
        except Exception as exc:  # noqa: BLE001 — isolate per-variant failure
            print(f"\n[forward] variant {label!r} FAILED: {exc}")
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
