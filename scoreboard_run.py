"""Scoreboard CLI — score a forward-test weekday track's comparators.

Runs our model vs GARCH benchmark vs hold-initial-no-rebalance vs a random
portfolio distribution (default 1,000) vs the mean-variance efficient-frontier
tangency portfolio, on the SAME dates/universe/constraints, and prints the
return+risk table. The random cloud + frontier are reconstructed here OFFLINE
from prices (they are not part of the nightly NAV run). Needs >= 2 rebalances.

Usage (track path is outputs/forward/<portfolio>/<weekday>):
    python scoreboard_run.py --track outputs/forward/diversified/monday
    python scoreboard_run.py --track outputs/forward/concentrated/friday --random 2000
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()

from agentic.quant.run_scoreboard import scoreboard_for_track


def main() -> int:
    p = argparse.ArgumentParser(description="Score a forward-test track.")
    p.add_argument("--track", required=True, help="Path to outputs/forward/<weekday>")
    p.add_argument("--period", default="weekly", choices=["daily", "weekly", "monthly"])
    p.add_argument("--random", type=int, default=1000, help="Number of random portfolios.")
    args = p.parse_args()

    try:
        table = scoreboard_for_track(args.track, rebalance_period=args.period,
                                     n_random=args.random)
    except ValueError as e:
        print(f"[scoreboard] {e}")
        return 1
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
