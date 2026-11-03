"""Diagnostics path bootstrap.

These scripts live in diagnostics/ but import the `agentic` package and read
fixtures (portfolio_test.json) that live at the project root. Importing this
module first puts the project root on sys.path and exposes ROOT so the scripts
run correctly from any working directory:

    python diagnostics/diag_step6_portfolio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
