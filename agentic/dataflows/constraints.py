"""Hard portfolio constraints — schema, enforcement, and load-time validation.

This module implements **Decision B** from the Step 7 roadmap (see the project docs
"Design records"): numeric portfolio limits (per-holding cap, bond/cash minima)
are enforced by CODE at one final layer — never by prompt instruction. The
objective's prose stays as *soft* reasoning guidance for the LLM; only the
numbers move into code enforcement.

Three pieces:

  * ``PortfolioConstraints`` — the pydantic schema for the optional
    ``constraints`` block in a portfolio JSON.
  * ``enforce_constraints`` — a pure function applied as the SINGLE last step of
    a rebalance (after the PM and any future overlay), to BOTH the PM portfolio
    and the GARCH benchmark, so every comparator plays by identical limits.
  * ``scan_objective_for_numeric_constraints`` — a load-time validator that
    warns when a numeric limit appears stranded in the ``objective`` prose
    (where it would be unenforced), nudging the user to move it into the
    ``constraints`` block.

Why enforcement must live here and not in a prompt: once a risk overlay can
scale the cash sleeve (Decision A), a prompt-level "keep >=10% cash" is unsafe —
the overlay could violate a minimum the PM respected. One enforcement point, in
code, applied after every transform, is the only safe design.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator

# Holding-type vocabulary (mirrors portfolio_graph's _VALID_CLASSES). We only
# need to know which types are "bond" and which is "cash" for the minima.
_BOND_TYPES = {"bond_fund"}
_CASH_TYPE = "cash"
# Equity-like types whose aggregate weight `max_total_stock` caps.
_STOCK_TYPES = {"equity", "equity_fund"}

# Float tolerance for "sums to 1.0" checks — matches portfolio_graph's idiom.
_TOL = 1e-9


class PortfolioConstraints(BaseModel):
    """Numeric hard limits enforced in code after all weight transforms.

    All fields optional; a missing field means "no limit on this dimension"
    (permissive default = today's behavior). Weights are fractions in [0, 1].
    """

    max_single_position: float | None = Field(default=None, ge=0.0, le=1.0)
    max_total_stock: float | None = Field(default=None, ge=0.0, le=1.0)
    min_bond_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    max_cash: float | None = Field(default=None, ge=0.0, le=1.0)
    min_cash: float | None = Field(default=None, ge=0.0, le=1.0)
    unparsed_notes: str = ""

    @model_validator(mode="after")
    def _coherent(self) -> "PortfolioConstraints":
        if (
            self.min_cash is not None
            and self.max_cash is not None
            and self.min_cash > self.max_cash + _TOL
        ):
            raise ValueError(
                f"min_cash ({self.min_cash}) exceeds max_cash ({self.max_cash})"
            )
        return self

    @classmethod
    def from_dict(cls, data: dict | None) -> "PortfolioConstraints":
        """Build from a portfolio JSON's ``constraints`` block (or None)."""
        if not data:
            return cls()
        return cls(**data)


# ---------------------------------------------------------------------------
# Load-time validation — catch numeric limits stranded in the objective prose
# ---------------------------------------------------------------------------

# Phrases that, paired with a percentage, almost always denote a hard limit the
# user expects to be enforced. Deliberately conservative: we warn, never block.
_LIMIT_CUE = re.compile(
    r"(no more than|more than|at most|maximum|max\.?|at least|minimum|min\.?|"
    r"no less than|less than|cap(?:ped)?|limit|under|below|above|over|"
    r"exceed|never)"
    r"[^.\n]{0,40}?(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def scan_objective_for_numeric_constraints(objective: str) -> list[str]:
    """Return human-readable warnings for numeric limits found in prose.

    A casual user might write "never hold more than 15% in one stock" in the
    objective text, expecting it to be enforced — but per Decision B only the
    ``constraints`` block is enforced. This scan surfaces such phrases so the
    caller can warn the user to move them into ``constraints``. Returns an empty
    list when nothing suspicious is found.
    """
    warnings: list[str] = []
    for match in _LIMIT_CUE.finditer(objective or ""):
        snippet = match.group(0).strip()
        warnings.append(
            f"objective prose contains a numeric limit ({snippet!r}) that is "
            "NOT enforced — move it into the 'constraints' block to enforce it."
        )
    return warnings


# ---------------------------------------------------------------------------
# enforce_constraints — the single final hard-limit layer (Phase 1d)
# ---------------------------------------------------------------------------

def _normalise(weights: dict[str, float]) -> dict[str, float]:
    """Drop tiny/negative weights and renormalise the rest to sum to 1.0."""
    cleaned = {t: w for t, w in weights.items() if w > _TOL}
    total = sum(cleaned.values())
    if total <= _TOL:
        return cleaned
    return {t: w / total for t, w in cleaned.items()}


def enforce_constraints(
    weights: dict[str, float],
    ticker_types: dict[str, str],
    constraints: PortfolioConstraints,
) -> dict[str, float]:
    """Clamp `weights` to satisfy every numeric limit, then renormalise to 1.0.

    Pure function — no I/O, no mutation of inputs. Applied identically to the PM
    portfolio and the GARCH benchmark (Phase 4a) so the comparison is
    apples-to-apples.

    The three limit families interact (lifting bonds shrinks cash; capping a
    name lifts others), so a naive one-pass application can re-violate a limit a
    later step set. We instead iterate the full set to a joint fixpoint:

      1. cash band (min_cash / max_cash) — pin the cash sleeve to its target;
      2. bond floor (min_bond_weight) — lift the bond sleeve to its floor,
         drawing the deficit ONLY from the equity/alt remainder (cash held
         fixed, so step 1 is preserved);
      2.5 aggregate equity ceiling (max_total_stock) — scale the combined
         equity sleeve down to the cap, pushing freed weight to bonds/alts
         (then cash);
      3. per-position cap (max_single_position) — clip non-cash offenders and
         redistribute the excess to uncapped non-cash names.

    The loop repeats until no limit is violated (or a small iteration budget is
    hit, after which it returns the best effort — possible only if the limits
    are mutually infeasible, e.g. cap × n_names < 1). With no constraints set,
    this is a single renormalise (today's behavior).
    """
    w = _normalise(dict(weights))
    if not w:
        return w

    cash_keys = [t for t in w if ticker_types.get(t, "").lower() == _CASH_TYPE]
    bond_keys = [t for t in w if ticker_types.get(t, "").lower() in _BOND_TYPES]
    stock_keys = [t for t in w if ticker_types.get(t, "").lower() in _STOCK_TYPES]
    cap = constraints.max_single_position

    def _cash_target(cur_cash: float) -> float:
        target = cur_cash
        if constraints.min_cash is not None:
            target = max(target, constraints.min_cash)
        if constraints.max_cash is not None:
            target = min(target, constraints.max_cash)
        return min(target, 1.0)

    for _ in range(32):
        before = dict(w)

        # ---- 1. Cash band --------------------------------------------------
        if (constraints.min_cash is not None or constraints.max_cash is not None) and cash_keys:
            cash_total = sum(w[t] for t in cash_keys)
            target = _cash_target(cash_total)
            if abs(target - cash_total) > _TOL:
                non_cash = {t: w[t] for t in w if t not in cash_keys}
                non_cash_total = sum(non_cash.values())
                for t in cash_keys:
                    share = (w[t] / cash_total) if cash_total > _TOL else 1.0 / len(cash_keys)
                    w[t] = target * share
                if non_cash_total > _TOL:
                    scale = (1.0 - target) / non_cash_total
                    for t in non_cash:
                        w[t] = non_cash[t] * scale
            w = _normalise(w)

        # ---- 2. Bond floor (cash held fixed) -------------------------------
        floor = constraints.min_bond_weight
        if floor is not None and bond_keys:
            bond_total = sum(w[t] for t in bond_keys)
            if bond_total < floor - _TOL:
                # Source the bond deficit from the non-cash, non-bond sleeve only,
                # so a satisfied cash floor (step 1) is not eroded.
                other = {t: w[t] for t in w if t not in bond_keys and t not in cash_keys}
                other_total = sum(other.values())
                cash_total = sum(w[t] for t in cash_keys)
                avail = 1.0 - cash_total           # weight not pinned to cash
                new_bond = min(floor, max(avail, 0.0))
                if bond_total > _TOL:
                    bscale = new_bond / bond_total
                    for t in bond_keys:
                        w[t] *= bscale
                else:
                    for t in bond_keys:
                        w[t] = new_bond / len(bond_keys)
                rem = max(avail - new_bond, 0.0)
                if other_total > _TOL:
                    oscale = rem / other_total
                    for t in other:
                        w[t] *= oscale
                w = _normalise(w)

        # ---- 2.5 Aggregate equity ceiling (max_total_stock) ---------------
        # Cap the COMBINED weight of all equity-type holdings (the concentrated
        # book's defining limit). Symmetric to the bond floor: when the stock
        # sleeve exceeds the cap, scale it down to the cap and push the freed
        # weight into non-stock, non-cash assets (bonds/alts) first, falling back
        # to cash if there is no other home.
        stock_cap = constraints.max_total_stock
        if stock_cap is not None and stock_keys:
            stock_total = sum(w[t] for t in stock_keys)
            if stock_total > stock_cap + _TOL:
                freed = stock_total - stock_cap
                sscale = stock_cap / stock_total if stock_total > _TOL else 0.0
                for t in stock_keys:
                    w[t] *= sscale
                # Prefer non-stock, non-cash recipients (bonds/alts); else cash.
                recip = {t: w[t] for t in w
                         if t not in stock_keys and t not in cash_keys and w[t] > _TOL}
                recip_total = sum(recip.values())
                if recip_total > _TOL:
                    for t in recip:
                        w[t] += freed * (recip[t] / recip_total)
                elif cash_keys:
                    per = freed / len(cash_keys)
                    for t in cash_keys:
                        w[t] += per
                else:
                    # No bonds/alts and no existing cash position: the freed
                    # equity weight is, by definition, cash. Create the slot —
                    # otherwise the final renormalise would just re-inflate the
                    # equity sleeve and silently undo the cap.
                    w["CASH"] = w.get("CASH", 0.0) + freed
                w = _normalise(w)

        # ---- 3. Per-position cap ------------------------------------------
        if cap is not None:
            for _inner in range(16):
                over = {t: w[t] for t in w
                        if ticker_types.get(t, "").lower() != _CASH_TYPE and w[t] > cap + _TOL}
                if not over:
                    break
                freed = sum(w[t] - cap for t in over)
                for t in over:
                    w[t] = cap
                recipients = {t: w[t] for t in w
                              if t not in over
                              and ticker_types.get(t, "").lower() != _CASH_TYPE
                              and w[t] > _TOL}
                rec_total = sum(recipients.values())
                if rec_total > _TOL:
                    for t in recipients:
                        w[t] += freed * (recipients[t] / rec_total)
                else:
                    break

        # Converged if nothing moved materially this pass.
        if all(abs(w.get(t, 0.0) - before.get(t, 0.0)) < _TOL for t in set(w) | set(before)):
            break

    return _normalise(w)
