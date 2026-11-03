"""Portfolio Manager — final allocation decision.

The PM emits a LIST OF CHANGES (deltas) against the current portfolio, not a
full weight vector. Code applies them to the baseline — positions not
mentioned are held unchanged, so the PM cannot accidentally drop a holding,
and mass is conserved by construction (changes must net to ~0).

Two callables:
  - run_initial(state) -> dict
  - run_uncertainty_pass(state, uncertainty_reports) -> dict

Both return the SAME final-decision shape consumed downstream:
  {
    "target_weights":         {ticker: float, sums to ~1.0},
    "per_position_rationale": {ticker: str},   # only for moved positions
    "narrative":              str,
    "uncertain_holdings":     [str, ...],
    "changes":                [ {ticker, action, amount, explanation}, ... ],
  }
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic.agents.portfolio._common import extract_json_block, load_prompt
from agentic.agents.portfolio.junior_pa import _composition_slice

# Signed effect of each action on a position's weight.
_ADD_ACTIONS = {"add", "buy"}
_REMOVE_ACTIONS = {"trim", "exit"}
_NET_TOLERANCE = 0.02   # how far the listed changes may net from zero


def _signed_delta(action: str, amount: float, current: float) -> float:
    """Convert an (action, amount) into a signed weight change.

    'exit' ignores `amount` and removes the position's full current weight,
    so a model that omits/mis-states the exit amount still zeroes it out.
    """
    a = action.lower().strip()
    amt = abs(float(amount))
    if a == "exit":
        return -current
    if a == "trim":
        return -amt
    if a in _ADD_ACTIONS:
        return +amt
    return 0.0   # unknown action → no-op (guarded again in apply)


def apply_changes(
    current_weights: dict[str, float],
    changes: list[dict],
    main_ticker: str = "",
) -> tuple[dict[str, float], dict[str, str], list[str]]:
    """Apply a list of PM changes to the baseline portfolio.

    Returns (target_weights, per_position_rationale, warnings). Positions not
    mentioned in `changes` are carried over unchanged. Weights are clamped to
    [0, 1], a fully-exited position is dropped, and the result is renormalised
    to sum to 1.0 so downstream code always sees a valid vector.
    """
    warnings: list[str] = []
    target = dict(current_weights)               # baseline = hold everything
    rationale: dict[str, str] = {}

    held = set(current_weights)
    net = 0.0
    for ch in changes or []:
        ticker = str(ch.get("ticker", "")).upper().strip()
        action = str(ch.get("action", "")).lower().strip()
        explanation = str(ch.get("explanation", "")).strip()
        try:
            amount = float(ch.get("amount", 0.0))
        except (TypeError, ValueError):
            amount = 0.0

        if not ticker:
            warnings.append(f"dropped change with no ticker: {ch!r}")
            continue
        if action not in _ADD_ACTIONS | _REMOVE_ACTIONS:
            warnings.append(f"{ticker}: unknown action {action!r}; skipped")
            continue
        # 'buy' may open the main ticker; everything else must be held.
        is_new_buy = action == "buy" and ticker == (main_ticker or "").upper()
        if ticker not in held and not is_new_buy:
            warnings.append(f"{ticker}: not a held position and not a main-ticker buy; skipped")
            continue
        if not explanation:
            warnings.append(f"{ticker}: change has no explanation; skipped")
            continue

        cur = float(target.get(ticker, 0.0))
        delta = _signed_delta(action, amount, cur)
        new_weight = round(cur + delta, 6)
        if new_weight < 0:
            warnings.append(
                f"{ticker}: trim {amount:.4f} exceeds weight {cur:.4f}; clamped to 0"
            )
            delta = -cur
            new_weight = 0.0

        net += delta
        target[ticker] = new_weight
        rationale[ticker] = explanation

    if abs(net) > _NET_TOLERANCE:
        warnings.append(
            f"listed changes net to {net:+.4f} (not zero); renormalising to conserve mass"
        )

    # Drop fully-exited positions; clamp; renormalise to 1.0.
    target = {t: max(w, 0.0) for t, w in target.items() if w > 1e-9}
    total = sum(target.values())
    if total > 0:
        target = {t: round(w / total, 6) for t, w in target.items()}

    return target, rationale, warnings


def scale_changes(changes: list[dict], cadence_factor: float) -> list[dict]:
    """Scale every change's `amount` by `cadence_factor` (partial adjustment).

    Phase 1e cadence damping (LLM-judges / code-does-math): the PM proposes its
    FULL-conviction move; this deterministic step shrinks each delta so only
    `cadence_factor` of the move is executed this rebalance. At the default 0.1
    (10%), a full reallocation completes over ~10 rebalances (~10 weeks at weekly
    cadence), damping whipsaw and noise-chasing. cadence_factor=1.0 recovers the
    full-move behavior.

    Action handling:
      - 'add' / 'buy' / 'trim': `amount` is multiplied by cadence_factor.
      - 'exit': left intact. apply_changes treats exit as amount-agnostic (it
        zeroes the whole position), so there is no magnitude to smooth — a full
        exit is a discrete de-risking decision, not a size to partially adjust.
        A PM that wants to scale OUT gradually should express that as 'trim'
        (which IS damped); a bare 'exit' is taken as a deliberate full close.
    """
    if cadence_factor >= 1.0:
        return changes or []
    scaled: list[dict] = []
    for ch in changes or []:
        action = str(ch.get("action", "")).lower().strip()
        new = dict(ch)
        if action == "exit":
            # A bare full-exit is a discrete decision; keep it. (Partial
            # de-risking should be expressed as 'trim' by the PM, which we damp.)
            scaled.append(new)
            continue
        try:
            amt = float(ch.get("amount", 0.0))
        except (TypeError, ValueError):
            amt = 0.0
        new["amount"] = amt * cadence_factor
        scaled.append(new)
    return scaled


def _format_weights(weights: dict[str, float]) -> str:
    return "\n".join(f"  {t:<8s} {w:.4f}" for t, w in weights.items())


def _format_research_section(per_ticker_reports: dict[str, dict],
                             focus_tickers: list[str]) -> str:
    """Render per-ticker analysis for the PM, FOCUS tickers only.

    For each focus ticker the PM gets, in order:
      • the Head Analyst assessment and the Researcher bull+bear note,
      • the market / sentiment / news ANALYST REPORTS (prose — the PM reads the
        analysts' written read, not the raw OHLCV/news dumps),
      • the COMPOSITION raw tool output (holdings + sector weights) — the only
        raw data forwarded, since look-through reasoning needs the numbers.

    Non-focus holdings are NOT rendered here; they appear in the
    non-prioritized block (weight + focus score) instead.
    """
    blocks = []
    for ticker in focus_tickers:
        payload = per_ticker_reports.get(ticker, {})
        head = (payload.get("head_assessment") or "").strip()
        research = (payload.get("researcher_report") or "").strip()
        mns = payload.get("mns_reports") or {}
        comp_raw = _composition_slice((payload.get("tool_outputs") or "").strip())

        parts = [f"### {ticker}"]
        if head:
            parts.append(f"**Head Analyst assessment:**\n\n{head}")
        if research:
            parts.append(f"**Bull + Bear research:**\n\n{research}")
        for label, key in (("Market", "market"), ("Sentiment", "sentiment"), ("News", "news")):
            rep = (mns.get(key) or "").strip()
            if rep:
                parts.append(f"**{label} analyst report:**\n\n{rep}")
        if comp_raw:
            parts.append(f"**Composition (raw holdings + sectors):**\n\n{comp_raw}")
        blocks.append("\n\n".join(parts))
    return "\n\n".join(blocks) if blocks else "(no focus-ticker analysis)"


def _junior_pa_block(state: dict) -> str:
    """The Junior PA's composition assessment + focus scores, as PM input.

    The PM treats this as INPUT to its own independent view — a higher focus
    score flags a holding the Junior PA judged needs attention (overlap /
    concentration / objective fit), not a mandate to trade it.
    """
    rationale = (state.get("priority_rationale") or "").strip()
    scores = state.get("priority_scores", {})
    selected = set(state.get("selected_tickers", []))

    score_lines = []
    for t, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        marker = "  (focus)" if t in selected else ""
        score_lines.append(f"  {t:<8s} focus_score={s:.3f}{marker}")
    score_block = "\n".join(score_lines) if score_lines else "  (no scores)"

    return (
        "JUNIOR PA COMPOSITION ASSESSMENT (input to your own analysis; HIGH "
        "focus_score = the desk flagged this holding for attention — overlap, "
        "concentration, or poor objective fit):\n"
        f"{score_block}\n\n"
        f"  Assessment narrative:\n  {rationale or '(none)'}"
    )


def _non_prioritized_block(state: dict) -> str:
    selected = set(state["selected_tickers"])
    held     = state["current_weights"]
    scores   = state.get("priority_scores", {})
    rows = []
    for t, w in held.items():
        if t in selected or t == "CASH":
            continue
        score = scores.get(t)
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        rows.append(f"  {t:<8s} weight={w:.4f}  focus_score={score_str}")
    if not rows:
        return "  (none)"
    return (
        "\n".join(rows)
        + "\n\n  NOTE: these holdings had only a composition pass — no market/"
        "news/sentiment analysis and no Head/Researcher deep chain (they fell "
        "below the focus cutoff). A low focus_score means the Junior PA saw no "
        "composition problem with the holding — it is NOT a sell signal. Absent "
        "a grounded reason, HOLD (leave weight unchanged); if one genuinely "
        "needs a deep read first, add it to `uncertain_holdings`."
    )


def _main_ticker_line(state: dict) -> str:
    """One line describing the candidate, or a pure-rebalance notice."""
    main = state.get("main_ticker", "")
    if main:
        return f"MAIN TICKER (candidate add): {main}\n"
    # No main ticker → pure rebalance. But every ticker in CURRENT WEIGHTS is a
    # LISTED holding the PM may move, INCLUDING those at 0.00 weight — adding to a
    # 0-weight listed holding deploys cash into it and is NOT "adding a new
    # ticker". (A cash-heavy book whose other sleeves sit at 0 is exactly the
    # deploy-the-cash case; refusing to fund them just leaves dead cash.)
    has_zero_weight = any(
        w <= 1e-9 and t != "CASH" for t, w in state.get("current_weights", {}).items()
    )
    note = (
        "MAIN TICKER: none. This is a PURE REBALANCE of the LISTED holdings "
        "— do not invent a ticker that is not in CURRENT WEIGHTS, but you MAY "
        "move weight among ANY listed position and CASH. A listed holding at "
        "0.00 weight is still a fundable position: use `add` to deploy CASH into "
        "it (that is not a 'new ticker').\n"
    )
    if has_zero_weight:
        note += (
            "  NOTE: some listed holdings currently sit at 0.00 weight. If the "
            "book is heavy in CASH, your job is to DEPLOY that cash into the "
            "listed sleeves per the OBJECTIVE — do not leave it parked at 100% "
            "cash simply because the target sleeves start at zero.\n"
        )
    return note


def create_portfolio_manager(deep_llm: BaseChatModel, cadence_factor: float = 0.1):
    """Return a pair (run_initial, run_uncertainty_pass).

    cadence_factor (Phase 1e): the fraction of each proposed move executed per
    rebalance (partial adjustment). Default 0.1 → ~10 rebalances to complete a
    full reallocation. 1.0 = full move (today's behavior). The PM's prose and
    change list are unchanged; only the applied delta magnitude is damped, in
    code (LLM-judges / code-does-math).
    """
    system_text = load_prompt("portfolio_manager_system.txt")

    def _invoke(human_text: str, tag: str) -> dict:
        resp = deep_llm.invoke(
            [
                SystemMessage(content=system_text),
                HumanMessage(content=human_text),
            ],
            config={"tags": [tag]},
        )
        return extract_json_block(resp.content)

    def _build_decision(state: dict, parsed: dict) -> dict:
        """Turn the LLM's change-list into the downstream final-decision shape."""
        proposed = parsed.get("changes", []) or []
        # Phase 1e: damp the full-conviction proposal by cadence_factor BEFORE
        # applying it. The reported `changes` are the DAMPED (executed) deltas so
        # the rationale section reflects what actually moved this rebalance; the
        # PM's full proposal is preserved under `proposed_changes` for audit.
        changes = scale_changes(proposed, cadence_factor)
        target, rationale, warnings = apply_changes(
            state["current_weights"], changes, state.get("main_ticker", "")
        )
        for w in warnings:
            print(f"[portfolio] PM change warning: {w}")
        return {
            "target_weights":         target,
            "per_position_rationale": rationale,
            "narrative":              str(parsed.get("narrative", "")),
            "uncertain_holdings":     parsed.get("uncertain_holdings", []) or [],
            "changes":                changes,
            "proposed_changes":       proposed,
            "cadence_factor":         cadence_factor,
        }

    def run_initial(state: dict) -> dict:
        human_text = (
            f"OBJECTIVE:\n{state['objective']}\n\n"
            f"CURRENT WEIGHTS:\n{_format_weights(state['current_weights'])}\n\n"
            f"{_main_ticker_line(state)}\n"
            f"{_junior_pa_block(state)}\n\n"
            f"PER-TICKER ANALYSIS (Head assessment + Bull/Bear research + raw "
            f"tool output):\n\n"
            f"{_format_research_section(state['per_ticker_reports'], state['selected_tickers'])}\n\n"
            f"NON-FOCUS HOLDINGS (analysed but no deep research; ticker  weight):\n"
            f"{_non_prioritized_block(state)}\n\n"
            "Form your OWN independent rebalance view from this evidence. Engage "
            "the structural concerns the Junior PA raised — do not paper over an "
            "overlap/tilt/fit problem with a uniform skim. Produce your list of "
            "CHANGES as the JSON object specified; omit any position you are "
            "holding unchanged. If a non-focus holding genuinely needs deep "
            "research before you can decide, list it in `uncertain_holdings`."
        )
        return _build_decision(state, _invoke(human_text, "portfolio_manager:initial"))

    def run_uncertainty_pass(state: dict, uncertainty_reports: dict[str, str]) -> dict:
        prior = state.get("pm_initial", {})
        prior_weights = prior.get("target_weights", {})
        prior_narrative = prior.get("narrative", "")
        new_block = "\n\n".join(
            f"### {t}\n\n{rep.strip() or '_(no report)_'}"
            for t, rep in uncertainty_reports.items()
        )
        human_text = (
            f"OBJECTIVE:\n{state['objective']}\n\n"
            f"CURRENT WEIGHTS:\n{_format_weights(state['current_weights'])}\n\n"
            f"{_main_ticker_line(state)}\n"
            f"{_junior_pa_block(state)}\n\n"
            f"PRIOR (initial) TARGET WEIGHTS:\n{_format_weights(prior_weights)}\n\n"
            f"PRIOR NARRATIVE:\n{prior_narrative}\n\n"
            f"PER-TICKER ANALYSIS (Head assessment + Bull/Bear research + raw "
            f"tool output):\n\n"
            f"{_format_research_section(state['per_ticker_reports'], state['selected_tickers'])}\n\n"
            f"NEW RESEARCHER REPORTS (previously uncertain holdings):\n\n"
            f"{new_block}\n\n"
            "Now produce your FINAL list of CHANGES against the CURRENT WEIGHTS "
            "(not the prior target). Reconcile the new evidence with your prior "
            "decision. Only move a position where the research gives you strong, "
            "specific conviction; omit the rest to hold them. An exit requires an "
            "explicit sell thesis. `uncertain_holdings` MUST be an empty list."
        )
        decision = _build_decision(state, _invoke(human_text, "portfolio_manager:uncertainty"))
        decision["uncertain_holdings"] = []
        return decision

    return run_initial, run_uncertainty_pass
