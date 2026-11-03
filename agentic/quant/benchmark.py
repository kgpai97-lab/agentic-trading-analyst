"""GARCH vol-targeting benchmark — a PARALLEL yardstick, never an overlay.

Phase 4a (Decision: build GARCH as a benchmark first, overlay deferred to 4b).
This strategy does NOT touch the live portfolio. It produces its OWN competing
weight series so the scoreboard (4c) can measure whether the LLM desk's risk
management actually beats a dumb classical vol-target rule. A benchmark cannot
whipsaw the real book, so it is strictly safe and strictly informative.

Mechanics (all per the locked Phase-1e cadence decisions):
  1. Per-asset GARCH/GJR vol forecast (BIC-selected) on each held asset's return
     history (forecast INPUT is the asset return series, so the rebalance cadence
     does not degrade the lookback).
  2. Portfolio vol = √(wᵀ Σ w), where Σ = D R D, D = diag(forecast vols),
     R = a SHRUNK sample correlation matrix (Ledoit-Wolf-style shrinkage toward
     identity — stabilises a noisy sample R on short histories).
  3. √t horizon scaling: forecast vol and the target are put on the SAME horizon
     as the rebalance cadence (weekly → daily×√5 forecast, annual_target/√52
     target). `rebalance_period` is a first-class parameter.
  4. exposure_scalar = target_vol(horizon) / forecast_vol(horizon), then a
     10-step PARTIAL ADJUSTMENT toward it (same speed as the PM's cadence_factor)
     so the comparison is fair and the scalar doesn't whipsaw.
  5. final weights = base_weights × exposure_scalar in the risky sleeve, the
     remainder to CASH, then the SAME enforce_constraints() with the SAME
     PortfolioConstraints as the PM (apples-to-apples).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentic.dataflows.constraints import PortfolioConstraints, enforce_constraints
from agentic.quant.garch import GarchFit, select_garch

# Trading periods per year by cadence — drives √t horizon scaling.
_PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}
_DAYS_PER_PERIOD = {"daily": 1, "weekly": 5, "monthly": 21}


def _shrink_correlation(returns_matrix: np.ndarray, shrink: float = 0.2) -> np.ndarray:
    """Sample correlation shrunk toward the identity by `shrink` ∈ [0,1].

    R_shrunk = (1-shrink)·R_sample + shrink·I. A simple, robust stand-in for
    full Ledoit-Wolf: on the short histories a weekly book sees, the raw sample
    correlation is noisy and can make wᵀΣw unstable; pulling toward I regularises
    it. shrink=0 → raw sample R; shrink=1 → assume uncorrelated.
    """
    if returns_matrix.shape[1] < 2:
        return np.array([[1.0]])
    if returns_matrix.shape[0] < 3:
        # Too few aligned observations to estimate correlation — assume
        # independence (identity). Avoids a degenerate/NaN corrcoef warning.
        return np.eye(returns_matrix.shape[1])
    R = np.corrcoef(returns_matrix, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    I = np.eye(R.shape[0])
    return (1.0 - shrink) * R + shrink * I


@dataclass(frozen=True)
class BenchmarkDecision:
    """Output of one benchmark rebalance step."""

    target_weights: dict          # final weights (post enforce_constraints)
    exposure_scalar: float        # applied gross-exposure scalar (after damping)
    raw_exposure_scalar: float    # target_vol/forecast_vol BEFORE damping
    forecast_vol: float           # portfolio vol forecast at the cadence horizon
    target_vol: float             # vol target at the cadence horizon
    per_asset: dict               # ticker -> {"model","vol_daily","persistence"}
    rationale: str                # two-line attribution (Decision A style)


def _annualised_to_horizon(annual_target: float, period: str) -> float:
    """Scale an annual vol target to the cadence horizon via √t."""
    ppy = _PERIODS_PER_YEAR.get(period, 52)
    return annual_target / np.sqrt(ppy)


class GarchVolBenchmark:
    """Vol-targeting benchmark over a fixed asset set.

    Construct once with the cadence + target; call `step()` each rebalance with
    the current base weights, the per-asset return histories, the prior exposure
    scalar (for partial adjustment), and the same constraints as the PM.
    """

    def __init__(
        self,
        annual_vol_target: float = 0.10,
        rebalance_period: str = "weekly",
        cadence_factor: float = 0.1,
        corr_shrink: float = 0.2,
    ) -> None:
        self.annual_vol_target = annual_vol_target
        self.rebalance_period = rebalance_period
        self.cadence_factor = cadence_factor
        self.corr_shrink = corr_shrink

    # ---- vol forecast -----------------------------------------------------

    def _forecast_portfolio_vol(
        self, base_weights: dict, returns: dict[str, np.ndarray]
    ) -> tuple[float, dict]:
        """Portfolio vol at the cadence horizon + per-asset diagnostics.

        Only NON-cash assets with sufficient history are modelled. Weights are
        renormalised over the modelled risky sleeve so the vol reflects the
        risky book (cash has zero vol).
        """
        period_days = _DAYS_PER_PERIOD.get(self.rebalance_period, 5)
        risky = {t: w for t, w in base_weights.items()
                 if t.upper() != "CASH" and w > 0 and t in returns}
        if not risky:
            return 0.0, {}

        tickers = list(risky)
        horizon_vol_map = {}
        per_asset = {}
        cols = []
        min_len = min(len(returns[t]) for t in tickers)
        for t in tickers:
            r = np.asarray(returns[t], dtype=float)
            r = r[np.isfinite(r)]
            # GARCH needs a decent history; below that, fall back to sample vol
            # (a too-short fit warns and is unreliable). This keeps early
            # rebalances — when little history has accrued — clean and honest.
            if r.size >= 30:
                try:
                    fit: GarchFit = select_garch(r)
                    # Term-structure-CORRECT multi-day vol: the cumulative √(Σσ²)
                    # over `period_days`, NOT daily×√t — under GARCH days 2..h
                    # revert toward the unconditional vol, so √t scaling of a
                    # 1-day forecast over/under-states the week. See
                    # GarchFit.forecast_vol_horizon.
                    hv = fit.forecast_vol_horizon(period_days)
                    per_asset[t] = {"model": fit.model,
                                    "vol_daily": fit.forecast_vol(1),
                                    "vol_horizon": hv,
                                    "persistence": fit.persistence}
                except Exception:  # noqa: BLE001 — fall back to sample vol
                    dv = float(np.std(r)) if r.size > 1 else 0.0
                    hv = dv * np.sqrt(period_days)   # constant sample vol IS √t-scalable
                    per_asset[t] = {"model": "sample", "vol_daily": dv,
                                    "vol_horizon": hv, "persistence": 0.0}
            else:
                dv = float(np.std(r)) if r.size > 1 else 0.0
                hv = dv * np.sqrt(period_days)
                per_asset[t] = {"model": "sample", "vol_daily": dv,
                                "vol_horizon": hv, "persistence": 0.0}
            horizon_vol_map[t] = hv
            cols.append(returns[t][-min_len:] if min_len > 0 else returns[t])

        # Per-asset horizon vols (cumulative GARCH forecast, or √t-scaled sample).
        horizon_vols = np.array([horizon_vol_map[t] for t in tickers])

        # Correlation from aligned recent returns, shrunk for stability.
        R = _shrink_correlation(np.column_stack(cols), self.corr_shrink) \
            if len(tickers) > 1 else np.array([[1.0]])
        D = np.diag(horizon_vols)
        Sigma = D @ R @ D

        w = np.array([risky[t] for t in tickers])
        w = w / w.sum()                            # weights within the risky sleeve
        port_var = float(w @ Sigma @ w)
        return float(np.sqrt(max(port_var, 0.0))), per_asset

    # ---- one rebalance ----------------------------------------------------

    def step(
        self,
        base_weights: dict,
        returns: dict[str, np.ndarray],
        ticker_types: dict[str, str],
        constraints: PortfolioConstraints | None = None,
        prior_scalar: float = 1.0,
    ) -> BenchmarkDecision:
        """One vol-target rebalance. Returns a BenchmarkDecision.

        `base_weights` are the strategy's current risky mix (the benchmark holds
        a FIXED cross-sectional mix — it only scales gross exposure, per Decision
        A's separation). `prior_scalar` is last step's applied scalar (for the
        10-step partial adjustment). `returns` maps ticker -> daily decimal
        return history.
        """
        constraints = constraints or PortfolioConstraints()
        target_vol = _annualised_to_horizon(self.annual_vol_target, self.rebalance_period)

        forecast_vol, per_asset = self._forecast_portfolio_vol(base_weights, returns)
        if forecast_vol <= 1e-9:
            raw_scalar = 1.0
        else:
            raw_scalar = target_vol / forecast_vol

        # Ceiling = 1/risky_total, i.e. "fully invested, zero cash" — the exact
        # point past which scaling up the risky sleeve would require negative
        # cash (leverage). This lets the scalar rise ABOVE 1.0 to deploy idle
        # cash back toward the vol target when under-risked, while a flat cap
        # at 1.0 (the prior behaviour) only ever let this de-risk and would get
        # stuck there even when the forecast was well below target.
        risky = {t: w for t, w in base_weights.items() if t.upper() != "CASH"}
        risky_total = sum(risky.values())
        max_scalar = (1.0 / risky_total) if risky_total > 1e-9 else 1.0
        raw_scalar = float(np.clip(raw_scalar, 0.0, max_scalar))

        # 10-step partial adjustment toward the target scalar (same speed as PM).
        applied = prior_scalar + self.cadence_factor * (raw_scalar - prior_scalar)
        applied = float(np.clip(applied, 0.0, max_scalar))

        # Scale the risky sleeve by `applied`; the remainder goes to CASH.
        scaled = {t: w * applied for t, w in risky.items()}
        cash_weight = max(1.0 - sum(scaled.values()), 0.0)
        weights = dict(scaled)
        weights["CASH"] = weights.get("CASH", 0.0) + cash_weight

        # SAME hard constraints as the PM — comparison is apples-to-apples.
        types = dict(ticker_types)
        types.setdefault("CASH", "cash")
        final = enforce_constraints(weights, types, constraints)
        final = {t: round(w, 6) for t, w in final.items()}

        rationale = (
            f"GARCH benchmark: forecast {self.rebalance_period} vol "
            f"{forecast_vol:.4f} vs target {target_vol:.4f} → raw scalar "
            f"{raw_scalar:.3f}; applied {applied:.3f} after 10-step damping "
            f"(prior {prior_scalar:.3f})."
        )
        return BenchmarkDecision(
            target_weights=final, exposure_scalar=applied,
            raw_exposure_scalar=raw_scalar, forecast_vol=forecast_vol,
            target_vol=target_vol, per_asset=per_asset, rationale=rationale,
        )
