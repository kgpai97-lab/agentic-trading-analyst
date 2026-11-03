"""GARCH(1,1) and GJR-GARCH(1,1) volatility models — from-scratch MLE.

Why from scratch (not the `arch` package): this is a Financial-Math learning
project; a hand-rolled Gaussian quasi-MLE fit demonstrates the mechanics
(conditional-variance recursion, the log-likelihood, stationarity constraints)
rather than hiding them behind `arch_model(...).fit()`. Pure numpy/scipy.

Why TWO models, selected by BIC (not just GARCH(1,1)): GARCH(1,1) is the
canonical parsimonious baseline (Bollerslev 1986) and Hansen & Lunde (2005)
found nothing beats it for FX — but for EQUITIES the *leverage effect* (negative
returns raise future vol more than positive ones) is a robust stylized fact, and
asymmetric models capture it. **GJR-GARCH** (Glosten–Jagannathan–Runkle 1993)
adds one term — γ·ε²·𝟙[ε<0] — for exactly that. Our forward book is equity-heavy,
so rather than *assume* a model we fit BOTH per asset and select by BIC. That
turns "why GARCH(1,1)?" into a measured result: GJR should win on equities.

Models (returns in PERCENT — ε scaled ×100 for numerical conditioning):
    GARCH(1,1):  σ²ₜ = ω + α·ε²ₜ₋₁                + β·σ²ₜ₋₁
    GJR(1,1):    σ²ₜ = ω + α·ε²ₜ₋₁ + γ·ε²ₜ₋₁·𝟙[εₜ₋₁<0] + β·σ²ₜ₋₁

Stationarity (finite unconditional variance):
    GARCH: α + β < 1
    GJR:   α + β + γ/2 < 1     (½ = E[𝟙[ε<0]] under a symmetric innovation)

We fit on **mean-zero** percent residuals (we subtract the sample mean; daily
means are negligible vs. vol but it keeps ε an innovation). Gaussian QMLE: even
if returns are fat-tailed, the Gaussian-likelihood estimator is consistent for
the variance dynamics, which is all the vol-target benchmark needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

_SCALE = 100.0          # work in percent returns for conditioning
_TINY = 1e-8            # variance floor to avoid log(0)


@dataclass(frozen=True)
class GarchFit:
    """Result of a GARCH(p,q)/GJR fit.

    params:    dict of estimated coefficients (in percent-return units):
               omega, alpha (=α₁ for back-compat), alpha_vec (α₁..α_q),
               beta (=β₁), beta_vec (β₁..βₚ), and gamma for GJR(1,1).
    model:     "garch" | "gjr".
    p, q:      GARCH order — p lagged-variance (β) terms, q lagged-shock (α) terms.
    loglik:    maximised Gaussian log-likelihood.
    aic, bic:  information criteria (lower = better).
    n:         sample size used.
    sigma2:    fitted conditional variance series (percent²), length n.
    converged: optimiser success flag.
    """

    params: dict
    model: str
    loglik: float
    aic: float
    bic: float
    n: int
    sigma2: np.ndarray
    converged: bool
    p: int = 1
    q: int = 1

    @property
    def order(self) -> tuple[int, int]:
        return (self.p, self.q)

    @property
    def persistence(self) -> float:
        """Σα + Σβ (+ γ/2 for GJR) — closeness to 1 = vol-shock persistence.

        The unconditional variance is finite iff persistence < 1 (stationarity).
        """
        pa = self.params
        base = sum(pa.get("alpha_vec", [pa.get("alpha", 0.0)])) \
            + sum(pa.get("beta_vec", [pa.get("beta", 0.0)]))
        if self.model == "gjr":
            base += 0.5 * pa.get("gamma", 0.0)
        return base

    def forecast_var(self, horizon: int = 1) -> float:
        """SINGLE-period variance forecast for day t+h (percent²).

        Iterates the variance recursion forward using E[ε²ₜ]=σ²ₜ (and the ½σ²
        symmetry term for GJR), which decays toward the unconditional variance
        ω/(1−persistence). For p,q>1 we use the reduced one-term recursion in the
        *effective persistence* (Σα+Σβ), the standard multi-step approximation.

        NOTE: this is the variance OF day t+h alone, not the cumulative variance
        of t+1..t+h. For a multi-day (e.g. weekly) horizon use
        `cumulative_forecast_var` / `forecast_vol_horizon` instead.
        """
        omega = self.params["omega"]
        persist = self.persistence
        var_h = float(self.sigma2[-1])
        for _ in range(max(1, horizon)):
            var_h = omega + persist * var_h
        return var_h

    def forecast_vol(self, horizon: int = 1) -> float:
        """SINGLE-period (day t+h) volatility forecast in DECIMAL units."""
        return float(np.sqrt(max(self.forecast_var(horizon), _TINY)) / _SCALE)

    def cumulative_forecast_var(self, periods: int = 1) -> float:
        """Cumulative variance of the next `periods` days: Σ_{h=1..periods} σ²_{t+h}.

        This is the term-structure-CORRECT multi-day variance under GARCH: sum
        the consecutive one-day-ahead conditional variances, each of which
        reverts toward the unconditional level. It is NOT σ²_{t+1}·periods (the
        naive √t rule), which assumes vol is flat over the window and overstates
        a multi-day forecast when current vol is above its long-run mean (and
        understates it when below). The two coincide only when persistence→1
        (no mean reversion) or current vol = unconditional vol.
        """
        omega = self.params["omega"]
        persist = self.persistence
        var_h = float(self.sigma2[-1])
        total = 0.0
        for _ in range(max(1, periods)):
            var_h = omega + persist * var_h     # σ² for the next day
            total += var_h
        return total

    def forecast_vol_horizon(self, periods: int = 1) -> float:
        """Cumulative `periods`-day volatility forecast in DECIMAL units.

        √(Σ_{h=1..periods} σ²_{t+h}) — the proper weekly (periods=5) vol for the
        vol-target benchmark. periods=1 reduces to `forecast_vol(1)`.
        """
        return float(np.sqrt(max(self.cumulative_forecast_var(periods), _TINY)) / _SCALE)


def _unpack(theta: np.ndarray, p: int, q: int, model: str):
    """Split a flat param vector into (omega, alphas[q], gamma, betas[p]).

    Layout: [omega, α₁..α_q, (gamma if gjr), β₁..βₚ]. GJR is only defined for
    (1,1) here (one leverage term), which is the standard asymmetric extension.
    """
    i = 0
    omega = theta[i]; i += 1
    alphas = theta[i:i + q]; i += q
    if model == "gjr":
        gamma = theta[i]; i += 1
    else:
        gamma = 0.0
    betas = theta[i:i + p]
    return omega, alphas, gamma, betas


def _filter_variance_pq(eps: np.ndarray, omega: float, alphas, gamma: float,
                        betas, model: str) -> np.ndarray:
    """Conditional-variance recursion for general GARCH(p,q) (+ GJR leverage).

        σ²ₜ = ω + Σ_{j=1..q} αⱼ ε²ₜ₋ⱼ  [+ γ ε²ₜ₋₁ 𝟙(εₜ₋₁<0)]  + Σ_{i=1..p} βᵢ σ²ₜ₋ᵢ

    Pre-sample ε² and σ² are backcast at the sample variance (standard).
    """
    n = eps.size
    q = len(alphas)
    p = len(betas)
    base_var = max(float(np.var(eps)), _TINY)
    s2 = np.full(n, base_var)
    e2 = eps ** 2
    for t in range(n):
        v = omega
        for j in range(q):
            v += alphas[j] * (e2[t - 1 - j] if t - 1 - j >= 0 else base_var)
        if model == "gjr" and t - 1 >= 0:
            v += gamma * e2[t - 1] * (eps[t - 1] < 0)
        for i in range(p):
            v += betas[i] * (s2[t - 1 - i] if t - 1 - i >= 0 else base_var)
        s2[t] = v if v > _TINY else _TINY
    return s2


def _neg_loglik_pq(theta: np.ndarray, eps: np.ndarray, p: int, q: int,
                   model: str) -> float:
    """Negative Gaussian log-likelihood for GARCH(p,q)/GJR."""
    omega, alphas, gamma, betas = _unpack(theta, p, q, model)
    if omega <= 0 or np.any(alphas < 0) or np.any(betas < 0) or gamma < 0:
        return 1e10
    # Stationarity: Σα + Σβ (+ γ/2) < 1 keeps the variance process finite.
    persist = float(np.sum(alphas) + np.sum(betas) + (0.5 * gamma if model == "gjr" else 0.0))
    if persist >= 1.0:
        return 1e10
    s2 = _filter_variance_pq(eps, omega, alphas, gamma, betas, model)
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + eps ** 2 / s2)
    return -ll if np.isfinite(ll) else 1e10


def _fit_pq(eps: np.ndarray, p: int, q: int, model: str) -> GarchFit:
    n = eps.size
    var = float(np.var(eps))

    # Start: small ω, modest α mass spread over q lags, most persistence in β₁.
    x0 = [0.05 * var] + [0.06 / q] * q
    bounds = [(1e-6, None)] + [(0.0, 1.0)] * q
    if model == "gjr":
        x0 += [0.04]; bounds += [(0.0, 1.0)]
    x0 += [0.90 / p] * p; bounds += [(0.0, 1.0)] * p
    x0 = np.asarray(x0)

    res = minimize(_neg_loglik_pq, x0, args=(eps, p, q, model), method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": 1000, "ftol": 1e-9})

    omega, alphas, gamma, betas = _unpack(res.x, p, q, model)
    params = {
        "omega": float(omega),
        "alpha_vec": [float(a) for a in alphas],
        "beta_vec": [float(b) for b in betas],
        "alpha": float(alphas[0]),   # back-compat scalars (first lag)
        "beta": float(betas[0]),
    }
    if model == "gjr":
        params["gamma"] = float(gamma)
    k = len(res.x)                   # number of free parameters
    loglik = -float(res.fun)
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik
    sigma2 = _filter_variance_pq(eps, omega, alphas, gamma, betas, model)
    return GarchFit(params=params, model=model, loglik=loglik, aic=aic, bic=bic,
                    n=n, sigma2=sigma2, converged=bool(res.success), p=p, q=q)


def fit_garch(returns: np.ndarray, model: str = "garch",
              p: int = 1, q: int = 1) -> GarchFit:
    """Fit a GARCH(p,q) (model="garch") or GJR-GARCH(1,1) (model="gjr") model.

    GJR is only defined here for (1,1) (one leverage term). `returns` is a 1-D
    array of DECIMAL returns; the fit is on mean-zero percent residuals.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 30:
        raise ValueError(f"need >=30 returns to fit GARCH, got {r.size}")
    if model == "gjr" and (p, q) != (1, 1):
        raise ValueError("GJR is only supported at order (1,1)")
    if p < 1 or q < 1:
        raise ValueError("p and q must each be >= 1")
    eps = (r - r.mean()) * _SCALE      # mean-zero, in percent
    return _fit_pq(eps, p, q, model)


def select_garch(returns: np.ndarray, models=("garch", "gjr")) -> GarchFit:
    """Fit each model and return the one with the lowest BIC.

    BIC (not AIC) penalises the extra GJR parameter more, so GJR is selected
    only when the leverage term genuinely improves fit — the principled answer
    to "why this model?". Non-converged fits are skipped; if all fail we fall
    back to the best-effort GARCH fit.
    """
    fits = []
    for m in models:
        try:
            f = fit_garch(returns, m)
            if f.converged:
                fits.append(f)
        except Exception:  # noqa: BLE001
            continue
    if not fits:
        # last resort: return a (possibly non-converged) GARCH fit
        return fit_garch(returns, "garch")
    return min(fits, key=lambda f: f.bic)


def order_search(
    returns: np.ndarray,
    max_p: int = 2,
    max_q: int = 2,
    include_gjr: bool = True,
    criterion: str = "bic",
) -> list[GarchFit]:
    """Fit GARCH(p,q) over the grid 1..max_p × 1..max_q (+ GJR(1,1)),
    and return ALL converged fits sorted best-first by `criterion` ("bic"|"aic").

    This is the (p,q) order-selection: rather than ASSUME GARCH(1,1), we search.
    BIC is the default selector — its ln(n)·k penalty is heavier than AIC's 2k,
    so it favours parsimony, which is why daily equity vol almost always lands on
    (1,1): the extra lags fit in-sample noise but BIC won't pay for them. The
    full list lets a diagnostic show the whole ranking (the evidence), not just
    the winner.
    """
    key = (lambda f: f.bic) if criterion == "bic" else (lambda f: f.aic)
    fits: list[GarchFit] = []
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                f = fit_garch(returns, "garch", p=p, q=q)
                if f.converged and np.isfinite(f.bic):
                    fits.append(f)
            except Exception:  # noqa: BLE001
                continue
    if include_gjr:
        try:
            g = fit_garch(returns, "gjr", p=1, q=1)
            if g.converged and np.isfinite(g.bic):
                fits.append(g)
        except Exception:  # noqa: BLE001
            pass
    if not fits:
        return [fit_garch(returns, "garch")]
    return sorted(fits, key=key)


def select_order(returns: np.ndarray, max_p: int = 2, max_q: int = 2,
                 include_gjr: bool = True, criterion: str = "bic") -> GarchFit:
    """Return the single best (p,q)[/GJR] fit by `criterion` (BIC default)."""
    return order_search(returns, max_p, max_q, include_gjr, criterion)[0]
