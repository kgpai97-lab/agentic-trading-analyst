"""Structured analyst output — the numeric contract (Step 7, Phase 1a).

Until now every analyst emitted only prose markdown. Nothing downstream could
optimise, size, or *evaluate* on top of that, because nothing was numeric until
the Portfolio Manager's final delta JSON. This module introduces the first
machine-readable analyst output: a directional `AnalystSignal` carrying a list
of individually falsifiable `Claim`s.

Scope is the **four analysts only** (market / news / sentiment / fourth-slot).
Structuring the Head Analyst / Researcher was deliberately deferred — they keep
their prose + `extract_json_block`. Revisit if a later phase needs to evaluate
them.

Two design points that the rest of the roadmap leans on:

  * `key_evidence` (the claims) is kept SEPARATE from `prose`. The Phase-5 claim
    checker scores each claim independently against realised market data, so each
    claim must stand alone as a falsifiable unit — not be buried in a paragraph.
  * The signal is produced by a cheap *second* LLM call that reads the prose and
    emits this schema (see `agents/analysts/_common.py`), so Call 1's reasoning
    quality is never compromised by a structured-output constraint.

Quant analogy: `AnalystSignal.direction` is a cross-sectional alpha score in
[-1, +1]; `confidence` is its (inverse) variance proxy that Phase 6's
inverse-variance ensemble will weight by; each `Claim` is a pre-registered,
datable, testable hypothesis whose realised hit-rate becomes the reliability
prior.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator

# Allowed vocabularies. Kept permissive (str fields validated against these
# sets) rather than Enums so the extraction LLM's output is easy to coerce and a
# slightly-off value degrades to a validation error we can retry, not a crash.
_DIRECTIONS = {"positive", "negative", "neutral"}
_HORIZONS = {"1W", "2W", "1M", "3M", "6M", "1Y"}

# Synonyms the extraction LLM reaches for that map cleanly onto the canonical
# three. The reasoning-first model often writes "mixed"/"bullish"/"bearish"
# (or qualified forms like "neutral-to-negative") instead of the bare enum;
# coercing these salvages the whole extraction rather than dropping the claim.
_DIRECTION_SYNONYMS = {
    "mixed": "neutral",
    "flat": "neutral",
    "balanced": "neutral",
    "bullish": "positive",
    "bull": "positive",
    "up": "positive",
    "bearish": "negative",
    "bear": "negative",
    "down": "negative",
}


def _coerce_direction(v: str) -> str:
    """Map a free-text direction onto {positive, negative, neutral}.

    Handles exact synonyms ("mixed" → neutral) and qualified forms
    ("neutral-to-negative", "positive (risk)") by scanning for the first
    canonical word present. Returns the lowercased value if already canonical,
    else raises ValueError so the caller can retry / drop the claim.
    """
    raw = (v or "").strip().lower()
    if raw in _DIRECTIONS:
        return raw
    if raw in _DIRECTION_SYNONYMS:
        return _DIRECTION_SYNONYMS[raw]
    # Qualified forms: pick the first canonical/synonym token that appears.
    # "neutral-to-negative" → neutral (first wins); "positive (risk)" → positive.
    import re  # noqa: PLC0415

    for token in re.split(r"[^a-z]+", raw):
        if token in _DIRECTIONS:
            return token
        if token in _DIRECTION_SYNONYMS:
            return _DIRECTION_SYNONYMS[token]
    raise ValueError(
        f"direction must be one of {sorted(_DIRECTIONS)}, got {v!r}"
    )


class Claim(BaseModel):
    """A single falsifiable statement an analyst made — scored later (Phase 5).

    The fields up to `made_on` are filled at emission time by the extraction
    call. `realized_outcome` / `correct` start None and are filled by the
    Phase-5 claim checker once `horizon` has elapsed and realised data exists.
    """

    statement: str            # human-readable falsifiable claim
    ticker: str
    metric: str               # "return" | "vol" | "earnings" | ...
    direction: str            # "positive" | "negative" | "neutral"
    horizon: str              # "1W" | "1M" | "3M" | ...
    made_on: date

    # Filled in later by the claim checker (Phase 5); None until the horizon
    # elapses and realised data is fetched via DataContext.
    realized_outcome: float | None = None
    correct: bool | None = None

    @field_validator("ticker", "metric", "statement")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()

    @field_validator("direction")
    @classmethod
    def _known_direction(cls, v: str) -> str:
        return _coerce_direction(v)

    @field_validator("horizon")
    @classmethod
    def _known_horizon(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in _HORIZONS:
            raise ValueError(f"horizon must be one of {sorted(_HORIZONS)}, got {v!r}")
        return v


class AnalystSignal(BaseModel):
    """One analyst's distilled directional view + its supporting claims.

    `direction` and `confidence` are the numeric handles the downstream
    ensemble (Phase 6) and PM-input layer consume; `key_evidence` is the list of
    claims the Phase-5 checker scores; `prose` is the unchanged markdown report
    so nothing in the human-facing output is lost.
    """

    direction: float = Field(..., ge=-1.0, le=1.0)   # -1 bearish .. +1 bullish
    confidence: float = Field(..., ge=0.0, le=1.0)    # 0 .. 1
    horizon: str
    key_evidence: list[Claim] = Field(default_factory=list)
    prose: str = ""                                    # the existing report, verbatim

    @field_validator("horizon")
    @classmethod
    def _known_horizon(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in _HORIZONS:
            raise ValueError(f"horizon must be one of {sorted(_HORIZONS)}, got {v!r}")
        return v


class _ClaimExtract(BaseModel):
    """Extraction-only view of a Claim — what the LLM is asked to fill.

    Deliberately omits `made_on` (stamped in code from the run's trade_date — we
    never let the model invent a date) and the Phase-5 checker fields
    (`realized_outcome` / `correct`, filled later). The extraction helper maps
    each `_ClaimExtract` into a full `Claim` by adding `made_on`. Keeping the
    LLM-facing schema minimal reduces the structured-decoding burden on the
    reasoning-first gpt-oss model (fewer required fields = fewer failure modes).
    """

    statement: str
    ticker: str
    metric: str
    direction: str
    horizon: str

    @field_validator("direction")
    @classmethod
    def _coerce_dir(cls, v: str) -> str:
        return _coerce_direction(v)

    @field_validator("horizon")
    @classmethod
    def _coerce_horizon(cls, v: str) -> str:
        v = (v or "1M").strip().upper()
        return v if v in _HORIZONS else "1M"


class AnalystSignalExtract(BaseModel):
    """Extraction-only view of an AnalystSignal — the schema Call 2 fills.

    Omits `prose` (the helper attaches the verbatim Call-1 report afterward) and
    uses `_ClaimExtract` for evidence (no `made_on`). The helper converts this
    into a full `AnalystSignal`.
    """

    direction: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    horizon: str = "1M"
    key_evidence: list[_ClaimExtract] = Field(default_factory=list)

    @field_validator("horizon")
    @classmethod
    def _coerce_horizon(cls, v: str) -> str:
        v = (v or "1M").strip().upper()
        return v if v in _HORIZONS else "1M"


def _merge_signals(
    left: dict[str, AnalystSignal] | None,
    right: dict[str, AnalystSignal] | None,
) -> dict[str, AnalystSignal]:
    """Reducer for AgentState.signals: shallow-merge right into left.

    Mirrors `_merge_tool_outputs` in agent_states.py. Each analyst's clear node
    returns {its_name: AnalystSignal}; without a merge reducer LangGraph would
    last-write-wins the whole dict and only the final analyst's signal would
    survive. Merging keeps every analyst's signal.
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged
