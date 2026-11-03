"""ToolResult — typed return for all dataflow vendor functions.

Why this exists: tool wrappers in agent_utils.py used to return a string on
both success and failure. The LLM had no way to distinguish "real data" from
"No price history available for ZZZZ." apart from prose, and analyst prompts
gave no fallback guidance for missing data.

ToolResult makes the success/failure split explicit. The vendor layer emits
one. The @tool wrapper renders it to a tagged string ([OK]/[EMPTY]/[ERROR])
that the LLM sees. Analyst prompts have a matching "Tool result handling"
block that tells the model what to do for each tag.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    payload: str = ""
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def to_llm_string(self) -> str:
        if self.ok:
            return f"[OK]\n{self.payload}"
        prefix = "[EMPTY]" if self.meta.get("empty") else "[ERROR]"
        body = self.error or self.payload or "no data"
        return f"{prefix} {body}"

    @classmethod
    def success(cls, payload: str, **meta) -> "ToolResult":
        return cls(ok=True, payload=payload, meta=dict(meta))

    @classmethod
    def empty(cls, reason: str, **meta) -> "ToolResult":
        return cls(ok=False, error=reason, meta={"empty": True, **meta})

    @classmethod
    def failure(cls, error: str, **meta) -> "ToolResult":
        return cls(ok=False, error=error, meta=dict(meta))
