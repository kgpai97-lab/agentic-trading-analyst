"""Shared helpers for portfolio-layer agents.

These nodes are tool-less: they consume already-produced reports and call
the LLM exactly once. The helpers here:
  - load_prompt(): same convention as analysts/_common.py, but exposed
    here so portfolio modules don't reach into analyst internals.
  - extract_json_block(): tolerant JSON parser that strips ```json fences
    and trailing prose, used by Junior PA and Portfolio Manager whose
    contracts are JSON-structured.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"


def load_prompt(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_block(text: str) -> dict:
    """Pull a JSON object out of an LLM response.

    Accepts: raw JSON, ```json fenced``` JSON, or JSON with leading/trailing
    prose. Raises ValueError if no JSON object can be parsed.
    """
    if text is None:
        raise ValueError("empty LLM response")

    candidates: list[str] = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1))
    candidates.append(text)

    for raw in candidates:
        raw = raw.strip()
        # Find the first { ... } that parses cleanly.
        start = raw.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(raw)):
                c = raw[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = raw[start : i + 1]
                        try:
                            return json.loads(chunk)
                        except json.JSONDecodeError:
                            break
            start = raw.find("{", start + 1)

    raise ValueError(f"no JSON object found in LLM response:\n{text[:500]}")
