"""UsageCallback — per-analyst token, latency, and tool-call accounting.

Why this exists: factory.py historically built ChatOllama / ChatOpenAI
instances with no callbacks attached, and trading_graph.py never passed
callbacks into graph.invoke(). We had zero visibility into per-analyst
token spend or where wall-clock time went. That blocks ask #6 (profile a
run) and downstream Step 7 backtest sizing.

This callback hooks LangChain's BaseCallbackHandler and buckets per-call
usage by the first item in the invocation's `tags` list. The analyst nodes
in _common.py set `config={"tags": [node_name]}` on each .invoke(), so each
LLM call lands in the correct bucket. Anything without a tag (the briefing
synthesis call in trading_graph._compose_briefing, if it ever becomes an
LLM call) falls into the "unattributed" bucket.

Token sources, in priority order:
  1. AIMessage.usage_metadata (input_tokens, output_tokens) — populated by
     langchain-ollama on .invoke() responses since 0.1.x.
  2. LLMResult.llm_output["token_usage"] — older path used by some providers.
  3. response_metadata.eval_count / prompt_eval_count — Ollama-specific
     fallback when usage_metadata is absent.

Transcript capture: in addition to the numeric buckets, every LLM call's raw
input (the system + human messages each agent reads) and raw output (what the
agent produced) is recorded as a transcript entry, stamped with the bucket
name, an ISO wall-clock timestamp, elapsed seconds, and token counts. Call
write_transcript(path) after a run to persist the full per-agent audit trail
(JSONL by default, or a human-readable .md). This answers "what did each agent
see and say, how long did it take, and how many tokens" across runs.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler


def _empty_bucket() -> dict:
    return {
        "in_tokens":   0,
        "out_tokens":  0,
        "llm_calls":   0,
        "tool_calls":  0,
        "llm_seconds": 0.0,
        "tool_seconds": 0.0,
    }


class UsageCallback(BaseCallbackHandler):
    """Buckets per-analyst LLM and tool usage. Pass-once per TradingGraph instance."""

    raise_error = False

    def __init__(self) -> None:
        self.buckets: dict[str, dict] = defaultdict(_empty_bucket)
        # run_id → (bucket_name, start_time)
        self._starts: dict[UUID, tuple[str, float]] = {}
        # run_id → (raw input text, ISO timestamp) pending until on_llm_end
        self._pending_input: dict[UUID, tuple[str, str]] = {}
        # Ordered list of per-LLM-call transcript entries. Each entry:
        #   {bucket, timestamp, elapsed_s, in_tokens, out_tokens, input, output}
        self.transcript: list[dict] = []

    def reset(self) -> None:
        self.buckets.clear()
        self._starts.clear()
        self._pending_input.clear()
        self.transcript.clear()

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _messages_to_text(messages: Any) -> str:
        """Flatten the message list(s) handed to on_chat_model_start into text.

        LangChain passes a list-of-lists (one inner list per generation). Each
        message renders as "<ROLE>: <content>" so the transcript shows exactly
        what the agent read (system prompt + human task + any tool results).
        """
        lines: list[str] = []
        try:
            for msg_list in messages or []:
                # messages may be a flat list (on_llm_start passes prompt strings)
                inner = msg_list if isinstance(msg_list, list) else [msg_list]
                for m in inner:
                    role = getattr(m, "type", None) or type(m).__name__
                    content = getattr(m, "content", m)
                    lines.append(f"[{role}] {content}")
        except Exception:  # noqa: BLE001
            return str(messages)
        return "\n\n".join(lines)

    @staticmethod
    def _response_to_text(response: Any) -> str:
        """Extract the generated text from an LLMResult."""
        parts: list[str] = []
        try:
            for gen_list in getattr(response, "generations", []) or []:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg is not None and getattr(msg, "content", None):
                        parts.append(str(msg.content))
                    elif getattr(gen, "text", None):
                        parts.append(str(gen.text))
        except Exception:  # noqa: BLE001
            return str(response)
        return "\n".join(parts)

    @staticmethod
    def _bucket_from_tags(tags: list[str] | None) -> str:
        if tags:
            for t in tags:
                # Skip LangChain auto-tags like "seq:step:1" — they aren't analysts.
                if t and not t.startswith("seq:"):
                    return t
        return "unattributed"

    @staticmethod
    def _extract_token_counts(response: Any) -> tuple[int, int]:
        """Return (input_tokens, output_tokens) from an LLMResult."""
        in_tok = 0
        out_tok = 0

        # 1. AIMessage.usage_metadata on each generation
        try:
            for gen_list in getattr(response, "generations", []) or []:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    um = getattr(msg, "usage_metadata", None) if msg is not None else None
                    if um:
                        in_tok += int(um.get("input_tokens", 0) or 0)
                        out_tok += int(um.get("output_tokens", 0) or 0)
        except Exception:  # noqa: BLE001
            pass

        if in_tok or out_tok:
            return in_tok, out_tok

        # 2. llm_output.token_usage (OpenAI-style)
        llm_output = getattr(response, "llm_output", None) or {}
        tu = (llm_output or {}).get("token_usage") or {}
        in_tok = int(tu.get("prompt_tokens", 0) or 0)
        out_tok = int(tu.get("completion_tokens", 0) or 0)
        if in_tok or out_tok:
            return in_tok, out_tok

        # 3. Ollama eval_count fallback on response_metadata
        try:
            for gen_list in getattr(response, "generations", []) or []:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    rm = getattr(msg, "response_metadata", None) if msg is not None else {}
                    rm = rm or {}
                    in_tok += int(rm.get("prompt_eval_count", 0) or 0)
                    out_tok += int(rm.get("eval_count", 0) or 0)
        except Exception:  # noqa: BLE001
            pass

        return in_tok, out_tok

    # ── LLM hooks ──────────────────────────────────────────────────────────

    def on_llm_start(
        self, serialized, prompts, *, run_id, parent_run_id=None,
        tags=None, metadata=None, **kwargs,
    ):
        self._starts[run_id] = (self._bucket_from_tags(tags), time.perf_counter())
        self._pending_input[run_id] = (self._messages_to_text(prompts), self._now_iso())

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None,
        tags=None, metadata=None, **kwargs,
    ):
        self._starts[run_id] = (self._bucket_from_tags(tags), time.perf_counter())
        self._pending_input[run_id] = (self._messages_to_text(messages), self._now_iso())

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        bucket_name, t0 = self._starts.pop(run_id, ("unattributed", time.perf_counter()))
        elapsed = time.perf_counter() - t0
        bucket = self.buckets[bucket_name]
        bucket["llm_calls"] += 1
        bucket["llm_seconds"] += elapsed
        in_tok, out_tok = self._extract_token_counts(response)
        bucket["in_tokens"] += in_tok
        bucket["out_tokens"] += out_tok

        raw_input, started_at = self._pending_input.pop(run_id, ("", self._now_iso()))
        self.transcript.append({
            "bucket":     bucket_name,
            "timestamp":  started_at,
            "elapsed_s":  round(elapsed, 3),
            "in_tokens":  in_tok,
            "out_tokens": out_tok,
            "input":      raw_input,
            "output":     self._response_to_text(response),
        })

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        bucket_name, t0 = self._starts.pop(run_id, ("unattributed", time.perf_counter()))
        elapsed = time.perf_counter() - t0
        bucket = self.buckets[bucket_name]
        bucket["llm_calls"] += 1
        bucket["llm_seconds"] += elapsed

        raw_input, started_at = self._pending_input.pop(run_id, ("", self._now_iso()))
        self.transcript.append({
            "bucket":     bucket_name,
            "timestamp":  started_at,
            "elapsed_s":  round(elapsed, 3),
            "in_tokens":  0,
            "out_tokens": 0,
            "input":      raw_input,
            "output":     f"[ERROR] {error}",
        })

    # ── Tool hooks ─────────────────────────────────────────────────────────

    def on_tool_start(
        self, serialized, input_str, *, run_id, parent_run_id=None,
        tags=None, metadata=None, **kwargs,
    ):
        # Tool tags propagate from the parent LLM invocation in LangGraph.
        self._starts[run_id] = (self._bucket_from_tags(tags), time.perf_counter())

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        bucket_name, t0 = self._starts.pop(run_id, ("unattributed", time.perf_counter()))
        bucket = self.buckets[bucket_name]
        bucket["tool_calls"] += 1
        bucket["tool_seconds"] += time.perf_counter() - t0

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        bucket_name, t0 = self._starts.pop(run_id, ("unattributed", time.perf_counter()))
        bucket = self.buckets[bucket_name]
        bucket["tool_calls"] += 1
        bucket["tool_seconds"] += time.perf_counter() - t0

    # ── Reporting ──────────────────────────────────────────────────────────

    def totals(self) -> dict:
        out = _empty_bucket()
        for b in self.buckets.values():
            for k, v in b.items():
                out[k] += v
        return out

    def report(self) -> str:
        """Markdown table of usage by bucket, plus a TOTAL row."""
        rows = sorted(self.buckets.items(), key=lambda kv: kv[0])
        total = self.totals()

        header = (
            f"{'bucket':<24} {'in_tok':>8} {'out_tok':>8} "
            f"{'llm':>5} {'tool':>5} {'llm_s':>8} {'tool_s':>8}"
        )
        lines = ["", "Usage by analyst:", header, "-" * len(header)]
        for name, b in rows:
            lines.append(
                f"{name:<24} {b['in_tokens']:>8} {b['out_tokens']:>8} "
                f"{b['llm_calls']:>5} {b['tool_calls']:>5} "
                f"{b['llm_seconds']:>8.2f} {b['tool_seconds']:>8.2f}"
            )
        lines.append("-" * len(header))
        lines.append(
            f"{'TOTAL':<24} {total['in_tokens']:>8} {total['out_tokens']:>8} "
            f"{total['llm_calls']:>5} {total['tool_calls']:>5} "
            f"{total['llm_seconds']:>8.2f} {total['tool_seconds']:>8.2f}"
        )

        if total["in_tokens"] == 0 and total["out_tokens"] == 0:
            lines.append(
                "(note: token counts are zero — the provider did not populate "
                "usage_metadata, llm_output.token_usage, or eval_count.)"
            )
        return "\n".join(lines)

    # ── Transcript persistence ─────────────────────────────────────────────

    def write_transcript(self, path: str | Path, meta: dict | None = None) -> Path:
        """Persist the full per-agent transcript for this run.

        Each LLM call is one record: which agent (bucket), when it started,
        how long it took, token counts, the raw input it read, and the raw
        output it produced. Extension drives the format:
          • .jsonl → one JSON object per call (machine-readable, default)
          • .md    → human-readable sections, one per call
        `meta` (ticker, date, etc.) is recorded as a header / leading record.

        Returns the resolved path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = meta or {}

        if path.suffix == ".md":
            blocks = [f"# Run transcript\n"]
            if meta:
                blocks.append(
                    "\n".join(f"- **{k}**: {v}" for k, v in meta.items()) + "\n"
                )
            blocks.append(self.report() + "\n")
            for i, e in enumerate(self.transcript, 1):
                blocks.append(
                    f"\n---\n\n## {i}. {e['bucket']}  ·  {e['timestamp']}\n\n"
                    f"- elapsed: {e['elapsed_s']}s  ·  in_tokens: {e['in_tokens']}  "
                    f"·  out_tokens: {e['out_tokens']}\n\n"
                    f"### Input (what the agent read)\n\n```\n{e['input']}\n```\n\n"
                    f"### Output (what the agent produced)\n\n```\n{e['output']}\n```"
                )
            path.write_text("\n".join(blocks), encoding="utf-8")
        else:
            with path.open("w", encoding="utf-8") as fh:
                if meta:
                    fh.write(json.dumps({"_meta": meta, "totals": self.totals()}) + "\n")
                for e in self.transcript:
                    fh.write(json.dumps(e, ensure_ascii=False) + "\n")

        return path
