"""UsageCallback transcript capture — offline, no LLM or network.

Verifies that every LLM call records a transcript entry carrying the raw
input (what the agent read), raw output (what it produced), token counts,
timestamp, and elapsed time, and that write_transcript() persists it in both
JSONL and Markdown form.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agentic.llm_clients.usage_callback import UsageCallback


def _fake_response(content: str, in_tok: int, out_tok: int):
    ai = AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        },
    )
    gen = SimpleNamespace(message=ai, text=content)
    return SimpleNamespace(generations=[[gen]], llm_output=None)


def _record_call(cb: UsageCallback, bucket: str, system: str, human: str,
                 output: str, in_tok: int, out_tok: int) -> None:
    rid = uuid4()
    cb.on_chat_model_start(
        {}, [[SystemMessage(content=system), HumanMessage(content=human)]],
        run_id=rid, tags=[bucket],
    )
    cb.on_llm_end(_fake_response(output, in_tok, out_tok), run_id=rid)


def test_transcript_captures_io_tokens_and_timing():
    cb = UsageCallback()
    _record_call(cb, "market", "You are the market analyst.",
                 "Analyse NVDA.", "Momentum is strong.", 120, 45)

    assert len(cb.transcript) == 1
    e = cb.transcript[0]
    assert e["bucket"] == "market"
    assert e["in_tokens"] == 120 and e["out_tokens"] == 45
    assert e["elapsed_s"] >= 0.0
    assert e["timestamp"]
    assert "market analyst" in e["input"]      # system prompt the agent read
    assert "Analyse NVDA" in e["input"]         # human task the agent read
    assert "Momentum is strong" in e["output"]  # what the agent produced

    # Bucket totals stay consistent with the transcript.
    assert cb.totals()["in_tokens"] == 120
    assert cb.totals()["out_tokens"] == 45


def test_error_call_still_recorded():
    cb = UsageCallback()
    rid = uuid4()
    cb.on_chat_model_start(
        {}, [[SystemMessage(content="sys"), HumanMessage(content="task")]],
        run_id=rid, tags=["news"],
    )
    cb.on_llm_error(RuntimeError("boom"), run_id=rid)

    assert len(cb.transcript) == 1
    assert cb.transcript[0]["bucket"] == "news"
    assert "boom" in cb.transcript[0]["output"]


def test_reset_clears_transcript():
    cb = UsageCallback()
    _record_call(cb, "market", "s", "h", "o", 1, 1)
    cb.reset()
    assert cb.transcript == []
    assert cb.totals()["in_tokens"] == 0


def test_write_transcript_jsonl(tmp_path):
    cb = UsageCallback()
    _record_call(cb, "market", "sys", "human", "out", 10, 5)
    path = cb.write_transcript(tmp_path / "run.jsonl", meta={"ticker": "NVDA"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # meta header + one call record
    assert json.loads(lines[0])["_meta"]["ticker"] == "NVDA"
    rec = json.loads(lines[1])
    assert rec["bucket"] == "market"
    assert rec["out_tokens"] == 5


def test_write_transcript_markdown(tmp_path):
    cb = UsageCallback()
    _record_call(cb, "market", "sys", "human", "out", 10, 5)
    path = cb.write_transcript(tmp_path / "run.md", meta={"ticker": "NVDA"})

    text = path.read_text(encoding="utf-8")
    assert "what the agent read" in text
    assert "what the agent produced" in text
    assert "market" in text
