# LLM Selection & Usage Guide for Agentic Trading Analyst

## How LLMs Are Used

This project runs a **multi-agent financial analysis system** on local Ollama inference. The architecture uses a **two-tier LLM strategy**:

### Tier 1: Quick-Think Analyst Nodes (llama3.2:3b)
- **Role**: Sentiment, Market, News, Fundamentals/Composition/Fixed-Income analysts
- **Usage Pattern**: Tool-calling loop — analyst examines state, decides if more data is needed, calls tools (get_news, get_stock_data, get_fundamentals, etc.), processes results, repeats
- **Output**: Structured prose report (written to state field like `sentiment_report`)
- **Key Requirement**: Must reliably emit `tool_calls` with function names and arguments; tool loop cannot work if the LLM skips this field
- **Latency Impact**: Looped execution — slower models compound across multiple tool roundtrips per analyst; current design has 4–5 analysts per ticker

### Tier 2: Deep-Think / Synthesis Nodes (deepseek-r1:8b, deferred)
- **Role**: Planned for Step 6 portfolio nodes (Portfolio Manager, Researcher)
- **Usage Pattern**: Single `llm.invoke()` call — no tools, pure reasoning + structured JSON extraction
- **Output**: `tool_calls`-free; JSON object with deltas/rationale embedded in response
- **Key Requirement**: Structured reasoning without hallucinating omitted holdings; arithmetic correctness (sums must close)
- **Latency Cost**: Deep-thinking models emit very long `<think>` blocks before answering; Ollama's context/predict config must accommodate this or responses get truncated to empty strings

### Portfolio Layer (Step 6, currently on Quick tier)
- **Junior PA**: Judges urgency, code does priority arithmetic (avoids small-model math bugs)
- **Head Analyst**: Consolidates four analyst reports, flags inconsistencies
- **Researcher**: Produces bull/bear synthesis from four reports
- **Portfolio Manager**: Emits JSON delta changes (not full weight vector) to guarantee mass conservation

All Step 6 roles are tool-less `llm.invoke()` calls; only the analyst tier uses tool loops.

---

## Most Desirable LLM Qualities (Priority Order)

### 1. **Reliable Structured Output & Tool Calling**
   - **Why it matters most**: The entire pipeline depends on tool_calls. If the LLM doesn't emit them, the analyst stalls
   - **Test**: Can the model reliably output JSON and function calls in the same response?
   - **Known failure**: llama3.2:3b frequently skips `tool_calls` field, causing analysts to hang or produce ungrounded prose
   - **Desirable**: Models trained on function-calling datasets (e.g., Claude, GPT-4, models fine-tuned on tool-calling synthetics)

### 2. **No Hallucination (Accuracy over Eloquence)**
   - **Why it matters**: Financial analysis must be grounded in provided data. Fabricated thesis/rationale is worse than silence
   - **Test**: When a tool result is `[EMPTY]`, does the analyst admit ignorance or invent an analysis?
   - **Known failure**: Small models on Step 6 PM emitted exit rationales not supported by Researcher reports
   - **Desirable**: Models with strong instruction-following and guardrails against creative filling-in

### 3. **Arithmetic Correctness**
   - **Why it matters**: Junior PA needs to compute priority scores; Portfolio Manager must balance deltas to sum ~1.0
   - **Test**: Can it sum weights, compute 0.3*A + 0.7*B, and produce a consistent vector?
   - **Known failure**: llama3.2:3b produced round-number scores that didn't sum and contradicted its own selection
   - **Workaround in place**: Arithmetic moved to code (Junior PA only judges, Python does the math); Step 6 PM contract changed to deltas (apply_changes() enforces conservation)
   - **Desirable**: If using the LLM for numeric tasks, prefer models with quantitative training (math benchmarks > 70%)

### 4. **Inference Speed (Local Execution)**
   - **Why it matters**: Analysts run 4–5 tool loops per ticker; slow models compound latency
   - **Test**: Can it produce a full response in < 10 seconds on local hardware?
   - **Known issue**: deepseek-r1:8b's `<think>` block is so long (3,000–5,000 tokens) that Ollama with `num_ctx=8192` truncates output; even with `deep_num_ctx=32768` and `deep_num_predict=16384`, responses are 30–60 seconds
   - **Current choice**: Quick tier locked at llama3.2:3b (3-4 seconds per response)
   - **Desirable**: 3B–7B parameter range; inference < 10s per analyst node on consumer hardware (CPU or RTX 4090-equivalent)

### 5. **JSON Extraction from Prose**
   - **Why it matters**: Analyst reports are prose; Step 6 roles must emit JSON (rationale dicts, change lists)
   - **Test**: Can it embed a clean JSON block inside prose, and can a simple regex or parser extract it?
   - **Known pattern**: `extract_json_block()` in `agentic/agents/portfolio/_common.py` looks for ` ```json ... ``` `
   - **Desirable**: Models that respect markdown code fences and produce valid, non-nested JSON in the first block

### 6. **Instruction Following & Long Context**
   - **Why it matters**: Analyst prompts are detailed (100–200 tokens); portfolio manager prompt includes full holdings list + non-prioritized block
   - **Test**: Does it read and adhere to complex multi-part instructions (e.g., "emit urgency_score for each, then compute sum, then select top-k")?
   - **Known issue**: Step 6 PM on 3b model skipped non-prioritized holdings entirely instead of considering them
   - **Desirable**: Models with strong context window (8k+) and instruction-following benchmarks > 80%

### 7. **Deterministic Output** (optional, for backtesting)
   - **Why it matters**: Future Step 7 backtest will replay the pipeline over historical dates; non-deterministic LLM output breaks reproducibility
   - **Test**: Given the same input, does it produce identical outputs (or very close)?
   - **Known limitation**: yfinance news is not point-in-time; tool outputs can drift over time
   - **Desirable**: Models with low temperature (0.0–0.3) in production; future: snapshot-cache articles by date

---

## Current Configuration & Trade-offs

### Quick-Think Tier
- **Model**: llama3.2:3b-instruct
- **Pros**: Fast (3–4s), low VRAM
- **Cons**: Weak tool calling, poor arithmetic, small context
- **Status**: Functional but brittle; relies on error handling + structured tool definitions

### Deep-Think Tier (Deferred)
- **Model**: deepseek-r1:8b
- **Intended Role**: Step 6 Portfolio Manager & Researcher (promoted from quick tier when latency resolved)
- **Blocker**: `<think>` chain causes Ollama truncation unless `num_ctx` ≫ response tokens
- **Workaround**: Increased `deep_num_ctx` (32768) and `deep_num_predict` (16384) in `DEFAULT_CONFIG`
- **Latency**: 30–60s per response; unfeasible for analyst tool loops
- **Unblocked**: The structural fixes (PM delta model, Junior PA arithmetic-in-code) made quick-tier acceptable

### Hypothetical Upgrade Path
- **Option A (Local)**: Switch quick tier to llama2-7b or mistral-7b (better tool calling, same latency class)
- **Option B (Hybrid)**: Keep quick-think local; use cloud API (Claude, GPT-4) for deep-think only
- **Option C (Cloud only)**: Replace Ollama with Claude API + structured outputs mode (requires auth, breaks offline requirement)

---

## Minimal LLM Specification

For this project to work, the LLM must satisfy:

| Criterion | Minimum | Ideal |
|-----------|---------|-------|
| **Tool Calling** | Emits `tool_calls` field reliably | > 95% call accuracy on multi-tool tasks |
| **Context Window** | 4,096 tokens | 8,192+ tokens |
| **Speed (local)** | < 30s per response | 2–6s per response |
| **JSON Extraction** | Produces valid JSON blocks | Markdown fence code blocks, no nested JSON |
| **Hallucination Rate** | < 5% of responses fabricate data | Admits uncertainty when data is missing |
| **Parameter Count** | < 13B (for local inference on consumer GPU) | 3B–7B (best speed/quality trade-off) |
| **Instruction Following** | Passes 70% of MMLU | Passes 80% of MMLU + instruction-following benchmarks |

---

## When to Revisit This Choice

- **Switch quick-think model**: If analysts produce > 20% ungrounded prose, or tool calling fails > 5% of the time
- **Promote deep-think to Step 6**: Once inference time < 15s per Portfolio Manager call (requires faster hardware or cloud API)
- **Add a third tier**: If Step 7 policy node needs very high reasoning quality (e.g., backtest shows significant alpha opportunity being missed)
- **Go cloud**: If offline requirement is lifted, Claude 4.x with structured outputs + batch API is the safest architectural choice

