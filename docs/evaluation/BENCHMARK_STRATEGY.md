# OpenGrad Post-Training Benchmark Strategy

> "Every gradient is a hypothesis. Every checkpoint is evidence."

## 1. Executive Philosophy

OpenGrad rejects collapsing model evaluation into a single opaque headline score:
```text
benchmark_accuracy = good model   <-- REJECTED
```

Small open-weight models (e.g., Qwen3.5-2B, SmolLM, Llama-3.2-1B/3B) exhibit severe trade-offs across capability boundaries:
- A checkpoint with higher tool syntax recall may regress on general instruction following.
- A model with aggressive parallel tool invocation may over-call in ambiguous or non-tool settings.
- Speculative decoding speedup is meaningless if it degrades tool call argument accuracy or corrupts JSON payloads.

Therefore, OpenGrad reports and stores metrics across **four independent axes**:
1. **Capability**: Tool selection, argument grounding, multiple calls, parallel calls, multi-turn state preservation, clarification seeking, safe refusals.
2. **Agent Behavior**: Step efficiency, tool execution loops, recovery from environment failure, policy compliance, unnecessary calls.
3. **System Performance**: Time to first token (TTFT), inter-token latency (p50/p95 ITL), throughput (tokens/sec), peak VRAM.
4. **Speculative Decoding / MTP**: Acceptance rate, accepted tokens per verifier step, effective depth, speedup, and strict quality parity.

---

## 2. Benchmark Priority Tiers

OpenGrad organizes evaluations into five explicit tiers:

### Tier A — Primary Tool-Use Research
- **BFCL V4** (*Berkeley Function Calling Leaderboard V4*): Evaluates simple, multiple, parallel, parallel-multiple, relevance, irrelevance, multi-turn, agentic, web-search, and memory categories.
- **tau3-bench**: Multi-turn agent benchmarking across realistic enterprise text domains (airline, retail, telecom, banking). Tracks task completion, policy compliance, conversation length, and invalid actions.
- **ACEBench**: Evaluates the decision boundary: normal function use, ambiguous/special/incomplete/impossible requests, clarification behavior, and when *not* to call.

### Tier B — General Capability & Regression
- **IFBench**: Fine-grained instruction following with strict and loose constraint verification.
- **IFEval**: Deterministic instruction-following regression testing.
- **LiveBench**: Monthly refreshed, contamination-minimized benchmarks across reasoning, math, coding, language, and data analysis.
- **MMLU-Pro**: Hard general capability retention check with 10 options per question.
- **GSM8K & ARC-Challenge**: Secondary regression checks for elementary math reasoning and scientific QA.

### Tier C — Agent Transfer
- **MCPMark / MCP-Universe**: Evaluates agent interaction within real Model Context Protocol (MCP) ecosystems (filesystem, GitHub, PostgreSQL, browser/Playwright, Notion).
- **AgentBench FC**: Function-calling agent benchmarks across operating system, database, web, and knowledge graph environments.
- **Terminal-Bench**: Command-line interface and terminal task completion. Full harness configuration is recorded.
- **TUA-Bench**: Terminal-use and agent interaction evaluation with full harness state capture.

### Tier D — Stretch
- **GAIA**: Complex multi-modal, tool-assisted reasoning. Used primarily to measure **base model vs. trained model delta** rather than absolute performance for ~2B parameters.

### Tier E — Systems & Speculative Decoding
- **Performance Microsuite**: 10 fixed, deterministic prompts with frozen SHA-256 hashes representing distinct text and tool workload profiles.
- **Speculative Replay**: Controlled replay of benchmark subsets under standard autoregressive (AR) vs. native MTP / speculative draft decoding.

---

## 3. Model-Only vs. Agent-System Metrics

It is crucial to distinguish what is measured:
- **Model-Only Metrics**: Token sequence parity, next-token accuracy, single-turn tool selection, argument syntax validity, zero-shot direct refusal.
- **Agent-System Metrics**: End-to-end task completion, loop termination, environment recovery rate, total conversation turns.
In benchmarks like Terminal-Bench, MCPMark, and tau3, performance is a joint function of:
```text
System Performance = Model + Prompt Template + Tool Interface + Agent Loop + Runtime Sandbox
```
OpenGrad run manifests record the complete harness configuration to prevent invalid cross-harness comparisons.

---

## 4. Benchmark Contamination Policy

Before freezing any training corpus or claiming generalization, OpenGrad requires contamination screening across five progressively stronger levels:
1. **Level 1**: Exact canonical conversation SHA-256 hash.
2. **Level 2**: Normalized prompt / answer whitespace and casefold hash.
3. **Level 3**: 5-gram Jaccard / MinHash near-duplicate detection.
4. **Level 4**: Semantic similarity candidate generation (SequenceMatcher / embedding similarity).
5. **Level 5**: Manual audit queue for suspicious matches.

> **Rule:** OpenGrad never silently deletes benchmark examples. Matches are queued for human audit and recorded in `reports/data/benchmark_contamination_registry.json`.

---

## 5. Post-Training Evaluation Workflow

The intended research progression:

```text
Untouched Base Model (e.g. Qwen3.5-2B)
       ↓  (Run full benchmark suite)
Immutable Baseline Artifacts Saved
       ↓
Train M0 (Targeted SFT)
       ↓  (Run identical benchmark suite & settings)
Compare M0 vs. Base (Generate Deltas)
       ↓
Train M1 (Preference / Residual Correction)
       ↓  (Run identical benchmark suite & settings)
Compare M1 vs. Base + M0
       ↓
Add Native MTP Heads / Speculative Engine
       ↓  (Run AR decoding vs. MTP/Speculative decoding)
Evaluate Quality Parity + Systems Speedup + Pareto Frontier
```

---

## 6. Research Evidence Standard

Never claim:
```text
"M1 is better."
```
Instead, provide multi-dimensional empirical evidence:
```text
M1 improves BFCL V4 parallel tool calling by +14.2 points,
improves tau3 retail task completion by +8.1 points,
preserves IFBench within -0.3,
preserves MMLU-Pro within -0.2,
but increases unnecessary-tool behavior by +1.4 points.
```
