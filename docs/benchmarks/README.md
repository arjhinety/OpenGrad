# Benchmarking

Benchmark evaluation is decoupled from model training. Every benchmark, evaluator, commit, split,
parser, generation setting, and trial count is pinned before a comparison, and a benchmark is a
measurement rather than a product requirement.

## Current measurement status

- **The behavioral held-out has executed scores.** When2Call is the frozen set B0 and every
  post-training checkpoint were scored on: 3,652 distinct items, 3,650 scored after two
  quarantines, engine vLLM 0.29.0 on an A100.
- **Three Tier B benchmarks have real scores: IFEval, GSM8K and MMLU-Pro.** They ran outside the
  registry harness, in the H200 capability-diagnosis campaign over Base → M0 → M1-v2, using the
  upstream scorers (vendored IFEval checkers, deterministic GSM8K numeric extraction, upstream
  MMLU-Pro answer regexes). Results: [`results/benchmarks/h200/capability_v1/`](../../results/benchmarks/h200/capability_v1/);
  verdict: [`results/final_campaign_verdict.json`](../../results/final_campaign_verdict.json).
  Their registry entries still read `FROZEN_NOT_EXECUTED` because that status describes the
  registry harness path, which has not run them.
- **The other 14 external benchmarks have no score.** Five families have run artifacts under
  `reports/benchmarks/` (`bfcl-v4`, `acebench`, `tau3`, `ifeval`, `performance-microsuite`), but
  each was produced by the deterministic mock backend — their `environment.json` records
  `"backend": "mock"` — so they validate the local result contract, not model capability.

Keeping the behavioral suite separate from the external ones is deliberate: the behavioral set
answers "did tool policy change", and the external sets would answer "did anything else break".

## Inventory and counting convention

`registry/benchmarks.yaml` holds **23 identifiers**, and a count is only meaningful when the subset
is named:

| Subset | Count | Notes |
|---|---:|---|
| External benchmarks in Tiers A–E | **17** | The primary external suite. IFEval, GSM8K and MMLU-Pro have real scores from the H200 capability campaign; the registry harness has run none. |
| Executed behavioral held-out | 1 | `when2call-eval` — scored for B0 and every post-training checkpoint. |
| Legacy registration | 1 | `tau-bench-tau2`, kept for its pinned revision; the Tier A `tau3` entry is current. |
| Stretch families awaiting authoritative metadata | 3 | `toolsandbox`, `mcpmark-verified`, `toolathlon`. |
| Internal regression suite | 1 | `internal-no-tool-regression` (public fixtures only; the secret holdout is not materialized here). |

17 + 1 + 1 + 3 + 1 = 23. The suite count is 17; **the registry total is 23**. Do not describe the
program with a single bare number.

## Tiers A–E

- **Tier A (primary tool-use research):** BFCL V4, tau3-bench, ACEBench.
- **Tier B (general and regression):** IFBench, IFEval, LiveBench, MMLU-Pro, GSM8K, ARC-Challenge.
- **Tier C (agent transfer and on-device):** MCPMark, AgentBench FC, Terminal-Bench, TUA-Bench, OpenWeights.
- **Tier D (stretch):** GAIA (multimodal transfer delta).
- **Tier E (systems and speculative):** Performance Microsuite (10 deterministic frozen prompts), Speculative Replay.

Priority tiers, metric axes, contamination policy, and the evidence standard are specified in
[Benchmark Strategy](../evaluation/BENCHMARK_STRATEGY.md).

## Harness status

The table below reflects what has actually run. A mock smoke harness validates the local result
contract with fixture predictions; it is not a model evaluation. When2Call is not a harness — it is
the frozen held-out set.

| Benchmark | Measures in the registry | Harness status | Real score available? | Revision state |
|---|---|---|---|---|
| BFCL V4 | Function-call accuracy | Mock smoke harness | **No** | Recommended Gorilla revision pinned |
| When2Call | Call decision, answer quality | **Executed** — 3,650 held-out examples, engine vLLM 0.29.0 | **Yes** — B0 and 31 candidate checkpoints across 8 runs with committed metrics under `runs/*/eval` (28 excluding the non-reproducible `qwen35_2b_m1_dpo_v1_restore` repeat) | Frozen in the baseline v2 config; training corpora exclude it by construction |
| IFEval | Instruction following (prompt/instruction, strict/loose) | **Executed** in the H200 capability campaign — 541 prompts, vendored upstream checkers | **Yes** — Base, M0, M1-v2 | Upstream revision pinned in `PINS`, [`scripts/prepare_capability_benchmarks.py`](../../scripts/prepare_capability_benchmarks.py) |
| GSM8K | Grade-school math | **Executed** in the H200 capability campaign — 1,319 questions, zero-shot and 8-shot arms | **Yes** — Base, M0, M1-v2 | Upstream revision pinned in `PINS`, [`scripts/prepare_capability_benchmarks.py`](../../scripts/prepare_capability_benchmarks.py) |
| MMLU-Pro | Knowledge and reasoning | **Executed** in the H200 capability campaign — 12,032 items, 5-shot | **Yes** — Base, M0, M1-v2 | Upstream revision pinned in `PINS`, [`scripts/prepare_capability_benchmarks.py`](../../scripts/prepare_capability_benchmarks.py) |
| τ-bench / τ² | Task success, reward | Mock smoke harness | **No** | Recommended repository revision pinned |
| ToolSandbox | Tool-use correctness | Mock smoke harness | **No** | Authoritative metadata pending |
| MCPMark Verified | Task success | Mock smoke harness | **No** | Stretch evaluation; metadata pending |
| Toolathlon | Task success | Mock smoke harness | **No** | Stretch evaluation; metadata pending |

The full machine-readable registry is [`registry/benchmarks.yaml`](../../registry/benchmarks.yaml).
It is the authority for identifiers, tiers, pinned revisions, splits, metrics, and
`prohibit_training` flags.

## Related

- [Benchmark strategy](../evaluation/BENCHMARK_STRATEGY.md) — tiers, metric axes, contamination, workflow
- [Evaluation subsystem](../EVALUATION.md) — suite definitions, commands, and the baseline execution gate
- [Adding a benchmark](../evaluation/ADDING_A_BENCHMARK.md) · [Adding an inference backend](../evaluation/ADDING_A_BACKEND.md)
- [Checkpoint selection](../evaluation/CHECKPOINT_SELECTION.md) · [Checkpoint selection rule](../evaluation/CHECKPOINT_SELECTION_RULE.md)
