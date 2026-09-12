# Benchmarking

Benchmark evaluation is decoupled from model training. Every benchmark, evaluator, commit, split,
parser, generation setting, and trial count is pinned before a comparison, and a benchmark is a
measurement rather than a product requirement.

## Current measurement status

- **The behavioral held-out has executed scores.** When2Call is the frozen set B0 and every
  post-training checkpoint were scored on: 3,652 distinct items, 3,650 scored after two
  quarantines, engine vLLM 0.29.0 on an A100. This is the only benchmark with real OpenGrad scores.
- **No external benchmark score exists.** Every *external* family in the registry is
  `FROZEN_NOT_EXECUTED`. Five families have run artifacts under `reports/benchmarks/`
  (`bfcl-v4`, `acebench`, `tau3`, `ifeval`, `performance-microsuite`), but each was produced by the
  deterministic mock backend — their `environment.json` records `"backend": "mock"` — so they
  validate the local result contract, not model capability.

Keeping the behavioral suite separate from the external ones is deliberate: the behavioral set
answers "did tool policy change", and the external sets would answer "did anything else break".

## Inventory and counting convention

`registry/benchmarks.yaml` holds **23 identifiers**, and a count is only meaningful when the subset
is named:

| Subset | Count | Notes |
|---|---:|---|
| External benchmarks in Tiers A–E | **17** | The primary external suite. All `FROZEN_NOT_EXECUTED`. |
| Executed behavioral held-out | 1 | `when2call-eval` — the only benchmark with real scores. |
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
| When2Call | Call decision, answer quality | **Executed** — 3,650 held-out examples, engine vLLM 0.29.0 | **Yes** — B0 and 12 candidate checkpoints | Frozen in the baseline v2 config; training corpora exclude it by construction |
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
