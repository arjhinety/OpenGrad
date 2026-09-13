# Evaluation Subsystem Specification

**Building in Public.** Evaluation is decoupled from model training and executed across standard benchmark suites.

---

## 0. Methodology lessons from the capability-regression campaign

Four findings that changed how evaluation is done here. Each cost real money to learn; each is now
a standing requirement.

### 0.1 An evaluation set can only detect what it contains

The frozen tool-policy partition held `tool_call`, `request_for_info` and `cannot_answer` examples
and **no ANSWER examples**. It therefore could not detect a checkpoint that stopped answering
ordinary questions — and did not. Coverage of response modes is now a promotion requirement; see
[`PROMOTION_POLICY.md` §2.5](PROMOTION_POLICY.md).

### 0.2 Never collapse "wrong" and "did not try"

Report `accuracy`, `answer_rate` and `accuracy_given_answer = correct / attempted` **separately**.
A checkpoint whose accuracy falls while conditional accuracy holds has not lost the ability; it has
stopped using it. Collapsing them hides exactly the distinction that matters. Where nothing was
attempted, `accuracy_given_answer` is `null` — never `0.0`, which would read as "tried everything
and failed".

### 0.3 A fixed generation budget is a confound when it binds unevenly

MMLU-Pro at 768 tokens showed Base 38.0% / M0 36.8% — apparently no regression. Base had hit the
cap on **37.6%** of items against M0's 7.2%, so the measurement reflected verbosity, not accuracy.
At 2048 tokens the observed gap is **12.0pp**, but Base still truncates on 21.8% of items against
M0's 6.3%, so under the rule below it is reported as an interval: **+6.4pp to +33.1pp** under
truncation-adversarial resolution (observed +12.0pp).

Requirements:

- Record `finish_reason` and report the truncation rate **per stage**, always.
- A materially uneven truncation rate invalidates the comparison; fix the budget and re-run.
- Where residual truncation remains, report a **truncation-adversarial interval** rather than a
  point estimate: with *unresolved* = `incorrect ∧ truncated`, true accuracy lies in
  `[correct/n, (correct+unresolved)/n]`. Do **not** call an observed difference a "lower bound"
  unless a bound has actually been derived.
- Changing a budget after inspecting only `finish_reason` counts, before any response is scored, is
  a **pre-scoring protocol amendment**. Apply it uniformly to every stage, and record it.

### 0.4 Aggregate scores hide behavioural change

On the 7-case OpenWeights sentinel, Base and the promoted checkpoint both scored **5/7** — Base by
answering `trap-arithmetic` and `multi-step-change` *wrongly*, the promoted checkpoint by passing
`trap-arithmetic` and *refusing* `multi-step-change` and `format-constraint`. Identical score,
different failures and opposite behaviour. Always inspect failure modes, not just totals, and never let a 7-case suite carry a
capability conclusion.

### 0.5 Scorers must be deterministic, and you have to check

The vendored upstream IFEval checkers are non-deterministic out of the box: one fixed generations
file scored 0.4510 / 0.4492 / 0.4492 across three runs, via `langdetect.detect()` and `random.*` in
`build_description`. Seed every stochastic dependency in the scorer wrapper — not in the vendored
source, which must stay byte-identical to upstream — and verify by scoring the same file more than
once.

---

## 1. Benchmark Suites

Defined in `configs/benchmark_suites/`:
- **`smoke.yaml`**: Fast sanity check across Tier A, B, and E (BFCL, tau3, ACEBench, IFEval, Microsuite).
- **`tool_use_core.yaml`**: Core capability benchmarks (BFCL V4, tau3, ACEBench).
- **`regression_core.yaml`**: General capability preservation (IFBench, IFEval, LiveBench, MMLU-Pro, GSM8K, ARC).
- **`agent_transfer.yaml`**: Ecosystem transfer (MCPMark, AgentBench FC, Terminal-Bench, TUA-Bench, GAIA).
- **`full_post_training.yaml`**: Comprehensive matrix across all tiers.
- **`speculative_decoding.yaml`**: Systems performance, quality parity, and Pareto analysis.

---

## 2. Evaluation Commands

Run a benchmark suite:
```bash
opengrad evaluate checkpoints/checkpoint-5 --suite tool_use_core --dry-run
```

Run a specific benchmark:
```bash
opengrad benchmark run --benchmark bfcl_v4 --dry-run
```

Compare two evaluation runs:
```bash
opengrad compare runs/m0_baseline/eval runs/m1_candidate/eval
```
Emits formatted markdown deltas for capability accuracy, category breakdowns, and systems metrics (TTFT, throughput, speedup).

---

## 3. Baseline Execution Gate

The frozen B0 definition is `configs/evaluation/tool_calling/qwen35_2b_baseline.yaml` and points to
the held-out v2 manifest. Before using a GPU, run the dry path:

```text
opengrad baseline --dry-run
```

It loads the held-out manifest, renders the exact Qwen prompt, calls
`InferenceBackend.generate()`, parses native Qwen tool calls, evaluates routing, and writes
predictions, metrics, a residual profile, and an environment capture.

The deterministic backend is a plumbing test only; it **must never be reported as a model score**.
Real scores require an accelerator and are recorded under `reports/baselines/` and
`runs/<experiment_id>/eval/`.

## 4. Benchmark Inventory

The registry, tier list, counting convention, and harness status consulted when choosing a suite
live in [benchmarks/README.md](benchmarks/README.md).
