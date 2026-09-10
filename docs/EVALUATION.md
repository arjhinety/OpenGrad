# Evaluation Subsystem Specification

**Building in Public.** Evaluation is decoupled from model training and executed across standard benchmark suites.

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
