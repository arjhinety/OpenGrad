# Speculative Decoding & Multi-Token Prediction (MTP) Benchmark Specification

## 1. Quality Parity Principle

Speculative decoding algorithms (native MTP heads, draft models, DSpark, Medusa/Eagle styles) promise inference latency reduction. However:
```text
QUALITY MUST NEVER BE HIDDEN BY SPEEDUP.
```

If a speculative runtime achieves a 1.8x throughput speedup but suffers a 4.5% drop on BFCL tool argument correctness, the configuration is defective for reliable agent workflows. OpenGrad treats speedup as a constrained optimization problem:
$$\max(\text{Speedup}) \quad \text{subject to} \quad \Delta(\text{Quality}) \ge -\epsilon$$

---

## 2. Supported Decoding Modes

OpenGrad defines five execution modes that share identical prompt templates, stop sequences, and sampling hyperparameters:

| Mode | Identifier | Description |
| :--- | :--- | :--- |
| **Mode A** | `MODE_A_AR` | Standard Autoregressive decoding baseline (exact ground-truth reference). |
| **Mode B** | `MODE_B_MTP` | Native Multi-Token Prediction heads attached directly to the base backbone. |
| **Mode C** | `MODE_C_DRAFT` | Two-model speculative decoding with a smaller distinct draft model. |
| **Mode D** | `MODE_D_DSPARK` | DSpark-style speculative execution and parallel verification. |
| **Mode E** | `MODE_E_EXPERIMENTAL`| Future novel draft/verifier algorithms or tree-based speculation. |

---

## 3. Required Metrics

### Latency
- **TTFT (Time to First Token)**: Cold vs. warm prefill latency in milliseconds.
- **Inter-Token Latency (ITL)**: Mean, p50, and p95 decode step latency.
- **End-to-End Latency**: Total wall-clock time from prompt delivery to EOS.

### Throughput
- **Output Tokens / Second**: Completion tokens generated per second.
- **Total Tokens / Second**: Combined prompt and completion tokens processed per second.
- **Tasks / Second**: Completed benchmark requests per unit time under concurrency.

### Speculative Telemetry
- **Proposed Draft Tokens**: Total candidate tokens drafted by speculative heads.
- **Accepted Tokens**: Number of draft tokens passing verifier acceptance criteria.
- **Acceptance Rate**: $\frac{\text{Accepted Draft Tokens}}{\text{Proposed Draft Tokens}}$.
- **Accepted Tokens / Verifier Step**: Effective tokens produced per forward verification pass:
  $$\tau = 1 + \text{Accepted Draft Tokens per Step}$$
- **Verifier Steps**: Total full-backbone forward verification passes.
- **Rollbacks & Recomputations**: Count of rejected branches and discarded KV states.

### Native MTP Per-Depth Diagnostics
When evaluating native MTP heads (+1, +2, +3, +4):
- **Acceptance by Depth**: Individual acceptance rate at position $+k$.
- **Conditional Acceptance**: Probability of token $+k$ accepted given token $+k-1$ was accepted.
- **Useful Speculation Depth**: The maximum depth index yielding positive net latency reduction (typically where marginal acceptance $\ge 30\%$).
- **Wasted Speculative Computation**: Percentage of drafted positions that were rejected.

---

## 4. Speedup Accounting

Speedup is calculated strictly from wall-clock measurements:
$$\text{Throughput Speedup} = \frac{\text{Speculative Tokens / Sec}}{\text{Autoregressive Tokens / Sec}}$$
$$\text{Latency Speedup} = \frac{\text{Autoregressive Wall Time}}{\text{Speculative Wall Time}}$$

> **Warning:** Never infer speedup from acceptance rate alone. A high acceptance rate with high verification kernel overhead can produce a *slower* system.

---

## 5. Pareto Tradeoff Analysis

For any speculative decoding grid search, OpenGrad computes the Pareto frontier between **Latency/Throughput Speedup** and **Quality Delta ($\Delta$)**:

```text
Configuration A:  1.4x faster,  -0.0 BFCL   --> Pareto Frontier
Configuration B:  1.8x faster,  -0.1 BFCL   --> Pareto Frontier
Configuration C:  1.3x faster,  -0.2 BFCL   --> Pareto Dominated (by A and B)
Configuration D:  2.1x faster,  -4.5 BFCL   --> UNACCEPTABLE (Quality regression)
```

Configuration D is rejected because quality degradation exceeds tolerance. OpenGrad never collapses speed and quality into an arbitrary weighted single score.

---

## 6. Execution Commands

### Run Performance Microsuite under Mock MTP:
```bash
opengrad benchmark run --benchmark performance_microsuite --dry-run
```

### Compare Autoregressive Baseline against MTP Candidate:
```bash
opengrad benchmark compare \
  --baseline reports/benchmarks/bfcl_ar_run \
  --candidate reports/benchmarks/bfcl_mtp_run
```
