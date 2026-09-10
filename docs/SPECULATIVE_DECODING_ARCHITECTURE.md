# Speculative Decoding & MTP Architectural Integration

**Building in Public.** OpenGrad establishes the architectural boundaries for multi-token prediction (MTP) and speculative decoding (such as DSpark-style parallel verification) before weights are trained.

---

## 1. Native MTP Heads Architecture

Multi-Token Prediction (MTP) attaches $K$ lightweight auxiliary prediction heads to the transformer backbone:

```text
                  Shared Transformer Backbone
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
       Primary Head        MTP Head 1         MTP Head 2
     (Position t+1)     (Position t+2)     (Position t+3)
```

### Integration Points:
1. **Training Boundary (`src/opengrad/training/sft.py`)**:
   - The forward pass computes joint cross-entropy loss:
     $$\mathcal{L} = \mathcal{L}_{\text{primary}} + \sum_{k=1}^K \lambda_k \mathcal{L}_{\text{mtp}, k}$$
   - When training native MTP heads, the `training_config` declares:
     ```yaml
     trainer:
       type: sft
       mtp:
         num_heads: 3
         loss_weight: 0.3
     ```
2. **Inference Backend Boundary (`src/opengrad/benchmarks/backends/protocol.py`)**:
   - The model generates draft tokens across heads $\{1, \dots, K\}$ concurrently.
   - The verifier validates candidates against the primary logits in a single parallel verification pass.
   - Emits `MTPDepthMetrics` via `speculative_metadata`.

---

## 2. DSpark / Speculative Decoding Integration

For draft-model speculative decoding or DSpark-style acceleration:
1. **`SpeculativeMode.MODE_D_DSPARK`**:
   - The primary verifier model runs alongside the speculative draft engine.
   - The `InferenceBackend` protocol captures:
     - `proposed_tokens`
     - `accepted_tokens`
     - `acceptance_rate`
     - `accepted_tokens_per_step`
     - `verifier_steps`
     - `rollback_count`
     - `verification_overhead_ms`
2. **Quality Parity Guarantee**:
   - Every speculative run is paired against an autoregressive (`MODE_A_AR`) baseline on identical prompts.
   - Evaluates token match rate, tool-call equivalence, and benchmark score delta.
   - Evaluates the **Pareto frontier** (`src/opengrad/benchmarks/speculative/pareto.py`) to reject configurations where speedup causes unacceptable degradation.
