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
1. **Training Boundary (`src/opengrad/training/sft.py`, `dpo_live.py`, `mtp.py`)**:
   - **Implemented** for the model's native MTP depth, which is one layer for Qwen3.5-2B
     (`full-model-components-v1`, [`MODEL_COMPONENT_POLICY.md`](MODEL_COMPONENT_POLICY.md)). Every trainer
     carries the checkpoint's `mtp.*` layer and trains it:
     $$\mathcal{L} = \mathcal{L}_{\text{primary}} + \lambda \, \mathcal{L}_{\text{mtp}}$$
     The layer's output at position $t$, built from $h_t$ and the embedding of $x_{t+1}$, is scored against
     $x_{t+2}$. This is the pairing vLLM's proposer drafts with.
   - The configuration is the checkpoint's native layer count, not a free `num_heads`:
     ```yaml
     trainer:
       mtp:
         loss_weight: 0.3          # SFT default; DPO defaults to 1.0
         gradient_scope: joint     # SFT default; DPO defaults to head_only
     ```
   - Recursive multi-step MTP ($K > 1$ on one native layer) is not implemented.
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
