# Synthetic Preference Generation & DPO Adjudication

**Building in Public.** High-quality preference pairs require targeting the policy decision boundary with hard negatives, not corrupting JSON syntax artificially.

---

## 1. Generation Pipeline

```text
Training-Only Prompt (Excluding Held-Out Benchmarks)
         │
         ▼
  Active Student Model Checkpoint
         │
         ▼ (Generate N=4 Candidates at varied temperatures)
  Candidate Responses {c1, c2, c3, c4}
         │
         ▼
  Stage 1: Deterministic Preference Judge (src/opengrad/preferences/deterministic_judge.py)
         │  (Evaluates schema, tool existence, call decision, grounding)
         ├─ Clear separation found (Margin ≥ 1.5) ──► Pair Formed (Source: "deterministic")
         │
         ▼ (Ambiguous semantic boundary)
  Stage 2: OpenAI DPO Judge (src/opengrad/preferences/openai_judge.py)
         │  (Strict JSON response, budget-bounded, persistent disk cache)
         ▼
  Quality Gate: chosen != rejected, non-empty, valid tokenization
         │
         ▼
  Immutable DPO Dataset (data/processed/synthetic_dpo_pairs.jsonl)
```

---

## 2. Targeting the Decision Boundary

OpenGrad rejects creating trivial pairs where the rejected response is merely broken JSON syntax. Instead, preference pairs focus on hard negatives where both candidates are syntactically valid but differ in policy correctness:

- **`CALL` vs. `DO_NOT_CALL`**:
  - *Chosen*: Directly answers a general knowledge question.
  - *Rejected*: Unnecessarily calls a search tool for a static fact.
- **`SELECT`**:
  - *Chosen*: Selects the exact domain-specific endpoint.
  - *Rejected*: Selects an overlapping but incorrect utility tool.
- **`ASK_FIRST` (Clarification)**:
  - *Chosen*: Requests user clarification when critical parameters are missing.
  - *Rejected*: Hallucinates default values and executes prematurely.
- **`GROUND_ARGUMENTS`**:
  - *Chosen*: Restricts argument values to entities provided in context.
  - *Rejected*: Invents plausible but ungrounded parameters.

---

## 3. OpenAI Judge & Secret Safety

1. **Environment-Only Credential**:
   `OPENAI_API_KEY` is read **only** from the environment variable. It is never logged, printed, serialized into run manifests, or checked into version control.
   Template provided in `.env.example`.
2. **Hard Budget Enforcement**:
   Configuration in `configs/providers/openai_judge.yaml` sets strict bounds (`max_requests`, `max_cost_usd`). When the budget is reached, generation halts immediately.
3. **Structured Outputs Only**:
   The judge returns pure JSON (`preferred_candidate`, `confidence`, `reason_codes`). Private chain-of-thought is never stored.
4. **Deterministic Disk Caching**:
   Requests are cached in `.cache/openai_judge/` keyed by model, prompt, and candidate sequences to eliminate duplicate API charges.

---

## 4. CLI Commands

Generate synthetic preference pairs:
```bash
opengrad preference generate --count 50 --candidates 4
```

Validate preference pairs:
```bash
opengrad preference validate data/processed/synthetic_dpo_pairs.jsonl
```

Build DPO mixture and manifest:
```bash
opengrad preference build
```
Emits `reports/data/DPO_MANIFEST.json` recording source provenance and checksums.
