# Teacher Model Selection & Tokenizer Compatibility

**Building in Public.** A teacher model is supervision, not ground truth. A larger model is never assumed to be superior without empirical verification.

---

## 1. Teacher Model Definition

- **Primary Student:** `Qwen/Qwen3.5-2B` (Pinned Revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`)
- **Primary Teacher:** `Qwen/Qwen3.8-27B` (or 27B–32B class Qwen teacher `Qwen/Qwen2.5-32B-Instruct`)

Both models share the Qwen architecture family, identical special tokens, and a 248,064 token vocabulary.

---

## 2. The Tokenizer Compatibility Gate

Token-level next-token distribution distillation requires that token IDs match identically between student and teacher. OpenGrad enforces a fail-closed gate:

```bash
opengrad distill validate-teacher \
  --student Qwen/Qwen3.5-2B \
  --teacher Qwen/Qwen3.8-27B
```

### Verification Criteria:
1. Exact vocabulary size match.
2. Identical token $\to$ ID mapping for all standard vocabulary elements.
3. Special tokens match (`<|im_start|>`, `<|im_end|>`).
4. Tool markup tokens match (`<tool_call>`, `</tool_call>`).
5. Representative text and tool call tokenizations produce identical sequences.

If any token differs, OpenGrad emits:
```text
TOKENIZER_INCOMPATIBLE
```
and training is blocked. OpenGrad never silently maps incompatible logits by index.

---

## 3. Thinking-Mode Compatibility

OpenGrad evaluates tool policy in deployed non-thinking mode (`thinking: false`). To prevent accidental objective misalignment:
- Student rollout: `thinking = false`
- Teacher scoring: `thinking = false`

A student non-thinking prompt is never paired with a teacher hidden thinking trajectory.

---

## 4. Teacher Advantage Evaluation Gate

Before running on-policy distillation, OpenGrad evaluates student vs. teacher performance on a representative prompt-state sample:
```bash
opengrad distill smoke
```
If the teacher does not demonstrate a statistically meaningful advantage ($\Delta \ge +3.0$ points) over the student on the target residual domain, execution halts with:
```text
TEACHER_GAP_INSUFFICIENT
```
Distillation is only executed when there is capability worth transferring.
