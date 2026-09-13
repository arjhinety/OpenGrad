# Teacher Model Selection & Tokenizer Compatibility

**Building in Public.** A teacher model is supervision, not ground truth. A larger model is never assumed to be superior without empirical verification.

---

## 1. Teacher Model Definition

- **Primary Student:** `Qwen/Qwen3.5-2B` (Pinned Revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`)
- **Primary Teacher:** `Qwen/Qwen3.8-27B` (or 27B–32B class Qwen teacher `Qwen/Qwen2.5-32B-Instruct`)

Tokenizer compatibility between these teachers and the student has **not** been verified: no
teacher tokenizer has been downloaded and compared. The code's own offline mock
(`src/opengrad/distillation/tokenizer_gate.py`) gives the student a 248,064-token vocabulary and
Qwen2.5 a 151,936-token vocabulary, which would fail the gate below.

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

**As currently wired, this gate is not fail-closed.** `opengrad distill validate-teacher` calls
`validate_teacher_tokenizer_offline(..., mock_compatible=True)` (`src/opengrad/agent_cli.py`), which
returns `TOKENIZER_COMPATIBLE` without loading either tokenizer, so the command always passes. The
real comparison (`compare_tokenizers`) exists but is not reached from the CLI. No distillation run
has depended on this gate: M2 was not run (`reports/M2_DECISION.md`).

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
