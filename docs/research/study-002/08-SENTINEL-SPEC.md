# 08 — Sentinel specification

Sentinels are the cheap, frequent guards that made Study 001's regression visible at all — and, in the
author's own accounting, they were also the reason it stayed invisible for so long. `docs/EVALUATION.md:46-52`
records the failure mode: *"On the 7-case OpenWeights sentinel, Base and the promoted checkpoint both
scored **5/7** — Base by answering `trap-arithmetic` and `multi-step-change` *wrongly*, the promoted
checkpoint by passing `trap-arithmetic` and *refusing* `multi-step-change` and `format-constraint*.
Identical score, different failures and opposite behaviour."*

Two rules follow, and they govern this document: a sentinel reports **failure modes**, not totals, and **no
small suite may carry a capability conclusion** (`EVALUATION.md:51-52`).

## Placement

Sentinels run **before** a promotion decision, not after it (G2, `docs/research/GUARDRAILS.md:29-33`).
`docs/research/STUDIES.md` records this as one of the two things Study 002 exists to fix: Study 001's
regression *"was measured by sentinels after promotion rather than before it"*. In Study 002 the sentinel
suite is a precondition of the verdict, and a missing sentinel artifact makes the verdict
`NOT_EVALUABLE` rather than either PASS or FAIL ([15](15-PROVENANCE-VALIDATORS.md)).

## The instrument Study 001 already got right

The 8-shot GSM8K arm is kept and promoted to a first-class instrument, because it was pre-registered in the
adapter *before any generation* precisely so that a claim was falsifiable:
`reports/GENERAL_CAPABILITY_REGRESSION.md:97-102`: *"`fewshot8` supplies eight worked exemplars, which
demonstrate the answer format without changing the arithmetic. It was registered in the adapter before any
generation, precisely so that 'the model can do it but declines to' is falsifiable rather than asserted."*

That is the pattern this specification generalises: every sentinel that supports a decision-layer claim is
registered before the run, and its interpretation is fixed in advance. The measured baseline it produced:

| stage | IFEval strict | GSM8K 0-shot | GSM8K acc\|answer | GSM8K 8-shot | MMLU-Pro @2048 | answer rate | refusal rate | 7-case |
|---|---|---|---|---|---|---|---|---|
| Base (Qwen3.5-2B, no SFT) | 67.8 | 67.4 | 67.4 | 70.4 | 49.0 | 100.0 | 0.0 | 5/7 |
| M0 — SFT | 45.1 | 0.0 | — | 56.3 | 37.0 | 0.0 | 100.0 | 5/7 |
| M1-v2 — DPO (promoted) | 45.8 | 0.0 | — | 55.5 | 37.0 | 0.0 | 100.0 | 5/7 |
| M1-v1 — DPO (other lineage) | 43.4 | 17.3 | 59.8 | 70.4 | — | 28.9 | 70.7 | 3/7 |

All figures are percentages of the relevant population; `answer rate` and `refusal rate` are GSM8K 0-shot.
`M1-v1` matters twice over: it is a different lineage with **no SFT stage** and it still refuses 70.7% of
zero-shot questions, which is why H1 is stated as refusal-`ANSWER` disagreement rather than as an SFT
effect, and its `GSM8K acc|answer` of 59.8% is the cleanest existing evidence that the ability survived
while the decision did not.

## Sentinel inventory

| id | Instrument | Population | Endpoint | Modes | Blocks? |
|---|---|---|---|---|---|
| `S-ANS-0` | GSM8K 0-shot | frozen file | `answer_rate`, `refusal_rate` | 0-shot | **yes** (anti-pattern guard) |
| `S-ANS-8` | GSM8K 8-shot | frozen file | `accuracy`, `accuracy_given_answer` | 8-shot | yes (H2 endpoint) |
| `S-ANS-E` | answer elicitation probe | ANSWER-gold with an explicit answering instruction | `answer_rate`, `no_call_accuracy` | elicit | yes (H2 limit) |
| `S-TP-4` | tool-policy four-mode population | `P-CONF` | four per-mode accuracies, `call_f1`, `over_call_rate` | 0-shot | yes (H5) |
| `S-REF` | refusal-correctness probe | `P-UNANS` | `refusal_correctness` | 0-shot | **yes** (H6) |
| `S-IF` | IFEval strict / loose | frozen file | strict and loose accuracy, failure classes | 0-shot | yes (format and instruction) |
| `S-MMLU` | MMLU-Pro 5-shot @2048 | frozen file | `accuracy`, truncation rate, adversarial interval | 5-shot | yes (ability) |
| `S-OW7` | OpenWeights 7-case | frozen 7 cases | per-case pass/fail and failure class | 0-shot | **no** — reported, never decisive |
| `S-NV` | non-vacuity census | every scored population | mode coverage, census reconciliation | — | **yes** |

`S-NV` is the sentinel L1 demands: it asks whether every required mode had `n > 0` and whether the
counters reconcile (`discovered == checked + blocked + skipped`), on the principle stated in
`src/opengrad/verification/accounting.py`: *"a gate that returns success without doing any work is
indistinguishable from one that did the work and found nothing wrong."* A sentinel suite that scores
nothing and reports PASS is the same bug as `no_call_accuracy` returning `0.0` on an empty class.

## Prompt modes

Every sentinel runs in the mode it is defined with, and the mode is part of the endpoint's identity:

| Mode | Purpose | Endpoint it answers |
|---|---|---|
| `0-shot` | deployed behaviour | the Study 001 headline condition; `S-ANS-0` |
| `8-shot` | demonstration of the answer format | H2 — capability survived while the decision did not |
| `elicit` | explicit statement that a direct answer is expected where the model can answer, and that refusal is permitted where it cannot | H2's practical limit, and `S-ANS-E` |

The three modes are not interchangeable evidence and are never averaged together. The existing numbers
show why: at 0-shot M0 answers nothing, at 8-shot it reaches 56.3% against Base's 70.4%. A single "GSM8K
score" for M0 could be either 0.0 or 56.3 depending on an unstated prompt convention, which is exactly the
`#9`/`#53` defect class in a different costume.

## Why `S-OW7` never decides anything

The 7-case suite is retained because it caught the regression first and because its per-case record is
informative (`reports/GENERAL_CAPABILITY_REGRESSION.md:4-6`: the regression *"was raised by the 7-case
OpenWeights sentinel"*). But it is barred from deciding, for two measured reasons:

1. Both Base and the promoted checkpoint scored 5/7 with opposite behaviour, so the total is not
   sensitive to the failure that matters.
2. A 7-case suite cannot resolve anything: the worst-case resolvable margin at n = 7 is 2 × 37.1pp = 74pp.
   A capability conclusion from it is arithmetically impossible, which is the point
   `docs/EVALUATION.md:51-52` makes.

`S-OW7` therefore reports a per-case table with failure classes on every arm, and any arm whose per-case
pattern differs from `C0`'s in the refusal direction is flagged for inspection. The flag is a prompt to
look at `S-TP-4`, `S-REF` and `S-ANS-*` — never itself a verdict.

## Blocking semantics

- A sentinel that **blocks** enters `study_002_gate_v1` ([11](11-THRESHOLDS.md)) with its margin and its
  resolvable margin; a failure is a blocking failure, and a `WITHIN_NOISE` margin is not a failure but
  cannot be counted as a pass either — it is reported `WITHIN_NOISE` and excluded from the verdict's
  support.
- A sentinel that **does not block** still must run. A missing non-blocking sentinel is a provenance error
  ([15](15-PROVENANCE-VALIDATORS.md)) because its absence is otherwise indistinguishable from a pass.
- A sentinel whose population is empty (`n = 0`, `S-NV`) blocks regardless of its own row, because the row
  would otherwise be read as satisfied.

## Reporting rules

1. **Failure modes, not totals.** Every sentinel prints its per-item outcome and its failure class, using
   the vocabulary already in the repository: `FAILURE_CATEGORIES` in `capability.py:54-63` —
   `REFUSAL`, `WRONG_CONTENT`, `FORMAT_VIOLATION`, `LENGTH_VIOLATION`, `MISSING_REQUIRED_ELEMENT`,
   `EXTRA_FORBIDDEN_CONTENT`, `PARSE_FAILURE`, `OTHER` — and, for IFEval, the deterministic per-instruction
   mapping `IFEVAL_ID_TO_CATEGORY` (`capability.py:69-95`), which is exhaustive over the registry and
   asserted in tests against the live registry so an upstream addition cannot fall through to `OTHER`.
2. **Truncation per stage.** `finish_reason` is recorded for every sentinel and the truncation rate is
   printed per stage per arm. The MMLU-Pro history is the reason: Base truncated on 4,527 of 12,032
   (37.6%) at a 768-token budget against M0's 869 (7.2%), and **99.6% of Base's unattempted examples
   (4,292 of 4,309) were truncations, not refusals** — the benchmark was measuring verbosity. At 2,048 the
   residual is Base 2,627 (21.8%) against M0 759 (6.3%), still a ~3.5× imbalance, so the MMLU-Pro
   comparison is reported as a **truncation-adversarial interval** rather than a point or a "lower bound"
   (`reports/GENERAL_CAPABILITY_REGRESSION.md:139-145`).
3. **Decoding pinned, scorer deterministic.** Greedy decoding at the frozen generation config, and the
   scorer run twice on one fixed generations file returning identical numbers
   (`docs/EVALUATION.md:54-60`: 0.4510 / 0.4492 / 0.4492 before seeding).
4. **The instrument is named in the row.** `refusal_detection_method` is printed beside every
   refusal-derived sentinel value, and a `HEURISTIC` classification is never presented as deterministic.
   The IFEval failure table is the precedent: one `REFUSAL` class is heuristic and the other five classes
   are deterministic, and the table says so (`GENERAL_CAPABILITY_REGRESSION.md:117-120`).
5. **Sentinel ≠ benchmark.** A sentinel is a guard run frequently across checkpoints; a benchmark is a
   frozen external suite. Neither substitutes for the other, and a sentinel result never appears in a
   benchmark column.

## Schedule

| When | What runs | Why |
|---|---|---|
| at each checkpoint evaluated during selection | `S-ANS-0`, `S-NV` | cheap; catches a collapse during selection rather than after it |
| once per arm, before any promotion decision | the full suite, every mode | G2: sentinels before promotion |
| after the arm set completes, before the verdict | `S-TP-4` and `S-REF` on `P-CONF` / `P-UNANS` | the confirmatory sentinels, scored once per arm |
| on any re-run or replacement | the full suite again | a replacement run has its own sentinel set or it is not a replacement |

## Interaction with H2

`S-ANS-0`, `S-ANS-8` and `S-ANS-E` together answer RQ2.2 and test H2 as a 3-point curve: refusal under
deployed prompting, capability under demonstration, and the recovery obtainable from instruction alone.
The expected shape on `C0` is the Study 001 shape — 0-shot answer rate 0.0%, 8-shot 55.5% against Base's
70.4%. H2 is supported if `R1` moves `S-ANS-0` materially while leaving `S-ANS-8` at `C0`'s level or
better, and the 8-shot/0-shot gap is the quantified statement of "the ability was never lost".

`S-ANS-8` is also the guard against a **false** H1. If `R1` raises `S-ANS-0` but drops `S-ANS-8`, the
intervention did not restore a decision — it damaged a capability, and the verdict is not
`MECHANISM_SUPPORTED_*`. That asymmetry is checked by the gate rather than left to a reader.