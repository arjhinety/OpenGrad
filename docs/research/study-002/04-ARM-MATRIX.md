# 04 — Arm matrix

Eleven arms, tiered. Each arm varies **one** factor and declares every factor it holds fixed, because
Study 001's `matched-exposure` arm claimed a matched comparison it did not have — *"The matched-exposure
ablation arm saw 1.19× the reference's supervised tokens (#6)"* — and G5 exists for that reason
(`docs/research/GUARDRAILS.md:43-48`).

## Fixed across every arm

| Factor | Held at | Why |
|---|---|---|
| Base model identity and revision | the canonical pinned revision | any change confounds the mechanism (`registry/models.yaml`, `readiness.py:1235-1250`) |
| Trainer, optimizer, LR schedule, step budget, batch composition | one frozen setting | the factor under test is data, not optimization |
| **Supervised-token budget** | matched, and **measured** per arm | G5: matched means measured, not intended |
| Tokenizer and chat-template hash | pinned | a template change is a separate factor (`readiness.py:1264-1281`) |
| Evaluator version, generation config, decoder | one version | Study 001 mixed engine and hardware (`#34`) |
| Seeds | `k = 3`, identical seed values across arms | [05](05-SEED-AND-REPRODUCIBILITY-POLICY.md) |
| Device class for primary runs | one class, with the provider recorded | [13](13-HARDWARE-AGNOSTIC-EXECUTION.md) |

Example counts are *not* held fixed, because holding supervised tokens fixed cannot hold both. Both are
reported for every arm — this is the `#11`/`#92` defect (*"a nominal 38,400-example estimate was
reported as examples seen"*) stated as a rule.

## Arms

| id | Tier | Corpus / intervention | Varies (the single factor) | Tests |
|---|---|---|---|---|
| `C0` | A | Canonical-v2 exactly as published | — (reference reproduction) | positive control: the harness reproduces the collapse |
| `R1` | A | Canonical-v2, flagged records relabelled **and** target text corrected | label + target of flagged records | H1, H4, primary confirmatory |
| `R2` | B | Canonical-v2, flagged records **relabelled only**, target text untouched | the target-text half of the contract | H4 |
| `R3` | B | Canonical-v2 with flagged records **removed** | presence of the records | H1 via removal; whether deletion would work (not a release candidate) |
| `C1` | B | Canonical-v2 without synthetic/back-translated records | the synthetic-data subset | competing explanation: any narrow SFT collapses |
| `C2` | B | Canonical-v2 without the When2Call-derived source | the source subset | competing explanation: source-specific |
| `X1` | B | Canonical-v2, supervised tokens at 0.5× | exposure | exposure-driven collapse; G5 parity check |
| `D25` | C | As `R1`, 25% of flagged records corrected | dose | H3 |
| `D50` | C | As `R1`, 50% of flagged records corrected | dose | H3 |
| `S1` | C | Canonical-v2 at a second model scale | model scale | H7 |
| `S2` | C | A second corpus construction | corpus construction | H7 |

`D100` is not an arm: it *is* `R1`. The dose ladder is `C0` → `D25` → `D50` → `R1`, all on the same
corpus with identical steps and identical supervised tokens, so the only thing that changes along the
ladder is how many flags were corrected.

### Tiers and spend discipline

- **Tier A** (`C0`, `R1`) is the confirmatory comparison and runs first: 2 arms × 3 seeds = 6 runs.
- **Tier B** (`R2`, `R3`, `C1`, `C2`, `X1`) runs **regardless of Tier A's outcome**, because these are
  the competing explanations, and a favourable Tier A with no control is exactly the Study 001 defect.
  5 arms × 3 seeds = 15 runs.
- **Tier C** (`D25`, `D50`, `S1`, `S2`) is conditional: it runs only if Tier A returns a decisive
  verdict, and the conditioning rule is fixed here rather than chosen later. **Tier C may not be used
  to rescue an inconclusive Tier A** — `MECHANISM_INCONCLUSIVE` is a reportable outcome and Tier C does
  not upgrade it. 4 arms × 3 seeds = 12 runs.

Total: 33 runs plus 2 repeats ([05](05-SEED-AND-REPRODUCIBILITY-POLICY.md)).

### Cost per tier

Priced at the recorded anchor rate of **$1.79 per GPU-hour** (A100 80GB, Verda, on-demand). One arm's cost is
*not* known, because the SFT/DPO split of the ≈36-hour M0 → M1 anchor was never recorded; the assumption
ladder, the arithmetic and the envelope-is-not-a-balance rule are in
[16](16-GPU-READINESS-GATE.md).

| Tier | Runs | At SFT ≈ ⅓ (12 h/arm) | At SFT ≈ ½ (18 h/arm) | At SFT ≈ ⅔ (24 h/arm) |
|---|---|---|---|---|
| A — confirmatory | 6 | ≈ $128.88 | ≈ $193.32 | ≈ $257.76 |
| B — competing explanations | 15 | ≈ $322.20 | ≈ $483.30 | ≈ $644.40 |
| C — conditional, exploratory | 12 | ≈ $257.76 | ≈ $386.64 | ≈ $515.52 |
| repeats | 2 | ≈ $42.96 | ≈ $64.44 | ≈ $85.92 |
| **total** | **35** | **≈ $751.80** | **≈ $1,127.70** | **≈ $1,503.60** |

Evaluation passes are additional: ≈ $1–3 per arm from one measured input, so ≈ $35–105 across the set
([16](16-GPU-READINESS-GATE.md)).

Two consequences worth stating plainly, because they are spend decisions rather than arithmetic:

- **Tier B costs more than the confirmatory comparison.** 15 runs of competing explanations against 6 runs of
  the primary test. That is deliberate: a favourable Tier A with no control is exactly the Study 001 defect,
  and the control is the part of the study that can falsify its claim.
- **Tier C is the only tier whose spend depends on a result.** It is conditional on a decisive Tier A verdict
  and may not rescue an inconclusive one, so ≈ $258–516 of the estimate is contingent by construction. A
  budget that assumes Tier C will run is a budget that has assumed the answer.

## Frozen anchors

Two Study 001 artifacts are **re-scored under the Study 002 evaluator** and reported as anchors:

- `A_B0` — the frozen base model.
- `A_M0` — the frozen M0 SFT checkpoint (`study-001` tag).

Their **Study 001 frozen numbers are historical and never placed in a table with Study 002 rows**
(G9; findings `#9`, `#53`, `#83`, `#84`). They may appear in a clearly separated
"as reported in Study 001 (historical)" block with the Study 001 evaluator version, and the Study 002
re-measurement printed immediately adjacent, so a reader can see both without the two being mixed.
## Single-factor discipline

| Arm | Varies | Held fixed | Confounds to disclose if any move |
|---|---|---|---|
| `R1` vs `C0` | labels + target text of flagged records | corpus membership, count, tokens, steps, seeds, evaluator | none intended; membership is identical by construction |
| `R2` vs `R1` | target text of flagged records | everything else | — |
| `R3` vs `C0` | corpus membership (records removed) | supervised tokens, steps | example count differs and is reported; removed records also remove their call/clarification supervision |
| `C1` vs `C0` | synthetic subset | tokens, steps, seeds | the source mix changes too, so this is *not* a pure synthetic effect — stated as such |
| `C2` vs `C0` | When2Call source | tokens, steps, seeds | see the warning below |
| `X1` vs `C0` | supervised tokens | corpus, steps | fewer tokens per step or fewer steps — whichever is logged, and logged |

The `X1` row is where Study 001's `#6` defect lands: *"Logged supervised tokens: 6,743,788 vs 5,678,531
(1.19×); 1.36× passes over the corpus. **Batches are token-budgeted, so steps don't hold tokens
fixed.**"* The consequence for this design is explicit: **steps are not a matched-exposure control in
this codebase.** Only measured supervised tokens are, and the batch-composition log is the artifact
that proves it. Any arm whose token tally differs from `C0`'s by more than the audit threshold in
[11](11-THRESHOLDS.md) is reported as unmatched and its comparison is downgraded to descriptive.

`C2` carries an explicit warning because Study 001 already made this exact mistake and corrected it in
writing: *"Because xLAM is currently the only CALL_PREDICTION source, those arms measure joint removal
of xLAM and that supervision channel, not a pure xLAM-content effect"* (`ROADMAP.md:48-50`). The same
applies to `C2` if When2Call is the dominant source of any label type: the joint removal is measured
and named rather than glossed as "isolating the source".

## Evaluation factors (not training arms)

Three prompt modes are applied to **every** arm and both anchors as a pre-registered factor of the
evaluation, not a post-hoc comparison:

| Mode | Definition | Answers |
|---|---|---|
| `0-shot` | task instruction only, no exemplars | the Study 001 headline condition |
| `8-shot` | the frozen 8-exemplar prompt | H2 — does demonstrating an answering mode recover it? |
| `elicit` | explicit instruction that a direct answer is expected where the model can answer, and that refusal is permitted where it cannot | H2, and the practical limit of prompt-only repair |

## Scaffold status (G8)

Each arm declares what it needs and whether that exists. This table is part of the pre-GPU gate
([16](16-GPU-READINESS-GATE.md)), because describing a scaffold as working is itself a Study 001
finding (`#66`: a tokenizer gate hardcoded to pass; `#67`: a documented training path that raises).

| Component | Status | Evidence / consequence |
|---|---|---|
| SFT trainer and experiment harness | `LIVE` | the M0 lineage exists (`runs/m0_sft_canonical_v2_final/`) |
| Frozen base and M0 checkpoints | `LIVE` | `study-001` tag |
| Capability sentinels | `LIVE` | `scripts/score_gsm8k.py`, `score_ifeval.py`, `score_mmlu_pro.py` |
| Refusal detector `HEURISTIC_REGEX_v1` | `LIVE`, unvalidated | `scripts/characterize_refusal.py`; **precision unmeasured** |
| Refusal detector `HEURISTIC_REGEX_v2` | `NOT_IMPLEMENTED` | required before any arm; precision and recall measured on a hand-labelled sample |
| Answerability triage partition | `NOT_IMPLEMENTED` | `ROADMAP.md:132-135`: *"This partition is the actual intervention design and does not exist yet"* |
| Corrected corpus builder | `NOT_IMPLEMENTED` | produces a new corpus version; Canonical-v2 stays published (`STUDIES.md:70-71`) |
| `ANSWER`-mode evaluation population | `NOT_IMPLEMENTED` | the L1 fix; required for `study_002_gate_v1` to be able to fail |
| Refusal-correctness metric | `NOT_IMPLEMENTED` | required by H6 |
| Mode-coverage and non-vacuity validators | **PARTIAL** | `study_002_gate_v1` (`src/opengrad/verification/study_002_gate.py`) and `V1`/`V2`/`V12` (`population_validators.py`, `resolvability.py`) implement mode coverage, census, vacuous-metric and resolvable-margin checks plus the behavioural floors through `tool_use_promotion_v5`; `V3`–`V11` and `S-NV` are not built |

Five of the ten components do not exist, and one is partial. That is the honest state of the study, and it is why this
design set is a pre-registration rather than a launch plan: every `NOT_IMPLEMENTED` row is a blocking
check in [16](16-GPU-READINESS-GATE.md), not a nice-to-have.

## What is explicitly not an arm

- **A refusal-removal arm as a release candidate.** `R3` exists to *measure* what deletion would do,
  because `ROADMAP.md:120-124` asserts it would trade over-refusal for hallucination. `R3` may never be
  promoted, and its hallucination result is reported next to its direct-answer result even if the
  direct-answer result is better than `R1`'s.
- **A distillation or speculative-decoding arm.** Moved to [Study 003](18-STUDY-003-ROADMAP.md) and
  [Study 004](19-STUDY-004-ROADMAP.md) by the scope decision in [README](README.md).
- **An engine, hardware, quantization or dtype arm.** Study 001 confounded engine with hardware (`#34`,
  `#43`). Those factors are recorded as provenance ([13](13-HARDWARE-AGNOSTIC-EXECUTION.md)) and are
  never varied inside this study.
- **A "more refusal data" arm.** Adding supervision is a different intervention. The mechanism here is
  a *label contract* defect, so the study tests correction, not augmentation.
- **A safety-tuning or preference-optimization arm.** Anything that changes the harmlessness target
  rather than fixing a mislabelled contract is out of scope for a mechanism study and would make the
  intervention's target undefined.