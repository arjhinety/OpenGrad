# 39 — canonical-v3 decision balance (specification)

**Status: SPECIFICATION, 2026-09-18.** Written under the C1 authorisation of
[38](38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md) §3, after the behaviour-labelling pass of
[21](21-C1-IMPLEMENTATION-STATUS.md) phase 3 produced labels for all 181,433 normalization-v3 records. It fixes
how a balanced decision mixture is computed **before** any corpus is materialized, so the numbers cannot be
chosen after seeing what they produce. It builds no training corpus, moves no arm and authorises no training.

## 1. What the labels give us, and what they do not

| Decision stratum | Where its membership comes from | Records available |
|---|---|---:|
| `CALL` | **structure**: the first assistant reply *is* a structured tool call (30 §4, layer A) | 92,851 |
| `ANSWER` | classifier label `DIRECT`, weight permitted (glaive only, 38 §2) | 33,927 |
| `UNSUPPORTED` | classifier label `UNSUPPORTED`, weight permitted | 22,419 |
| `CLARIFY` | classifier label `CLARIFY`, weight permitted | 22,014 |

Two exclusions follow from 38 §2 and are not revisited here: ToolACE's 86 and When2Call's 177 `DIRECT` labels
carry no weight (their post-stratified precision row was `NOT_EVALUABLE`), and the classifier's 12 `CALL`
labels carry no weight. `ABSTAIN` (7,791) and `UNLABELLED` (2,156) are never weighted.

**Why `CALL` may define a stratum although a classifier `CALL` label may not carry weight.** 38 §2 withholds
weight from *a CALL label produced by this classifier*, because a false textual CALL would inject a wrong
tool-call target into training. The `CALL` stratum here is not that: membership is the structural fact that the
reply carries `tool_calls`, which 30 §4 keeps out of the classifier entirely. No classifier judgment enters it.
The labels artifact's `never_weighted` list is about classifier labels, and is worded accordingly.

**What is out of reach.** `configs/data/tool_calling/balanced_policy_v1.yaml` balances eleven *capabilities*
(`must_call`, `direct_answer_retention`, …), not decisions. No capability labeller exists, and the classifier
cannot produce one: it answers "what does this reply do", not "what capability does this record exercise".
That config stays `HYPOTHESIS_ONLY` and is **not** the mixture materialized here. Saying otherwise would claim
a balance we cannot measure.

## 2. The balance rule, fixed here

**Target: equal shares across the four decisions, supply-limited.** Each stratum contributes
`n = min(supply across strata)` records, so the realized mixture is 25% / 25% / 25% / 25%.

- **Why equal.** The defect C1 exists to correct is a *collapse*: Study 001's corpus taught one decision almost
  exclusively, and the evaluation could not see the loss (35 §6, 21). A flat decision prior is the plainest
  statement that no decision is privileged, and it needs no free parameter chosen after seeing results.
- **Why supply-limited rather than up-weighted.** Repeating records to hit a share would train on duplicates and
  inflate the rarest stratum's few prompts. Cutting the larger strata instead keeps every record distinct.
- **Cost, stated plainly.** The binding stratum is `CLARIFY` at 22,014, so the balanced set is 88,056 records
  and about 70,800 structural-call records are left out of it. That is a real loss of call supervision, and it
  is the price of a flat prior. Any later arm that wants more call data must say so as its own factor (31 §9.6),
  never silently.

**Selection within a stratum is deterministic:** records are ranked by `sha256(seed | record id)` and the first
`n` taken, with seed `opengrad-canonical-v3-decision-balance-v1`. Re-running reproduces the same set.

**Per-source caps.** None beyond 38 §2's source rule for `ANSWER`. The realized per-source composition is
recorded, not constrained, so that a later analysis can see what the flat decision prior did to source mix.

## 3. What the artifact contains

`python -m opengrad.data.canonical_v3_balance --build` writes, under `reports/canonical-v3/`:

- the supply table per stratum and per source;
- the realized selection: counts per stratum, per source and per `unit_kind`, with the record ids of the
  selected set kept beside it in git-ignored data;
- the seed, the labels manifest hash, the corpus fingerprint, the classifier tag and hash, the contract
  version, and the authorisation (38);
- the statement that this is a **plan**, not a corpus: no record has been rendered, no shard written, no arm
  changed.

`--verify` recomputes the selection from the same inputs and checks it byte for byte.

## 4. Conditions and limits

1. **Not a corpus.** Materializing canonical-v3 (21 phase 5) is a separate step: it must re-check contamination
   (`configs/data/tool_calling/contamination.yaml`), renderability and the supervision contracts before writing
   anything.
2. **Not an arm.** No arm of [04](04-ARM-MATRIX.md) moves to canonical-v3. Entering a study needs 21 phase 8's
   preregistration amendment, decided separately (31 §9.6).
3. **Not training.** 38 §4 stands: authorising C1 permits building, not spending GPU time.
4. **Provisional footing.** Every stratum but `CALL` rests on classifier labels whose qualifications are
   provisional `MODEL_REFERENCE` ones (37 §8). If 38 §5's withdrawal conditions fire, this balance is void.
