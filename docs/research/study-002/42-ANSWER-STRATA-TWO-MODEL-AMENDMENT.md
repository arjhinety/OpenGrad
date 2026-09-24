# 42 — The `ANSWER` strata reference: two models, both must agree (`study_002_prereg_v10`)

> **Status: ADOPTED 2026-09-24.**
> - **What happened:** the third annotator of [41](41-ANSWER-STRATA-AMENDMENT.md) §7, gpt-5.6-sol, cannot
>   run.
> - **Decision:** the study owner was shown three options: restore access to the same model, substitute
>   another model, or use the two remaining models with a both-must-agree rule. The owner chose the third:
>   *"Two-model rule (v10)"*.
> - **What this document does:** it amends 41 §7 and §9 only.
> - **When it was written:** before any label was read. Recorded in [03](03-PREREGISTRATION.md) and
>   `reports/ERRATA.md` §27, in the same commit.

## 1. Why

- **The refusal.** On 2026-09-24 every call of `codex exec -m gpt-5.6-sol` returned HTTP 400: *"The
  'gpt-5.6-sol' model is not supported when using Codex with a ChatGPT account."*
  - The same refusal came back for a one-line test prompt with no item in it, and with Codex's default
    configuration.
  - So it is a property of the account's access, not of the items or the network.
- **No partial labels.** The runner recorded no label for gpt-5.6-sol: its session holds 0 of 1,767. The
  failed batch's streams stay in the git-ignored working directory.

## 2. What changes

- **41 §7, the annotators.** Two, not three: Gemini 3.8 Flash (High) and deepseek-v4.1-flash. Everything
  else in §7 stands:
  - the one task, the mixed order and the blinding;
  - the pinned procedure and the isolated runner;
  - no Claude model labels an item.
- **41 §9, the consensus rule.** An item's reference label is the label **both** annotators give. When they
  differ, the item is `NO_CONSENSUS` and is excluded, exactly as a three-way split was. Everything else in §9
  stands:
  - the two strata, reported separately;
  - a pooled figure printed only beside them;
  - pool N's other labels as a by-product, and pool K's other labels as construction failures;
  - `MODEL_REFERENCE`, provisional.
- **The configuration.** gpt-5.6-sol is removed from `model_annotators` in
  `configs/annotation/answer-strata-v1.yaml`. The consensus step (`pdet_coverage_reference`) declares the two
  annotators for this task and applies the both-agree rule to it only. P-DET-COVERAGE's three-model rule is
  unchanged.

## 3. What was known when this was written

- **Labels recorded so far:**
  - deepseek-v4.1-flash, 160 of 1,767 items;
  - Gemini 3.8 Flash (High), 0;
  - gpt-5.6-sol, 0.
- **What was read:** completion counts only. No label, label distribution, flag or rationale was read, so
  this change could not be chosen for its effect on the strata.

## 4. Consequences, stated before the results

- **Stricter agreement means smaller strata.** Under 41, an item reached `ANSWER` when any two of three
  agreed. Now both annotators must agree, so a single dissent excludes the item. The strata can only be as
  large as the items the two agree on, and every §10 sizing status is judged on that smaller n.
- **One fewer independent view.** A shared error of the two models is no longer out-voted by a third.
  Agreement is reported as the one pair's model-model agreement, never as inter-annotator agreement.
- **Not the same reference as P-DET-COVERAGE.** That population's reference is a two-of-three consensus of
  three models (34). The two references are not comparable as instruments. A result that relies on both
  says so.
- **One procedure sentence no longer holds, and is kept.** The pinned procedure tells each model its labels
  are "combined with two other models' independent labels"; there is now one other.
  - **Why it stays:** changing it would change the procedure's pinned hash partway through labelling, and
    160 items are already labelled under it.
  - **Why it does not matter:** the sentence says how labels are combined afterwards, not how an item is
    decided, so it cannot change a label.

## 5. Not changed

- **The population:** `ANSWER-STRATA-v1`, 1,767 items, population sha256 `25ef4c25…`. The screen, the rubric
  (41 §8) and the procedure file are unchanged.
- **41 §10–§13:** the sizing table, the no-redraw rule, balance measured not enforced, the limitations, and
  everything 41 §13 lists.
- **P-DET-COVERAGE-v1 and v2,** their references, and 34's three-model rule.
