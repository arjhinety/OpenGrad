# 35 — Owner decisions after the one-shot test of `prose-decision-classifier-v1`

**Status: DECISION RECORD, 2026-09-17, decided by the study owner (arjhinety) after the one-shot result of
[33](33-PROSE-DECISION-CLASSIFIER-DEVELOPMENT-PLAN.md) §8 was known.** It changes no threshold, population,
label, input contract or acceptance rule. It authorises no C1 balancing, mixture or training, and creates no
gold label.

## 0. What the result left open

By the rules of [22](22-PDET-PROTOCOL.md) §6, C1 needs the classifier to qualify for **DIRECT and UNSUPPORTED**
(`c1_authorised` in `src/opengrad/verification/pdet_coverage_metrics.py`). CALL and CLARIFY failing deny only
balancing for those modes. `prose-decision-classifier-v1` qualified for UNSUPPORTED and CLARIFY. DIRECT did not
qualify, because neither population holds 50 reference DIRECT items (44 on P-DET-COVERAGE-v1, none on P-DET-v1).
So **DIRECT is the one mode that blocks C1.** Two questions followed.

## 1. A `MODEL_REFERENCE` qualification may count toward balancing permission, provisionally

This is the decision [34](34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md) §4 item 2 (and
[28](28-PDET-MODEL-LABEL-AMENDMENT.md) item 2) left to the study owner.

**Decision.** A qualification measured against the three-model consensus reference may count toward balancing
permission, under these conditions:
1. **It is always named as such.** Every such qualification is reported as `MODEL_REFERENCE`, with the
   reference named as in 34 §4 item 1. It is never called human-validated.
2. **Human labels take precedence.** If a human label ever exists for an item, it replaces the consensus label
   and every affected qualification is re-evaluated (34 §4 item 4). A qualification lost that way withdraws the
   permission that rested on it.
3. **It grants eligibility, not action.** Meeting the rules still does not start C1, a mixture change or
   training; each needs the owner's explicit instruction.

**Why.** The owner annotates nothing and no other human annotator is available (34 §1). Without this decision
no result on P-DET-COVERAGE could ever grant permission, and C1 would stay blocked by resources rather than
evidence.

**Disclosure.** The question was open before any test (34 §4, adopted before any label existed), but it was
answered after the v1 result was seen. The v1 result does not by itself gain C1 from it, because DIRECT did not
qualify. Any later report that relies on this decision cites this file.

## 2. DIRECT is made measurable by a classifier v2 and a new untouched population

**Decision.** Rather than testing v1 again on a larger DIRECT population, or pausing:
1. **Classifier v2.** The false-DIRECT weakness is addressed in `prose-decision-classifier-v2`. v1 stays frozen
   and its §8 result stands as v1's result.
2. **Exposure.** From the first v2 rule change, P-DET-v1 and P-DET-COVERAGE-v1 are `DEVELOPMENT_EXPOSED` for
   v2 (22 §6). The v1 test result showed where v2 should look (33 §8 caveat 4: ToolACE 0 of 9 DIRECT
   predictions agreed; 8 DIRECT predictions on P-DET-v1), so no qualification of v2 may rest on either
   population.
3. **A new untouched population** is required before v2 can gain balancing permission (22 §6). It is sized so
   that DIRECT can reach its minimum of 50 reference items, overall and in the challenge component where
   30 §11 requires it, and so that per-source DIRECT precision can be evaluated for ToolACE. CALL is included
   only if the pool allows; C1 does not depend on it.

**Order of work, each step documented before it runs:**
1. **Counts-only dry run** of the eligible pool, excluding every item already used (P-DET-v1,
   P-DET-COVERAGE-v1, `prose-classifier-dev-v1`, `-devcheck-v1`, `-devcheck-v2`), to learn how many DIRECT and
   textual CALL candidates exist per source. Whether prose CALL can reach any minimum is **unknown** until then.
2. **A preregistration amendment** for the new population (its draw, sizes, strata and minimums), adopted
   before it is drawn. Its reference follows 34: the same three non-Claude models, two of three.
3. **A v2 development plan**, on fresh development data that excludes every population, labelled and archived
   as in 33 §3–§5a, with held-out check sets.
4. **Freeze v2** (commit and tag) before any result on the new population is computed. The developer does not
   read the new population's items, labels or rationales.
5. **One-shot test** of the frozen v2 on the new population.

## 3. What does not change

- P-DET-v1, its human labels and its unfrozen status (the optional sample re-label stays available).
- P-DET-COVERAGE-v1, its consensus reference and the v1 test result.
- `prose-decision-classifier-v1` and its tag.
- Every threshold and minimum of 22 §6 and 30 §11, including the rule that C1 needs DIRECT and UNSUPPORTED.
