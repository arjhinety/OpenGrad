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

## 4. Step 1 result: the unused pool cannot supply 50 DIRECT items (2026-09-17)

Counts-only supply audit, `scripts/audit_pdet_coverage_v2_supply.py`, written to
`reports/pdet-coverage-v2/pdet-coverage-v2.supply-audit.json`. The pool is every layer B candidate of the two
sources 30 §6 allows (Glaive, ToolACE), after 30 §9's exclusions, minus P-DET-COVERAGE-v1 and the three
development and check sets, deduplicated. DIRECT yield per stratum and source is taken from labels that already
exist, on two bases reported separately.

| Stratum / source | Unused, after dedup | DIRECT yield: coverage reference | DIRECT yield: development labels |
|---|---:|---:|---:|
| Q / Glaive | 25 | 22 of 30 | 25 of 32 |
| R / Glaive | 121 | 1 of 30 | 0 of 32 |
| M, P1, P2 / Glaive | 0 | – | – |
| M / ToolACE | 606 | 0 of 49 | 0 of 51 |
| P1 / ToolACE | 208 | 0 of 74 | 0 of 83 |
| P2 / ToolACE | 415 | 0 of 20 | 0 of 52 |
| Q / ToolACE | 97 | 0 of 30 | 1 of 33 |
| R / ToolACE | 164 | 0 of 30 | 0 of 33 |
| X (textual call), both | 0 | – | – |

**Expected reference DIRECT if the whole unused pool were drawn: about 22 on either basis**, all in the challenge
strata. Both minimums (50 overall, 30 challenge) are out of reach, before any `NO_CONSENSUS` or `UNKNOWN` loss.

- **What the counts show.** Of the 115 DIRECT labels that exist on these sources (44 reference, 71 development),
  114 are Glaive items and 1 is ToolACE. When2Call, which 30 §6 keeps out of layer B, has 0 DIRECT in P-DET-v1's
  581 human labels and 0 of 138 development labels. Glaive's plain strata (P1, P2) and M are used up: every
  distinct Glaive prompt there is already in P-DET-COVERAGE-v1 or a development set. No textual-call item is left.
- **Estimates only.** Yields come from model labels; the development yields are Claude's, and the coverage
  reference is a two-of-three model consensus. The two bases agree closely (22.3 and 22.4).
- **Consequence.** A new untouched population drawn from canonical-v3's allowed sources cannot make DIRECT
  measurable, so step 2 onwards of §2 cannot proceed as written. Under 22 §5 this is a coverage finding: DIRECT
  gets no balancing permission from this corpus, and C1 is not authorised. How to continue is the owner's next
  decision. No population was drawn and nothing else changed.

