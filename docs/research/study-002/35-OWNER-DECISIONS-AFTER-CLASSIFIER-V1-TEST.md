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

## 5. How rare DIRECT is among single-exchange prose records (2026-09-17)

> **Correction (§6).** This section was first titled "across the whole corpus", and its best reading and
> consequence below claimed that the corpus holds almost no direct answers and that rebalancing cannot restore
> them. That was wrong. It measured only prose *single exchanges*, the records the input contract admits
> (32). The multi-turn records it set aside hold thousands of direct answers, with and without tools (§6). The
> counts below stand; the conclusions are rescoped.

Decided by the owner after §4: measure DIRECT across all of normalization-v3 before choosing between a new
DIRECT-bearing source and a redesign of C1. Counts only, `scripts/audit_corpus_direct_prevalence.py`, written to
`reports/pdet-coverage-v2/normalization-v3.direct-prevalence.json`.

**Where the 181,433 records sit.** 93,198 carry a structured call in the first assistant turn (xLAM
57,342, Glaive 26,843, ToolACE 9,013). 30,975 are prose single exchanges, the only records that can be DIRECT
in the 22 §2 sense (Glaive 14,303, When2Call 14,700, ToolACE 1,972). The rest cannot be sampled as a prose
single exchange: 34,447 multi-turn (34,446 of them Glaive), 20,923 with the call after the first assistant turn,
1,881 malformed, 9 held-out. Of the prose single exchanges, 27,908 offer tools and 3,067 do not.

**Three estimates of DIRECT among the prose single exchanges:**

| Source, tools | Records | Development-label yields | Coverage-reference yields | Frozen classifier v1 |
|---|---:|---:|---:|---:|
| Glaive, no tools | 153 | 131 | 125 | 139 |
| Glaive, tools offered | 14,150 | 16 | 486 | 25 |
| ToolACE, no tools | 697 | 2 | 0 | 13 |
| ToolACE, tools offered | 1,275 | 4 | 0 | 92 |
| When2Call, no tools | 2,217 | 0 | no labels | 29 |
| When2Call, tools offered | 12,483 | 0 | no labels | 150 |

- **Glaive without tools: about 125 to 139 DIRECT records, and the three estimates agree.** They come from about
  150 distinct prompts.
- **Glaive with tools offered: probably about 16 to 25 records.** The reference-yield figure of 486 is one
  labelled refusal-stratum item (1 of 30) multiplied across 14,049 records that repeat 209 prompts; the
  development labels (0 of 32) and the classifier (25 of 14,150) both point low. Treat 486 as an artefact of
  weighting one item, not an estimate.
- **ToolACE and When2Call: probably close to none.** Label yields give 0 to 4. The classifier's 284 DIRECT
  predictions there match its known false-DIRECT behaviour: 0 of 9 agreed on ToolACE (33 §8), and on P-DET-v1,
  a natural When2Call sample with 0 human DIRECT labels, it predicted 8 of 575 (1.4%, against 1.2% here).

**Best reading, for single exchanges only.** Roughly 150 to 165 single-exchange DIRECT records exist, all from
Glaive: about 0.5% of prose single exchanges. Direct answers with tools offered number about 20 of the 27,908
single exchanges that offer tools.

- **Directly shown:** the disposition and record counts; the classifier counts; the label counts.
- **Strongly suggested:** DIRECT with tools offered is nearly absent from the corpus, since three estimates
  built differently agree except where one label is multiplied by 14,049 records.
- **Hypothesis only:** that this absence contributed to Study 001's loss of direct answering. Study 001's own
  corpus composition has not been measured this way.
- **Unknown when written, answered in §6:** DIRECT behaviour inside the multi-turn records and later assistant
  turns.

**Consequence, rescoped.** Under the current input contract, which admits single exchanges only, a
classifier-based C1 sees about 20 DIRECT-with-tools records and has almost nothing to upweight. That is a
limit of the contract's scope, not of the corpus (§6). No population was drawn and nothing else changed.

## 6. Study 001's training corpus, turn by turn (2026-09-17)

Decided by the owner after §5: test the hypothesis on the corpus M0 actually trained on. Counts only,
`scripts/audit_canonical_v2_final_direct.py`, written to
`reports/pdet-coverage-v2/canonical-v2-final.direct-prevalence.json`. The `canonical-v2-final` release
(173,237 records) was downloaded from `arrochi112/OpenGrad-ToolPolicy-Canonical-v2` at revision `df1a1f51…`,
and all 176 shards match the release manifest that `runs/m0_sft_canonical_v2_final/dataset_manifest.json` pins
(`8ced403b…`). M0 trained 161,966 of these records.

SFT supervises every assistant turn, so turns are counted. Of 388,988 assistant turns: 132,895 carry a
structured call, 66,656 are prose right after a tool result, **78,514 are prose first answers to one user
message, and 110,923 are other prose turns in multi-turn conversations.** Most of the prose sits in Glaive
conversations that continue past the first exchange, which the input contract (32) and every P-DET population
exclude.

**Frozen classifier v1 on the prose turns** (input: the turn and the nearest preceding user message):

| Prose turns | Turns | Predicted DIRECT | Glaive DIRECT, distinct user messages |
|---|---:|---:|---:|
| First answer, tools offered | 41,002 | 9,455 | 3,137 |
| Later turn, tools offered | 22,915 | 13,182 | 1,614 |
| First answer, no tools | 37,512 | 33,831 | 33,756 |
| Later turn, no tools | 88,008 | 87,281 | 62,267 |

Nearly all predicted DIRECT is Glaive: 9,288 of the 9,455 tools-offered first answers, and 33,789 of the
33,831 without tools.

- **Directly shown:** the turn counts and the classifier's outputs.
- **Strongly suggested:** Study 001's corpus holds thousands of direct answers with tools offered, and tens of
  thousands without. The classifier's DIRECT precision on Glaive single exchanges was 0.975 (33 §8). Even if its
  precision on these first answers were far lower, the count would stay in the thousands.
- **Not shown:** the classifier was never tested on turns taken from multi-turn conversations. For later turns
  its input drops the conversation, so those counts are rough. The label-yield projection in the report is
  not used here: its Glaive yields come from single exchanges that mostly offer no tools, and do not carry over.
- **The hypothesis of §5 is not supported.** A lack of direct answers in the training data does not explain
  Study 001's loss of direct answering: the corpus holds many, including tens of thousands with no tools offered.
  M0's measured failure was on bare zero-shot GSM8K questions (README).

**Consequences.**
1. **§4 is scoped to the current population unit.** The unused pool of *single exchanges* cannot supply 50
   DIRECT items. First answers inside multi-turn Glaive conversations could: about 3,100 distinct user
   messages with tools offered are predicted DIRECT. Using them needs a new input contract and population unit.
2. **§5's C1 consequence is a limit of the contract, not the corpus.** Direct answers exist to reweight; C1 as
   specified cannot see them.
3. How to continue is the owner's next decision. Nothing was drawn and no rule changed.

