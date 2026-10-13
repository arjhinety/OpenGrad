# 48 — Gate contract 5: the design the owner approved

**Status:** adopted as a plan by the owner, 2026-10-13. This is not an amendment. Part (a) implements rules the
adopted documents already state, with an ERRATA entry for each step, on the §19 and §35 precedent. Part (b) lists
the items that change preregistered meaning. Those go to a numbered amendment draft, and the owner decides each
one before anything is adopted.

**Basis:**
- the open items in `reports/ERRATA.md` §35;
- the literature-grounded audits of 2026-10-13 (ERRATA §36, UP-0018, UP-0019);
- the owner's routing of their findings.

## Owner decisions (2026-10-13)

| Question | Decision |
|---|---|
| Contract number for the new checks | **contract 5**, since contract 4 is already pushed and a PASS under one contract is never a PASS under another |
| The Study 001 scorers' known bugs (MMLU-Pro letter case, IFEval refusal regex, the audit script's refusal count, the report generator's hash wording) | **a new, versioned Study 002 scorer** with the fixes and a test per failure case; the Study 001 scripts stay as the frozen record, so they still reproduce its artifacts |
| Scope and order | **approved as below** |

## (a) Implemented as written

| # | What the documents say | What the code did | Change | Status |
|---|---|---|---|---|
| 1 | 15 V3: every number carries metric + version, arm, seed, partition id + fingerprint, `n`, denominator, evaluator version, artifact sha256; V9: fingerprints are read from artifacts | check 13 checked that field names were listed; V3 and V9 were never called | V3 on every metric the verdict reads, each bound to a partition fingerprint; V9 on every fingerprint; population sizes read by the gate from the frozen `P-CONF` partition | contract 5, ERRATA §37 |
| 2 | 46 §10 (v14): a between-arm difference on a population resolving more than 10 points is `WITHIN_NOISE` | check 14 counted any row clearing M(n) | rows name their population; `CLARIFY`, `ANSWER-natural` and any population with M(n) > 10pp support nothing | contract 5, ERRATA §37 |
| 3 | 04: tokenizer and template hash pinned; evaluator, generation config and decoder at one version; one device class | no check compared the arms' generation metadata | a check that every arm's run record agrees on those, `INVALID_COMPARISON` otherwise | contract 6, ERRATA §38 |
| 4 | 04 "one generation config"; the H200 runs pin prefix caching off | `VLLMInferenceBackend` set nothing, and B0's record cannot show vLLM's default | pinned `False`, recorded in the engine metadata | done (`a88ba88`) |
| 5 | 05:54: REP-A's residual is "the number against which every threshold margin is compared" | the gate never read REP-A | the bundle carries REP-A's residual; a margin below it is `WITHIN_NOISE`; absent is `BLOCKED` | next |
| 6 | 10 "The primary test": paired cluster bootstrap, all three seed deltas with one sign, `delta_no_call_acc` in the same direction, `MECHANISM_INCONCLUSIVE` | none of it existed | what 10 fixes; what it leaves open (`EVALUATION-FORMULAS-DRAFT.md` questions 1–3, 6, 7) stays a required parameter, so H1 is `BLOCKED` until decided, never PASS by a default | after those questions |
| 7 | selection reads the development side (every freeze record: `select_checkpoint.py --side dev`) | `--side confirmatory` was accepted | refused | done (`a88ba88`) |
| 8 | 07 defines the decision-layer rates from per-example outcomes | nothing computes `refusal_correctness` or `over_refusal_rate`; the gate trusts the bundle | the real-run bundle producer computes every rate from per-example records, and the gate re-derives them | after B2 |

## (b) For the numbered amendment

| # | Item |
|---|---|
| B1 | H6 is called a *safety* hypothesis, but `P-UNANS` holds only questions nobody can answer, and no "should not answer" items |
| B2 | 07 defines `refusal_rate` and `over_refusal_rate` over the same items; 11 says ANSWER-gold |
| B3 | 620 of 1,277 `P-CONF` items share a question with a dev item, and the items are not independent; 06 C2's margins assume they are |
| B4 | no thresholds for `S-ANS-0/8/E`, `S-IF`, `S-MMLU` (08 sets none) |
| B5 | the sign of a comparison outside H1 (10 fixes H1's direction only) |
| B6 | empty replies, and reconciling the confusion matrix with the reported metrics |
| B7 | the ANSWER set keeps only items both labellers agreed on (202 of 1,767 dropped), and the constructed items are all at most 97 characters: a reporting requirement |
| B8 | contamination: an `INCIDENTAL_OVERLAP` verdict can clear an exact match, and the reviewer is not shown to be a person |
| B9 | When2Call gives 16 pairs of identical inputs contradictory gold, `CALL` against `CLARIFY` (UP-0019). No pair lies wholly in `P-CONF`: it holds 8 single items whose contradictory twin is in dev. Whether to report those 8 separately is the question |
| B10 | 05's seed table says seeds 0, 1, 2 cover "both repeated controls", but `REP-B` re-scores `A_M0`, trained on seed 42 in Study 001; the gate reads 42 (ERRATA §38, later line), and 05's wording needs correcting |

## Order

1. Items 4 and 7, then items 1 and 2 (contract 5).
2. Items 3 (contract 6) and 5.
3. Item 6, after the formula questions are answered.
4. Item 8, with the bundle producer, after B2.
5. The amendment draft (B1–B10), shown to the owner before anything is adopted.
