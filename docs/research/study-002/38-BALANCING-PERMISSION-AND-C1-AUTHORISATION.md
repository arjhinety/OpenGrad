# 38 — Balancing permission and C1 authorisation

**Status: DECISION, 2026-09-18.** Taken by Claude under the study owner's explicit delegation ("you have full
authority", 2026-09-18), on the result of the single preregistered test of `prose-decision-classifier-v2`
([37](37-PROSE-DECISION-CLASSIFIER-V2-DEVELOPMENT-PLAN.md) §8). It changes no threshold, no population, no
contract and no metric. It grants permissions that 22 §6 and 35 §1 left to the study owner, and it names, in
advance, what would withdraw them.

## 1. What the test established

Against P-DET-COVERAGE-v2's three-model consensus reference (412 of 420 unanimous, 8 two-of-three, no
`NO_CONSENSUS`), the frozen classifier (tag `prose-decision-classifier-v2`, source sha256 LF `47436ca9…`):

| Mode | Rows | Verdict |
|---|---|---|
| DIRECT | recall 0.985, precision 0.929, challenge recall 0.982 | qualifies |
| UNSUPPORTED | recall 0.964, challenge recall 0.981 | qualifies |
| CLARIFY | f1 0.942, challenge recall 0.905 | qualifies |
| CALL | no CALL reference label exists in the population | `NOT_EVALUABLE`, does not qualify |

Global rows: macro F1 0.956, abstention rate 0.003. Post-stratified DIRECT precision permits **glaive** only;
**toolace** is `NOT_EVALUABLE` (11 DIRECT predictions, stratum X unsampled).

Every one of these qualifications is a `MODEL_REFERENCE` qualification, measured against three non-Claude
models agreeing, not against human gold. 35 §1 permits such a qualification to count toward balancing
permission **provisionally**, and that is the only footing it has here.

## 2. Decision A — balancing permission (provisional)

Granted, per mode:

- **UNSUPPORTED:** permitted, on both layer B sources.
- **CLARIFY:** permitted, on both layer B sources.
- **DIRECT:** permitted **for glaive only**. No balancing weight may depend on a DIRECT label on ToolACE
  records, because its post-stratified precision row is `NOT_EVALUABLE`, not passing. ToolACE DIRECT
  predictions are retained as labels but carry no balancing weight.
- **CALL:** **not permitted.** Its rows could not be evaluated, so 22 §6's failure behaviour applies: records
  may be retained, but no balancing weight may depend on a CALL label from this classifier.

The permission is **provisional** in the precise sense of 35 §1: it rests on a model reference. It is not
evidence that the classifier matches human judgement, and no report may describe it as accuracy.

## 3. Decision B — C1 authorised, with conditions

**C1 (the intervention of [21](21-C1-IMPLEMENTATION-STATUS.md)) is authorised to proceed** past its
classifier gate: phases 3–5 may be built, i.e. the deterministic versioned behaviour classifier wired into the
mixture machinery, a materialized balance, and new immutable canonical-v3 artifacts.

Conditions, all binding:

1. **One labeller.** Behaviour labels come only from the frozen `prose-decision-classifier-v2` through contract
   `prose-decision-input-v2` (and contract v1 where a record is a single exchange). No other classifier, no
   hand-editing of a label, no re-tuning: any rule change is a new classifier version and re-enters 22 §6.
2. **Provenance on every artifact.** Each canonical-v3 manifest records the classifier tag and source hash, the
   contract version, the metrics version, and the reference status `MODEL_REFERENCE_PROVISIONAL` with a pointer
   to 37 §8.
3. **Scope of weights.** DIRECT weights on glaive only; no CALL-dependent weight; UNKNOWN and abstentions
   never carry weight.
4. **Nothing historical is touched.** Study 001's artifacts, canonical-v1, canonical-v2, P-DET-v1,
   P-DET-COVERAGE-v1 and -v2, and every frozen manifest and hash stay byte-identical.
5. **Pre-GPU gates first.** Phase 6's validation gates must pass, and be recorded, before any training run.

## 4. What this does not authorise

- **No training run.** Authorising C1 permits building the mixture and the artifacts, not spending GPU time.
  Starting a training run remains a separate decision, to be taken with its cost and its evaluation plan in
  front of the study owner.
- **No threshold, population, contract, metric, arm, seed or partition change.** 22 §6, 30 §11 and
  `pdet-coverage-metrics-v1` are untouched.
- **No gold freeze.** P-DET-v1 stays unfrozen (35), and no model judgment becomes gold anywhere.
- **No claim of accuracy.** Every downstream document repeats that these qualifications are provisional and
  model-referenced.

## 5. What would withdraw these permissions

Named in advance, so that no later result can be read charitably after the fact:

1. **A human check that disagrees.** If a human pass over a sample of P-DET-COVERAGE-v2 puts any qualifying
   row below its threshold, that mode's balancing permission is withdrawn and C1 returns to the owner.
2. **A reference defect.** If the consensus reference is found to be built from an invalid package, or from
   labels that are not the three declared annotators', the test result is void and so is this decision.
3. **A classifier change.** Any edit to `decision_classifier_v2.py` voids the freeze, and with it this
   authorisation, until a new version is tested on a new untouched population (22 §6).
4. **A population defect.** If P-DET-COVERAGE-v2 is found to overlap the classifier's development data, the
   test is void.

## 6. Record

- Test result: `reports/prose-classifier/test-v2/prose-decision-classifier-v2.test-result.json`
  (predictions sha256 `a1aa1d03…`), commit a62c9ec.
- Reference: `reports/pdet-coverage-v2/reference/` (`MODEL_REFERENCE_PROVISIONAL`), commit 150f10f.
- Classifier: tag `prose-decision-classifier-v2`, commit ff1b306, source sha256 LF `47436ca9…`.
- Preregistration: recorded as `study_002_prereg_v7` in [03](03-PREREGISTRATION.md).
