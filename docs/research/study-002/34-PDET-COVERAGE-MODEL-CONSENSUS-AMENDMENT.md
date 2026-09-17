# 34 — P-DET-COVERAGE-v1 amendment: a three-model consensus reference (`study_002_prereg_v5`)

**Status: AMENDMENT, 2026-09-17, decided by the study owner (arjhinety), before any P-DET-COVERAGE-v1 label
exists.** No annotation store for `pdet-coverage-v1` or `pdet-coverage-v1-routing` has been created, and no
item has been labelled by anyone. This file changes **who produces the reference labels** of
P-DET-COVERAGE-v1 ([30](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md), adopted as `study_002_prereg_v4`). It
does not change the population file `reports/pdet-coverage/pdet-coverage-v1.population.jsonl`
(`population_sha256` `755bc16e79ceb1cb9e461c0fe8b9125628c16f263ed24f9e008f55b613a7158c`, 336 rows).

## 1. The decision

30 §10 said: "Gold is human only." The study owner will not annotate, and no other human annotator is
available, so that rule cannot be met. The reference labels are therefore produced by three declared
**non-Claude** model annotators, each labelling every item independently:

| Annotator id | Model, as pinned | Run through |
|---|---|---|
| `model.gemini-3.8-flash-high` | Gemini 3.8 Flash (High) | Antigravity CLI: `agy --model "Gemini 3.8 Flash (High)" -p` |
| `model.gpt-5.6-sol` | gpt-5.6-sol | Codex CLI: `codex exec -m gpt-5.6-sol` |
| `model.deepseek-v4.1-flash` | deepseek-v4.1-flash | Cline CLI: `cline -m cline-pass/deepseek-v4.1-flash` |

**The reference label of an item** is the label that at least two of the three give (`gold_policy_label`,
one of CALL, DIRECT, CLARIFY, UNSUPPORTED, UNKNOWN).
- **No majority:** when all three differ, the item is `NO_CONSENSUS`. It is counted and reported, and it is
  left out of every metric and every minimum-size count.
- **Majority UNKNOWN:** an item whose majority label is UNKNOWN stays UNKNOWN, exactly as a human UNKNOWN
  would, so it is not a mode example.

## 2. Why these three, and why not Claude

- **Independence from the classifier's developer.** Claude Opus 5 builds the prose decision classifier and
  labels its development set ([33](33-PROSE-DECISION-CLASSIFIER-DEVELOPMENT-PLAN.md)). If Claude also made
  the reference, agreement between the two could reflect one model's shared habits rather than correctness.
  28 item 3 already names this `NON_INDEPENDENT`. So no Claude model labels this population and none breaks
  ties. The Antigravity CLI also offers Claude models, so its Gemini model is pinned explicitly.
- **Three families rather than one.** A single model's mistakes cannot be told apart from the truth. With a
  two-of-three majority, one model's quirk cannot decide a label on its own, and the disagreement rate is a
  measured statement of how reliable the reference is.

## 3. How labelling is run

- **The audited route.** The same route as P-DET-v1's model labels (28): `opengrad-annotate model-batch`
  renders each batch exactly as the annotation screen shows items, with no blinded field; `model-ingest`
  checks every answer against the task's value rules and records it as a **model judgment** under the
  declared annotator id. Each task config declares the three annotators, with the procedure file pinned by
  SHA-256.
- **Isolation.** Each external CLI runs in an empty scratch directory that holds only its batch file, with
  its tools restricted, so it cannot read the repository, the population file or the manifest. The source,
  stratum, layer and trajectory-gate status therefore stay hidden, as 30 §10 requires.
- **Independence between annotators.** No annotator sees another's labels.
- **The developer does not read the items.** Claude orchestrates the batches but, as the classifier's
  developer, does not read their item text or the answers' rationales (22 §7, 30 §9). Only labels and value
  fields are checked, by `model-ingest`.
- **Provenance.** Every batch, raw answer, procedure and CLI version is archived with a hash manifest, as
  for `model-a`.

## 4. What changes for claims

1. **Model labels are not human gold.** Every metric on P-DET-COVERAGE-v1 names its reference as "three-model
   consensus (Gemini 3.8 Flash High, gpt-5.6-sol, deepseek-v4.1-flash)". Nothing computed on it may be called
   human-validated.
2. **Qualifications are provisional.** A classifier qualification measured against this reference is
   recorded as `MODEL_REFERENCE`. As under 28 item 2, this amendment does not decide whether a
   `MODEL_REFERENCE` qualification may grant balancing permission. The study owner records that decision
   separately, before it is used.
3. **Agreement is model–model agreement.** Report, with item counts: how many items all three agreed on,
   how many had a two-of-three majority, how many are `NO_CONSENSUS`, and each pair's agreement. It is never
   called inter-annotator agreement, which refers to human annotators.
4. **Human labels take precedence if they ever exist.** Any later human label replaces the consensus label
   for its item, and qualifications are re-evaluated.

## 5. What does not change

- **Unchanged:**
  - the population, its order, ids and contents;
  - the two tasks and their split by layer;
  - the rubric (22 §1–§4 with 30 §2–§3);
  - the fields and their values;
  - the blinding of source, stratum, layer, gate status and match kind;
  - the acceptance rules and minimum sizes of 30 §11, now counted on consensus items;
  - the 22 §6 prohibition on iterating a classifier against validation results;
  - P-DET-v1 and its human labels.
- **Shortfalls:** a stratum or boundary that falls below its minimum because of `NO_CONSENSUS` items is
  `NOT_EVALUABLE`, never topped up (30 §11, unchanged).
- **Scored so far:** nothing. No label exists and no classifier exists.
- **Nature:** a resource-driven change made before any label or result exists. It is not a defect repair.
