# 37 — How `prose-decision-classifier-v2` is developed

**Status: PLAN, 2026-09-17. Written under [36](36-FIRST-REPLY-CONTRACT-AND-PDET-COVERAGE-V2-DRAFT.md) §4–§5
(`study_002_prereg_v6`) after P-DET-COVERAGE-v2 was drawn and before any v2 development set, v2 rule or result
exists.** It follows [33](33-PROSE-DECISION-CLASSIFIER-DEVELOPMENT-PLAN.md), the plan v1 was built under, and
records only what differs. It changes no threshold, population or acceptance rule.

## 1. What v2 is

- **Same job, wider input.** It labels a reply as DIRECT, CLARIFY, UNSUPPORTED, CALL (a textual call) or abstains,
  from the four features of contract `prose-decision-input-v2`: the first reply of any record (36 §2).
- **Starts from frozen v1.** The v1 rules are copied into a new module,
  `src/opengrad/data/decision_classifier_v2.py`, with the version string `prose-decision-classifier-v2`. The frozen
  `decision_classifier.py` and its tag stay untouched, so v1's result remains reproducible.
- **Known targets:** false DIRECT on ToolACE and When2Call single exchanges (33 §8 caveat 4; 35 §5), and whatever
  first replies of continuing conversations show (they were never development data for v1).

## 2. Exposure, disclosed before development starts

- **Exposed and never gating v2:** P-DET-v1 (the developer read its items as `model-a`) and P-DET-COVERAGE-v1 (v1's
  test results per row, and per-stratum label counts in 35 §4–§5).
- **Aggregates the developer has seen:** v1's predictions counted per stratum and source over the unused
  first-reply pool (36 §3, sizing). Those counts cover items later drawn into P-DET-COVERAGE-v2, as sums only.
- **Never read:** any item, label or rationale of P-DET-COVERAGE-v2. Its reference is built by the three external
  models, and the developer sees counts only.

## 3. Development data

1. **The earlier development sets are kept as development data:** `prose-classifier-dev-v1` (250),
   `-devcheck-v1` (125) and `-devcheck-v2` (125), with their Claude labels. They are single exchanges, so they also
   guard against v2 losing what v1 got right. Their status as check sets ended with v1.
2. **A new set, `prose-classifier-dev-v2`,** of first replies:
   - unit and pool: contract v2, the layer B sources (Glaive, ToolACE);
   - excluded before drawing: 30 §9's exclusions, held-out material, and every item of P-DET-COVERAGE-v2,
     P-DET-COVERAGE-v1 and the three earlier development sets, by identity, normalized prompt and normalized
     response;
   - 300 items, 50 per stratum of 30 §7.2, split equally across sources, seed `opengrad-prose-classifier-dev-v2`,
     30 §9's dedup; a short stratum is reported, not backfilled;
   - reported per `unit_kind`, which is never a rule input.
3. **Labels:** Claude Opus 5 subagents, as in 33 §4, session `model-dev-v2`, under a new pinned procedure: the
   development procedure plus 36 §3.7's instruction to judge the reply as the answer to that one message. Model
   judgments, never gold, archived with a hash manifest.

## 4. Rounds, checks and freezing

- **Development runs** against the development data as often as needed, reported as agreement with model labels.
- **Check sets:** before each freeze decision the committed candidate rules are scored **once** on a fresh check
  set of first replies, `prose-classifier-v2-devcheck-N`: 150 items, 25 per stratum, its own seed, excluding
  everything above and every earlier check set, labelled the same way, never printed to the developer. A check
  set read after scoring becomes development data, as in 33 §5a.
- **Stopping rule:** at most two check rounds, then the study owner chooses freezing or a further round, as for v1.
- **Freeze:** commit and tag `prose-decision-classifier-v2`; a one-shot runner pinned to its source hash that
  refuses any other.
- **Guards, as for v1:** a test fails if the v2 module reads any population or annotation path, and the runner
  checks the frozen hash before scoring.

## 5. The test

Run once, after P-DET-COVERAGE-v2's consensus reference exists, with `pdet-coverage-metrics-v1` unchanged:

- **Gating:** P-DET-COVERAGE-v2 only, reported as `MODEL_REFERENCE` (34 §4; 35 §1 allows it to count toward
  balancing permission, provisionally). Challenge rows use strata R, Q, M and X. Results are also given per
  `unit_kind`, never gated.
- **Reported, never gating:** P-DET-v1 and P-DET-COVERAGE-v1 rows, marked `DEVELOPMENT_EXPOSED`.
- **C1** remains the owner's decision even if DIRECT and UNSUPPORTED qualify (35 §1).

## 6. What this plan does not do

- It changes no threshold, population, contract or rule.
- It authorises no C1 balancing, mixture change or training.
- It creates no gold label.
