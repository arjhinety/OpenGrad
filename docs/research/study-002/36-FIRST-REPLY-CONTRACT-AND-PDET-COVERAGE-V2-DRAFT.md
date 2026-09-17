# 36 — Amendment: the first-reply input contract, P-DET-COVERAGE-v2 and classifier v2

**Status: ADOPTED as written, 2026-09-17, by the study owner, as `study_002_prereg_v6`.** Adopted before
anything was implemented, drawn or labelled under it. The file name keeps `-DRAFT` so the paths already cited
stay valid. Written at the study owner's direction after [35](35-OWNER-DECISIONS-AFTER-CLASSIFIER-V1-TEST.md)
§6 (decision: "admit first replies"). Its supply figures come from a counts-only dry run,
`scripts/audit_first_reply_supply.py` → `reports/pdet-coverage-v2/first-reply.supply-dry-run.json`.

It changes, on adoption:
- the classifier input contract, from `prose-decision-input-v1` ([32](32-CLASSIFIER-INPUT-CONTRACT.md)) to
  `prose-decision-input-v2`;
- the validation population for a new classifier version, P-DET-COVERAGE-v2;
- the classifier version, `prose-decision-classifier-v2`.

It changes no threshold, minimum size or acceptance rule of [22](22-PDET-PROTOCOL.md) §6 and
[30](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md) §11, nor any frozen artifact.

## 1. Why

C1 needs the classifier to qualify for DIRECT and UNSUPPORTED (22 §6). DIRECT could not qualify for v1, and no
single-exchange population can fix that: about 22 DIRECT items are expected in the whole unused single-exchange
pool (35 §4). But Study 001's corpus holds thousands of direct answers, as the **first reply of conversations that
continue** (35 §6), a shape contract v1 excludes as multi-turn.

A first reply is made with exactly the context a single exchange has: one user message and the offered tools.
Nothing earlier exists, and what comes later cannot have influenced it. So the decision is the same kind of
decision v1 was built for. The multi-turn concern of 32 §6, that dropping earlier turns changes the decision,
does not arise, because there are none. Later turns remain out of scope.

## 2. `prose-decision-input-v2`

**Unit.** The **first assistant reply** of a normalization-v3 record: the first assistant turn, directly after
exactly one user turn (a leading system message allowed), carrying non-empty text and no structured call,
**whatever follows it**. A v1-eligible single exchange is the special case with nothing after it.

**Features: unchanged** (32 §2). `user_message` is that one user turn, `assistant_response` is the first reply,
`tools` are the record's offered tools, `structured_call_present` is `false`. The system message stays hidden.

**Never read, in addition to 32 §3:**
- any turn after the first reply;
- whether the record continues, and how. It is recorded in provenance as `unit_kind` (`single_exchange` or
  `has_continuation`), never as a feature: continuation is far more common in Glaive (55,394 of its 69,697
  first-reply units) than in ToolACE (57 of 2,029), so it would identify the source.

**Eligibility, in precedence order:**

| Reason | Applies when |
|---|---|
| `EVALUATION_ONLY_OR_HELDOUT` | as 32 §4, over every turn of the record |
| `MALFORMED_OR_UNRENDERABLE` | as 32 §4 over the whole record, **except** the two checks on the final message (final role, empty final assistant turn), which a first reply does not depend on. A trajectory defect anywhere still excludes the record |
| `FIRST_TURN_NOT_ONE_USER_THEN_ASSISTANT` | the body does not start with one user turn followed by an assistant turn |
| `FIRST_REPLY_IS_STRUCTURAL_CALL` | the first reply carries structured calls (layer A, never the prose classifier) |
| `FIRST_REPLY_EMPTY` | the first reply has no text |

**Serialization.** As 32 §7, with `contract_version: prose-decision-input-v2`, so v1 and v2 feature hashes never
coincide. A pinned-hash test, as for v1.

**Still ineligible:** post-tool responses and every later turn. Each would need its own contract.

**Measured on normalization-v3 (dry run).** Glaive: 69,697 first-reply units (14,303 single exchanges, 55,394
with a continuation), 27,770 first replies that are structured calls, 864 malformed, 8 held-out. ToolACE: 2,029
units (1,972 and 57), 8,970 structured calls, 52 malformed.

## 3. P-DET-COVERAGE-v2

A constructed boundary-coverage population, drawn like P-DET-COVERAGE-v1 (30 §7–§9) with these differences.

1. **Unit:** the first reply (§2). Layer B only. Structural routing was audited on P-DET-COVERAGE-v1 layer A and
   is not repeated.
2. **Sources:** Glaive and ToolACE, as 30 §6. When2Call stays out (P-DET-v1 covers it).
3. **Exclusions, beyond 30 §9:** every item of P-DET-COVERAGE-v1 and of `prose-classifier-dev-v1`,
   `-devcheck-v1` and `-devcheck-v2`, by identity, normalized prompt and normalized response.
4. **Dedup, strata, ranking and allocation:** 30 §7.2, §8 and §9 unchanged, with a new seed
   `opengrad-pdet-coverage-002-v2`. One item per user prompt across the population.
5. **Quotas** (below). No backfill; a shortage is reported before annotation.
6. **Blinding:** as 30 §10, and additionally the continuation and `unit_kind` are never shown or filterable.
   Annotators see the user message, the offered tools and the first reply only.
7. **Reference:** a two-of-three consensus of the three non-Claude annotators of 34, under the same procedure,
   with one added instruction: the reply is judged as the assistant's answer to that one message.
8. **Metrics:** `pdet-coverage-metrics-v1` unchanged: minimum 50 usable gold per row, 30 per challenge row
   (strata R, Q, M, X), the frozen 22 §6 thresholds, DIRECT balancing per source only where pool-weighted
   precision passes, C1 needing DIRECT and UNSUPPORTED. Results are reported per `unit_kind` too, never gated.

**Proposed quotas.** Allocation splits each stratum's quota equally across sources that have supply (30 §7.3).

| Stratum | Quota | Glaive / ToolACE | Unused supply after dedup, Glaive / ToolACE |
|---|---:|---|---|
| X | 40 | 40 / 0 | 300 / 0 |
| M | 100 | 50 / 50 | 292 / 609 |
| R | 80 | 40 / 40 | 223 / 166 |
| Q | 100 | 50 / 50 | 11,324 / 143 |
| P1 | 60 | 30 / 30 | 705 / 212 |
| P2 | 40 | 20 / 20 | 23,418 / 414 |
| **Total** | **420** | | |

**How the quotas were sized.** Each mode must reach 50 usable reference items, 30 of them in R, Q, M or X, after
losing some items to UNKNOWN and no consensus (7.5% UNKNOWN on P-DET-COVERAGE-v1 layer B). Two estimates:

- **Frozen classifier v1 predictions on the unused supply, aggregated per stratum and source (sizing only; no
  item is selected by prediction).** At these quotas they imply about 208 DIRECT (156 in challenge strata),
  117 UNSUPPORTED (87) and 78 CLARIFY (74). CLARIFY is the binding mode, which is why M and Q are 100.
- **Reference yields where the unit is unchanged.** ToolACE single exchanges make up most of ToolACE's quota
  (continuations are 55 of its 1,544 unused units, 45 of them in Q), and P-DET-COVERAGE-v1 labelled the same
  kind: M 38 of 49 CLARIFY; Q 11 of 30 CLARIFY and 15 of 30 UNSUPPORTED; R 30 of 30 UNSUPPORTED; P1 72 of 74
  UNSUPPORTED. v1's predicted shares on the unused ToolACE supply are close (M 76% CLARIFY; Q 39% CLARIFY and
  52% UNSUPPORTED; R 95% UNSUPPORTED), except P1, where it predicts 83% UNSUPPORTED against the reference's 97%.
  No reference label exists yet for any Glaive first reply with a continuation.

**Disclosure.** The developer of classifier v2 computed these aggregates and chose the quotas. No item of the
future population was read, and the draw is fixed by seed and rule before anyone sees it. The risk is that v1's
view of the pool shapes the mix; the stratum design (30 §7.2) and the challenge rows exist to limit exactly that.

## 4. `prose-decision-classifier-v2`

- **Starts from v1.** Its known weakness is false DIRECT on ToolACE and When2Call (33 §8, 35 §5).
- **Exposure.** P-DET-v1 and P-DET-COVERAGE-v1 are `DEVELOPMENT_EXPOSED` for v2 (35 §2) and never gate it. The
  developer has seen v1's test results, the label counts per stratum in 35 §4–§5 and this dry run's
  aggregates; it has read no item of P-DET-COVERAGE-v2 and will not.
- **Development data:** a new development set and held-out check sets drawn from the first-reply unit, excluding
  P-DET-COVERAGE-v2 and everything above, labelled and archived as in 33 §3–§5a. A separate plan document fixes
  them before any is drawn.
- **Freeze before the test,** as v1: commit, tag, a runner that refuses any other source hash. Tested once on
  P-DET-COVERAGE-v2.

## 5. Order of work, each step committed before the next

1. The owner adopts this draft, with or without changes (an engineering review first, as for 30, is optional).
2. Contract v2 in code, with tests (`classifier_input.py`, version constant). v1 stays callable and unchanged.
3. The P-DET-COVERAGE-v2 builder, with `--dry-run`, committed before the draw; then the draw.
4. One annotation task (`pdet-coverage-v2`) configured, the three external annotators run, the consensus
   reference built and archived. The developer sees counts only.
5. The v2 development plan, development set, labels, rounds and check sets; then the freeze.
6. The one-shot test of v2 on P-DET-COVERAGE-v2.

Steps 4 and 5 can run side by side, because the developer never reads the reference.

## 6. What this draft does not decide

- **How C1 would reweight.** A record whose first reply is DIRECT also carries its later turns, including tool
  calls, and weighting the record weights all of them. C1's design has to settle that before any balancing.
- **Later turns and post-tool responses**, which hold most of the corpus's prose (110,923 later prose turns in
  canonical-v2-final, 35 §6).
- **What explains Study 001's regression.** 35 §6 showed missing direct answers do not.
- **P-DET-v1's freeze**, which stays optional.

## 7. Record of work under this amendment

- **Contract v2 in code** (§5 step 2): `first_reply_eligibility` and `build_first_reply_input` in
  `src/opengrad/data/classifier_input.py`, tests `tests/data/test_classifier_input_v2.py`.
- **Builder committed before the draw** (§5 step 3): `src/opengrad/verification/pdet_coverage_v2.py`, with its
  counts-only dry run `reports/pdet-coverage-v2/pdet-coverage-v2.dry-run.json`.
- **Drawn 2026-09-17:** `reports/pdet-coverage-v2/pdet-coverage-v2.population.jsonl`, 420 items, population sha256
  `8fa4868c64845a93b0627cd4463887b03181e6479f2e9f48d75e8c80f626f013`, equal to the dry run's. Every stratum met
  its quota (no shortage). 210 items are single exchanges and 210 have a continuation. Of 71,726 first-reply
  units, 12,120 records matched the QAD recovery set and 2,581 matched an earlier population or development set.
  `--verify` re-derives the bytes and checks contamination: PASS. No item was read.

