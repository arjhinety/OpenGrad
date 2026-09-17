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

## 7. Record

Agreement with Claude model labels, not accuracy. "Agree" counts items the model gave a mode label; items it
labelled UNKNOWN are counted separately: on check set 1 the classifier abstained on all 25; on dev-v2 it
abstained on 48 of 52 and gave a mode to 4.

- **Development data.** `prose-classifier-dev-v2`: 300 first replies (sha `ec4c6756…`), labelled by session
  `model-dev-v2` (35 flagged). Round 0, the v1 rules unchanged, agreed on 234 / 248 items with a mode label
  (continuing conversations 132 / 137, single exchanges 102 / 111).
- **Round 1 rules** (commit fbbe5db, source sha256 LF `f2b99e17…`). Additions: capability wordings,
  statements that information is missing, two request question forms, "can't believe" as doubt, a short
  acknowledgement + "let me …" reply read as a narrated call, and a tool-limit tail after "but".
- **Check set 1.** `prose-classifier-v2-devcheck-1`: 150 first replies (sha `b4477eae…`, 85 continuing,
  65 single), drawn after the round-1 rules were committed. Labelled by session `model-v2-devcheck-1`
  (21 flagged, WIP export verify PASS), then scored **once**. As with v1's check sets, the subagents' final
  answers, one-sentence rationales included, reached the developer session before scoring. The developer did
  not print any item or disagreement.

| Round 1 rules | Agree | Rate | DIRECT predictions agreeing | Model DIRECT labels predicted DIRECT |
|---|---:|---:|---:|---:|
| dev-v2 (in-sample) | 245 / 248 | 0.988 | 117 / 119 | 117 / 117 |
| dev-v1, devcheck-v1, devcheck-v2 (in-sample) | 223 / 226, 110 / 110, 108 / 113 | | | |
| **Check set 1 (unexposed, scored once)** | **118 / 125** | **0.944** | **58 / 61** | **58 / 59** |

On check set 1, continuing conversations agreed 68 / 70 and single exchanges 50 / 55. The seven disagreements
by model label → prediction: UNSUPPORTED → DIRECT 2, UNSUPPORTED → CLARIFY 3, CLARIFY → DIRECT 1,
DIRECT → UNSUPPORTED 1. (Corrected 2026-09-17: the first version of this line read the confusion matrix
backwards for the three UNSUPPORTED → CLARIFY items, and the owner was told the same; the report was right.) Per mode against model labels: UNSUPPORTED 35 / 40, CLARIFY 25 / 26. The drop from
0.988 in-sample to 0.944 is the expected cost of tuning on the development items. As with v1, 61 DIRECT
predictions are far too few to estimate precision against the 0.80 threshold, model labels are not the
reference, and neither dev-v2 nor check set 1 has a model CALL label, so textual CALL is untested before the test.
Report: `reports/prose-classifier/dev-v2/prose-decision-classifier-v2.round1.v2-check-1-agreement.json`.

**Second round (decided by the study owner, 2026-09-17).** Offered freezing the round-1 rules or one more
round, the owner chose one more round, the last §4 allows. This is recorded before the developer reads any of
check set 1's disagreements. From here **`prose-classifier-v2-devcheck-1` is exposed**: it becomes development
data, and its 118 / 125 above is the only unexposed score it will ever give. Round-2 rules are committed before
`prose-classifier-v2-devcheck-2` (150 first replies, its own seed, excluding dev-v2, check set 1 and everything
they exclude) is drawn, then labelled the same way and scored once. After it, §4's stopping rule applies: the
owner chooses between freezing and a further round. (Corrected 2026-09-17: this line first said §4 allowed no
further round.)

**Round 2 rules** (source sha256 LF `0d5cce9a…`), made after reading check set 1's seven disagreements and
committed before check set 2 is drawn. Each change is a general wording or structure rule, tested on
paraphrased examples:

1. an ability counts as inability only when negated or limited ("have the ability to see" is not a decline);
2. tools that are lacking ("lacks the necessary functions", "without the appropriate functions"), tools that
   cannot do the task, and "none/neither of which are relevant" are capability wordings;
3. "I do not have (any specific) information about X" is a decline;
4. "you may need to consult …" is a referral elsewhere, and "the given question" is talk about the task;
5. "I'll need specific …" is a request;
6. "Here is why I cannot proceed:" introduces reasons, not a delivered answer;
7. in a decline with no capability gap, text between the first and last request (a list of the missing
   fields) belongs to the request, as step 4 already treats it.

One disagreement is left as a genuine boundary: a reply that the offered function "lacks the parameters"
while the request is also outside that function's scope (model label UNSUPPORTED, prediction CLARIFY).

| Round 2 rules, in-sample | Agree | Round 1 |
|---|---:|---:|
| dev-v2 | 245 / 248 | 245 / 248 |
| dev-v1 | 223 / 226 | 223 / 226 |
| devcheck-v1 | 110 / 110 | 110 / 110 |
| devcheck-v2 | 109 / 113 | 108 / 113 |
| v2 check set 1 (exposed) | 124 / 125 | 118 / 125 |

Across the 950 development items, eight predictions moved to agree with the model labels and none that agreed
moved away. The in-sample gain on check set 1 is expected and says nothing about unseen replies; check set 2
measures that.

- **Check set 2.** `prose-classifier-v2-devcheck-2`: 150 first replies (sha `8358c344…`, 87 continuing, 63
  single), drawn after the round-2 rules were committed (587098e). Labelled by session `model-v2-devcheck-2` (17
  flagged, WIP export verify PASS), then scored **once**, without printing any item or disagreement. The same
  disclosure applies: the subagents' final answers and rationales reached the developer session before scoring.

| Round 2 rules | Agree | Rate | DIRECT predictions agreeing | Model DIRECT labels predicted DIRECT |
|---|---:|---:|---:|---:|
| **Check set 2 (unexposed, scored once)** | **122 / 126** | **0.968** | **60 / 63** | **60 / 61** |
| Check set 1 (round 1, for comparison) | 118 / 125 | 0.944 | 58 / 61 | 58 / 59 |

On check set 2, continuing conversations agreed 71 / 73 and single exchanges 51 / 53. The four disagreements by
model label → prediction: CLARIFY → DIRECT 2, UNSUPPORTED → DIRECT 1, DIRECT → UNSUPPORTED 1. Per mode against
model labels: UNSUPPORTED 35 / 36, CLARIFY 26 / 28, CALL 1 / 1 (the first textual call in any v2 set). Of 24
items the model labelled UNKNOWN, the classifier abstained on 23 and predicted UNSUPPORTED on 1. The same limits
hold: agreement with one Claude labeller, not accuracy; 63 DIRECT predictions cannot settle precision against the
0.80 threshold; one CALL label says almost nothing about CALL precision's 0.95 threshold. Report:
`reports/prose-classifier/dev-v2/prose-decision-classifier-v2.round2.v2-check-2-agreement.json`.

**Third round (decided by the study owner, 2026-09-17).** After both check rounds, offered freezing the round-2
rules or a further round, the owner chose a further round under §4's stopping rule. Recorded before the developer
reads any of check set 2's disagreements. From here **`prose-classifier-v2-devcheck-2` is exposed** and becomes
development data; its 122 / 126 above is the only unexposed score it will ever give. Round-3 rules are committed
before `prose-classifier-v2-devcheck-3` (150 first replies, its own seed, excluding dev-v2, check sets 1 and 2
and everything they exclude) is drawn, labelled the same way and scored once; then the owner chooses again.

**Round 3 rules** (source sha256 LF `39494c4f…`), made after reading check set 2's four disagreements and the one
UNKNOWN item predicted UNSUPPORTED, committed before check set 3 is drawn. General rules, tested on paraphrases:

1. "lacks a function …" and "cannot be processed using the given functions" are capability wordings;
2. a concessive disclaimer ("Though I can't give medical advice, it's a good idea to …") is read like a "but"
   clause, so the answer after it counts as delivered;
3. "the functions cannot be called / used" for want of arguments is no longer a capability gap. This narrows
   round 2's rule 2, which had caught it;
4. "The function that retrieves X is …" is talk about the tools, not content;
5. "Are you interested in / looking for / asking about …?" is a request.

| Round 3 rules, in-sample | Agree | Round 2 |
|---|---:|---:|
| dev-v2 | 245 / 248 | 245 / 248 |
| dev-v1 | 223 / 226 | 223 / 226 |
| devcheck-v1 | 110 / 110 | 110 / 110 |
| devcheck-v2 | 109 / 113 | 109 / 113 |
| v2 check set 1 (exposed) | 124 / 125 | 124 / 125 |
| v2 check set 2 (exposed) | 126 / 126 | 122 / 126 |

Across the 1,100 development items, four predictions moved to agree with the model labels, none that agreed moved
away, and one UNKNOWN item moved from UNSUPPORTED to CLARIFY (not counted either way).

- **Check set 3.** `prose-classifier-v2-devcheck-3`: 150 first replies (sha `7e541757…`, 85 continuing, 65
  single), drawn after the round-3 rules were committed (944c5a7). Labelled by session `model-v2-devcheck-3` (26
  flagged, WIP export verify PASS), then scored **once** without printing any item or disagreement. Same
  disclosure as before about subagent rationales.

| Unexposed check, scored once | Rules | Agree | Rate | DIRECT predictions agreeing | Model DIRECT labels predicted DIRECT |
|---|---|---:|---:|---:|---:|
| Check set 1 | round 1 | 118 / 125 | 0.944 | 58 / 61 | 58 / 59 |
| Check set 2 | round 2 | 122 / 126 | 0.968 | 60 / 63 | 60 / 61 |
| **Check set 3** | **round 3** | **114 / 122** | **0.934** | **58 / 62** | **58 / 60** |

On check set 3, continuing conversations agreed 68 / 71 and single exchanges 46 / 51. The eight disagreements by
model label → prediction: DIRECT → UNSUPPORTED 1, DIRECT → abstain 1, CLARIFY → DIRECT 2, UNSUPPORTED → DIRECT 2,
UNSUPPORTED → CLARIFY 2. Per mode against model labels: DIRECT 58 / 60, CLARIFY 21 / 23, UNSUPPORTED 35 / 39; no
CALL label. Of 28 items the model labelled UNKNOWN, the classifier abstained on 24 and predicted CLARIFY on 4.

The three unexposed scores (0.944, 0.968, 0.934) come from different sets of about 125 items each; differences
of a few items are within what sampling alone produces, so they show no clear trend across rounds. Pooled, the
three rounds agree on 354 / 373 (0.949), with DIRECT predictions agreeing 176 / 186 (0.946). Each round fixed
every disagreement it read without moving any development item away from its label, yet unseen agreement stayed
near 95%: the remaining errors are different on each new set. Report:
`reports/prose-classifier/dev-v2/prose-decision-classifier-v2.round3.v2-check-3-agreement.json`.

**Fourth round (decided by the study owner, 2026-09-17).** Offered freezing the round-3 rules or a further round,
with the developer recommending a freeze, the owner chose a further round. Recorded before the developer reads any of
check set 3's disagreements. From here **`prose-classifier-v2-devcheck-3` is exposed** and becomes development
data; its 114 / 122 above is the only unexposed score it will ever give. Round-4 rules are committed before
`prose-classifier-v2-devcheck-4` (150 first replies, its own seed, excluding dev-v2, check sets 1-3 and everything
they exclude) is drawn, labelled the same way and scored once; then the owner chooses again.

**Round 4 rules** (source sha256 LF `309ab4df…`), made after reading check set 3's eight disagreements and the
four UNKNOWN items predicted CLARIFY, committed before check set 4 is drawn. General rules, tested on paraphrases
that fail on the round-3 rules:

1. list items introduced by a sentence ending in ":" that talks about the task, the tools or a decline restate
   what was asked or is missing; they are part of that sentence, not delivered content;
2. "Once I have that information, I can …" is part of the request;
3. text after a rewrite label ("Corrected sentence:") is content, even when it reads like a decline;
4. a worked result ("gamma(3) = 2") is not a narrated call;
5. "you may need to look up / use …" is a referral; "only deal with / work with / handle" is a scope limit;
6. "the functions cannot be **used** to achieve the purpose" is a capability gap again. Round 3's exclusion
   (rule 3) had wrongly caught "used" along with "called".

**Tried and dropped.** A rule to abstain when no tool is offered and the reply asks for a named function's
parameters (the four UNKNOWN items) moved seven items the model had labelled CLARIFY away from their label. It
gained eight abstentions on UNKNOWN items, which are not counted either way. The model's own labels do not separate
the two cases, so the rule was removed.

| Round 4 rules, in-sample | Agree | Round 3 |
|---|---:|---:|
| dev-v2 | 245 / 248 | 245 / 248 |
| dev-v1 | 225 / 226 | 223 / 226 |
| devcheck-v1 | 110 / 110 | 110 / 110 |
| devcheck-v2 | 109 / 113 | 109 / 113 |
| v2 check set 1 (exposed) | 124 / 125 | 124 / 125 |
| v2 check set 2 (exposed) | 126 / 126 | 126 / 126 |
| v2 check set 3 (exposed) | 122 / 122 | 114 / 122 |

Across the 1,250 development items, ten predictions moved to agree with the model labels and none that agreed moved
away. The four UNKNOWN items stay predicted CLARIFY.

- **Check set 4.** `prose-classifier-v2-devcheck-4`: 150 first replies (sha `817ab7cf…`, 89 continuing, 61
  single), drawn after the round-4 rules were committed (5943983). Labelled by session `model-v2-devcheck-4` (16
  flagged, WIP export verify PASS), then scored **once** without printing any item or disagreement. Same
  disclosure about subagent rationales.

| Unexposed check, scored once | Rules | Agree | Rate | DIRECT predictions agreeing | Model DIRECT labels predicted DIRECT |
|---|---|---:|---:|---:|---:|
| Check set 1 | round 1 | 118 / 125 | 0.944 | 58 / 61 | 58 / 59 |
| Check set 2 | round 2 | 122 / 126 | 0.968 | 60 / 63 | 60 / 61 |
| Check set 3 | round 3 | 114 / 122 | 0.934 | 58 / 62 | 58 / 60 |
| **Check set 4** | **round 4** | **122 / 124** | **0.984** | **63 / 63** | **63 / 63** |
| All four | | 476 / 497 | 0.958 | 239 / 249 | 239 / 243 |

On check set 4, continuing conversations agreed 75 / 75 and single exchanges 47 / 49. The two disagreements by model
label → prediction: CLARIFY → UNSUPPORTED 1, UNSUPPORTED → CLARIFY 1. Per mode against model labels: DIRECT 63 / 63,
CLARIFY 27 / 28, UNSUPPORTED 32 / 33; no CALL label. Of 26 items the model labelled UNKNOWN, the classifier abstained
on 24 and predicted DIRECT on 1 and CLARIFY on 1. No false DIRECT among the items with a mode label.

Check set 4 is the best single result, but one set of 124 cannot separate a real gain from a favourable draw: at
the pooled rate of about 96%, a set of this size would show 2 or fewer disagreements by chance a noticeable fraction
of the time. The pooled figures are the steadier estimate. Report:
`reports/prose-classifier/dev-v2/prose-decision-classifier-v2.round4.v2-check-4-agreement.json`.

**Fifth round (decided by the study owner, 2026-09-17).** Offered freezing the round-4 rules or a further round,
with the developer recommending a freeze, the owner chose a further round. Recorded before the developer reads any of
check set 4's disagreements. From here **`prose-classifier-v2-devcheck-4` is exposed** and becomes development
data; its 122 / 124 above is the only unexposed score it will ever give. Round-5 rules are committed before
`prose-classifier-v2-devcheck-5` (150 first replies, its own seed, excluding dev-v2, check sets 1-4 and everything
they exclude) is drawn, labelled the same way and scored once; then the owner chooses again.
