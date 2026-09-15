# 22 — P-DET protocol (frozen)

**Protocol version:** `pdet-002-v1`. **Status: `FROZEN`** for the specification; the **gold population is
`BLOCKED` pending human annotation** (see §7 and [24](24-PHASE-3-REPORT.md)).

P-DET answers exactly one question, and only that question:

> Can a deterministic prose classifier distinguish the Study 002 policy modes with sufficient reliability to
> let its inferred labels influence corpus construction and behaviour-balanced sampling?

P-DET **does not** validate a trained model, does not measure Study 002's hypothesis, and is not evidence
that any model policy improved. It validates an ETL classifier, and a pass grants one permission only: that
heuristic labels may be used to build a corpus.

This contract was written **before** the final validation examples were selected.

## 1. Policy modes (operational definitions)

The annotated quantity is **the intended behaviour represented by the assistant response, in context** — not
surface punctuation, not keywords, and not the word the upstream dataset used. A response ending in `?` is
not automatically `CLARIFY`; a response containing refusal-like language is not automatically `UNSUPPORTED`.

### 1.1 `CALL`

**Intended behaviour:** invoke an available tool.

- **Positive:** a machine-readable call payload is present — structured `tool_calls` on the assistant turn,
  a serialized JSON call (`{"name": …, "arguments": …}`), or an equivalent explicit call block.
- **Exclusion → `CLARIFY`:** no payload, and the response asks the user for something required before it can
  act.
- **Exclusion → `AMBIGUOUS`:** no payload, and the response only *describes*, *proposes* or *narrates* a
  call ("I would use `get_weather` here", "the right tool for this is X") without asking permission and
  without a payload. **Prose about a call is not a call.**
- **Exclusion → `AMBIGUOUS`:** the response names a tool that is **not** in the offered tool list.
- **Boundary:** "Shall I go ahead and call it?" — ceding control to the user *is* a request required to
  proceed → `CLARIFY`, not `CALL`.

### 1.2 `DIRECT` (canonical `ANSWER`)

**Intended behaviour:** respond without a tool, without requesting missing information, and without declining
capability.

- **Positive:** substantive content that answers or performs the user's request in prose, including
  reasoning, code, lists, enumerations, or an ordinary conversational reply.
- **Exclusion → `CLARIFY`:** the substantive content is *withheld* pending information only the user can
  supply, and the task cannot be completed correctly without it.
- **Exclusion → `UNSUPPORTED`:** the response declines, or explains that it cannot act.
- **Exclusion → `AMBIGUOUS`:** empty, truncated, or non-substantive (only a marker or an ellipsis).
- **Tolerated inside `DIRECT`:** a *courtesy* question after a complete answer ("Does that help?", "Want me
  to expand on step 3?"), rhetorical questions, an epistemic hedge embedded in a delivered answer, and a
  partial answer delivered with a caveat. Completeness is not required; **delivery** is.

### 1.3 `CLARIFY`

**Intended behaviour:** request information necessary to proceed correctly.

- **Positive:** asks for a missing *required* element (a tool parameter, an identifier, a subject, a
  destination) **or** asks to disambiguate genuinely ambiguous intent where proceeding risks a wrong action.
  The request must be *necessary* for correct completion.
- **Exclusion → `DIRECT`:** the question is optional, stylistic or social, and the answer was delivered.
- **Exclusion → `UNSUPPORTED`:** the response asks the *user* to perform the task, or to use an external
  service, as a substitute for doing it. That is a decline, however politely phrased.
- **Exclusion → `AMBIGUOUS`:** several questions are asked and none is demonstrably required.
- **Rule when mixed:** if one request is required and others are courtesy, the required request dominates →
  `CLARIFY`, and the rationale records which request was required.
- **Boundary vs `UNSUPPORTED`:** inability *because a tool or capability is absent* is `UNSUPPORTED` even
  when phrased interrogatively ("Do you have a tool that can read VINs?"), because the missing thing is
  capability, not user input.

### 1.4 `UNSUPPORTED`

**Intended behaviour:** the request cannot appropriately be performed with the available capabilities, so the
correct behaviour is to decline or explain the limitation.

- **Positive:** explicit inability or decline; refusal-shaped prose ("Apologies, but I'm unable to…",
  "I don't have the capability to…", "I can't access real-time data"); parking the task on an external
  service *instead of* performing it; stating the request falls outside the offered tools or scope.
- **Exclusion → `CLARIFY`:** the only missing thing is information the user could supply.
- **Exclusion → `DIRECT`:** an epistemic hedge inside an otherwise delivered answer ("I can't know your exact
  location, but at latitude X the answer is Y").
- **Exclusion → `DIRECT`:** a partial answer plus a caveat, where substantive content was delivered.
- **Boundary vs epistemic uncertainty:** *uncertainty about the world* is not tool-policy inability. "I'm not
  certain of today's rate" is not `UNSUPPORTED` on that basis alone; "I can't retrieve today's rate" is.
- **Boundary:** "I can't perform calculations" is `UNSUPPORTED` even though arithmetic is within a general
  model's ability — it is a policy-level inability to act, which is what the class is for.

### 1.5 `UNKNOWN` / `AMBIGUOUS` — an annotation status, not a mode

Assigned when the rubric cannot resolve the item to exactly one of the four modes: two modes are equally
supported; the response is empty, truncated or non-substantive; the intent is unreadable without context the
frozen record does not contain; or the offered tool list is empty so `CALL`-vs-`DIRECT` is undecidable.

It is **never** used to reach balance, and ambiguous material is **never** relabelled to complete coverage. It
is counted and reported, and excluded from per-class precision/recall denominators while remaining visible as
a human-ambiguity rate.

## 2. The eight boundary discriminations

| # | Discrimination | Resolving question (semantic, in this order) |
|---|---|---|
| 1 | `DIRECT` vs `CLARIFY` | Was substantive content delivered? If yes → `DIRECT`. If withheld pending user-supplied information → `CLARIFY`. |
| 2 | `CLARIFY` vs `UNSUPPORTED` | Is the missing thing *user input* → `CLARIFY`; is it *capability/tool/scope* → `UNSUPPORTED`. |
| 3 | `UNSUPPORTED` vs epistemic uncertainty | Does the response decline to **act**, or merely express doubt about the **world**? Only the former is `UNSUPPORTED`. |
| 4 | `DIRECT` vs rhetorical/interrogative prose | Does proceeding require the user to answer? If not, the question is rhetorical → `DIRECT`. |
| 5 | actual `CALL` vs describing a call | Is a machine-readable payload present? If not → not `CALL` (→ `CLARIFY` if it asks, else `AMBIGUOUS`). |
| 6 | inability from missing user info vs unavailable capability | Ask what would fix it: user supplying a value → `CLARIFY`; a tool that is absent → `UNSUPPORTED`. |
| 7 | ordinary factual uncertainty vs tool-policy inability | "I'm not sure" ≠ "I cannot act". |
| 8 | polite follow-up after a complete answer vs required clarification | Would completing the task correctly be impossible without the user's reply? If no → `DIRECT`. |

## 3. Annotation decision tree

Applied in this order. It resolves behaviour semantically so a human can reproduce labels; it deliberately
does **not** mirror any proposed regex cascade, and P-DET must be able to *fail* that cascade.

1. Is there a machine-readable tool-call payload? → **`CALL`**.
2. Is the response empty, truncated, or non-substantive? → **`AMBIGUOUS`**.
3. Does the response decline or state inability to perform the request? →
   does it *also* deliver substantive content? yes → **`DIRECT`** (part-answer with caveat);
   no → **`UNSUPPORTED`**.
4. Does the response ask the user for something? →
   is the reply *necessary* to complete the task correctly? yes → **`CLARIFY`**;
   no (courtesy/stylistic/optional) → **`DIRECT`**.
5. Is the response advice to use an external service *instead of* doing the task? → **`UNSUPPORTED`**.
6. Does the response only describe, propose or narrate a call, or name an unoffered tool? → **`AMBIGUOUS`**.
7. Otherwise, was substantive content delivered? yes → **`DIRECT`**; no → **`AMBIGUOUS`**.
8. If two rules compete and neither dominates → **`AMBIGUOUS`**. Never guess to fill a class.

## 4. Annotation procedure

- **Two independent passes are the target.** Each item is labelled by two annotators who do not see each
  other's labels. Disagreements are **preserved** (both original labels retained) and adjudicated by a third
  deterministic pass; the adjudicated label becomes the gold label, and the record keeps the full trail.
- **If only one annotator is practical**, this is recorded as an explicit limitation, and a deterministic
  adjudication procedure applies instead: every item whose rationale invokes a boundary rule (§2) and every
  item the annotator marks uncertain is re-read against §3 and must cite the rule used. A single-annotator
  P-DET **cannot** report inter-annotator agreement, and the report must say so.
- **Per-item preserved fields** (the frozen record schema): `pdet_id`, `source_dataset`, `source_revision`
  or its documented null, `source_id`/`upstream_id`, `raw_record_hash`, `canonical_hash`, the offered tool
  schema (or a hash of it), the **user context shown to the annotator**, the **assistant response**,
  `gold_policy_label`, `ambiguity_status`, `annotator_id`, `annotator_rationale`, `boundary_rule_cited`,
  `annotation_version`, and for duplicates `prompt_duplicate_group`.
- Annotators see context and tools, never the future classifier's output, and never its proposed rules.
- **Inter-annotator agreement** is reported as raw agreement and Cohen's κ over the four modes (excluding
  `AMBIGUOUS` from κ, reported separately). Disagreements are never resolved silently.

## 5. Sampling design (frozen before drawing)

**Candidate population.** Prose-based training-side records whose upstream format provides no authoritative
decision label, drawn from `data/processed/normalization-v1/when2call-sft` (15 shards). This is the material
the classifier will actually be applied to. The population size is read from the artifact at freeze time and
recorded in the manifest (measured: **14,829** valid records).

**Two separated components**, both deterministic:

| Component | Size | Purpose |
|---|---|---|
| **Prevalence** | 400 | behaviour on naturally occurring data, at natural proportions |
| **Challenge** | 200 | stress the boundaries most likely to fail (§2) |
| **Total** | **600** | |

**Deterministic selection.** Mirroring this repository's existing partition idiom, selection is a keyed
SHA-256 ranking — `sha256(seed || pdet_id)` descending — with `seed = "opengrad-pdet-002-v1"`. No RNG state
and no ordering dependence, so the sample is byte-reproducible.

**Prevalence allocation.** Proportional to the population's *observable* strata, which are measured rather
than assumed: prompt-duplicate group, presence or absence of an offered tool list, response-length bucket,
and the `HEURISTIC_REGEX_v1` refusal signal — used **only** as a sampling stratum, never as a label.
Proportions and realised counts are reported so a reader can confirm the sample was not silently rebalanced.

**Challenge families** (≥15 each where the population permits; shortfalls reported, never backfilled):
clarification questions with and without `?`; direct answers containing rhetorical questions; polite
questions that are not required clarification; refusal-shaped wording that actually requests missing
information; unsupported-capability responses; epistemic uncertainty that must not be read as incapability;
prose mentioning a tool without invoking it; serialized/textual tool-call forms in every observed shape;
answers with caveats ("I can't know X, but …"); multi-clause answer-plus-follow-up; unusual-but-legitimate
direct answers likely to confuse a regex detector.

Challenge items are selected **from the frozen specification and observable dataset characteristics only**.
Selecting them by running a classifier is impossible here (none exists) and would be forbidden in any case.

**Exclusions and dedup.**
- Excluded: every example id in the Study 002 confirmatory partition (1,277) **and** the DEV partition
  (2,373) — both are model-evaluation material.
- Excluded: quarantined ids, and any record whose user prompt matches a held-out question (measured: **1**
  such record, which must be dropped).
- Deduplicated by `raw_record_hash`, then by normalized assistant response text; the surviving member of each
  prompt-duplicate group is chosen by the same keyed ranking, and the group id is recorded so a reader can see
  the duplication was handled rather than ignored.
- **No item may appear in both components.**

**Coverage rule.** Desired minimum **50 gold examples per mode** across P-DET. If annotation cannot reach 50
for a mode — because the natural population does not contain enough confidently annotatable examples — that is
**reported as a finding**, that mode's metrics are reported with its actual (smaller) `n` and wide intervals,
and **the classifier may not be granted balancing permission for that mode**.

## 6. Classifier acceptance criteria (preregistered before implementation)

Evaluated on P-DET with: overall accuracy · macro F1 · per-class precision · per-class recall · per-class F1 ·
full confusion matrix · abstention rate · **and each of those reported separately for the prevalence and
challenge subsets**.

| Criterion | Threshold | Rationale |
|---|---|---|
| `DIRECT` recall | **≥ 0.80** | the exact failure Study 001 exposed: a metric that cannot see direct answering |
| `DIRECT` precision | **≥ 0.80** | otherwise direct-labelled retention data is polluted |
| `UNSUPPORTED` recall | **≥ 0.75** | under-detecting decline is the mirror failure |
| `CLARIFY` F1 | **≥ 0.70** | the rarest and most confusable mode |
| `CALL` precision | **≥ 0.95** | a false `CALL` injects a wrong tool-call target into training |
| macro F1 | **≥ 0.75** | prevents one dominant class carrying the score |
| abstention rate | **≤ 0.15** | an abstaining classifier cannot label a corpus |
| each mode, **challenge** subset | recall ≥ **0.60** | a classifier that works only on easy prevalence data is not trusted for balancing |

These are **Study 002 design choices**, chosen to represent the reliability actually required for
training-mixture construction. They are **not** empirically established, and they are **not** derived from the
audit-time 3,652-item MCQ feasibility probe — that probe is exploratory evidence only and may not seed or
justify thresholds after the fact.

**Failure behaviour, fixed in advance.**
- A mode below its threshold → the classifier does **not** qualify for balancing that mode. Its records may
  still be retained with `UNKNOWN`, but no balancing weight may depend on them.
- `DIRECT` or `UNSUPPORTED` failing → C1 is **not authorised**, because those are the two modes the
  intervention exists to restore.
- **Iterating regexes against P-DET until it passes is prohibited.** If the classifier changes after P-DET
  results are seen, the classifier version is incremented, P-DET is marked `DEVELOPMENT_EXPOSED`, and a
  **second untouched validation set is required** before balancing permission can be granted.

## 7. Lifecycle and evidence layers

```
human P-DET gold  →  validates  →  HEURISTIC_POLICY_v1  →  labels  →  training corpus
                                   →  trains  →  Study 002 model  →  evaluated independently
```

Recorded explicitly: classifier development data and P-DET data are **distinct**; Study 002 model-evaluation
data is separate again; P-DET labels validate the **ETL classifier, not the model**; P-DET results are **not**
evidence that Study 002's model policy improved; and classifier qualification grants permission **only** for
heuristic labels to influence corpus construction.

**Freeze package:** this protocol; the decision tree (§3); the sampling specification (§5); the seed; selected
item ids; source revisions; contamination exclusions; raw annotation records; adjudication records; final gold
labels; the metrics specification (§6); the thresholds; hashes; a manifest; the protocol version; and an
explicit statement that the classifier had **not** yet been implemented when the examples were selected.

**Frozen-artifact validator:** `src/opengrad/verification/pdet.py` re-hashes the frozen population and the
manifest and fails if either has changed, so "frozen" is checkable rather than asserted.