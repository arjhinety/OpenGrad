# 41 — The `ANSWER` strata set: sources, construction and labelling (`study_002_prereg_v9`)

> **Status: ADOPTED 2026-09-24.**
> - **Decision:** after reading the source screening (`registry/source_screening.yaml`, screening
>   `study-002-answer-heldout`), the study owner chose: *"Proceed with your recommendation on the ANSWER
>   source."*
> - **What that recommendation was:** relabel BFCL's no-call items into the four modes with the three-model
>   protocol of [34](34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md), and add a separately reported constructed
>   stratum of Natural Questions questions paired with tools that cannot serve them.
> - **What this document does:** it fixes every parameter of that recommendation before any item is drawn or
>   labelled.
> - **Where it is recorded:** in [03](03-PREREGISTRATION.md) and `reports/ERRATA.md` §26, in the same commit.

## 1. Why

- **The gap.** [06](06-SPLIT-SPEC.md) requires `P-CONF` to hold all four gold modes, with an `ANSWER` strata set
  of n ≥ 200, about 385 to resolve 10 points and about 601 to resolve 8. No pool in the repository reaches
  the 200 floor while staying disjoint from training: its status note of 2026-09-20 found the largest
  non-training pools hold 140 and 100 `ANSWER` items.
- **The search.** The screening of 2026-09-24 looked at 27 public candidates
  (`reports/source-screening/study-002-answer-heldout/REPORT.md`, flow in `docs/datasets/SOURCE_SCREENING.md`):
  - Most hold no knowledge-answerable item with tools offered.
  - Two that match the need exactly are unreleased or unlicensed.
  - BFCL's no-call files are the one usable natural source. A sized sample estimates about 389 `ANSWER`
    items (95% range 274–600), too few for 8 points on their own.
- **So this amendment builds two strata.** One is natural, the other constructed, and they are reported
  separately.

## 2. Items changed

- **[06](06-SPLIT-SPEC.md) "Building `P-CONF` for four modes", item 2.**
  - **The source is named:** the two pools of §3–§5, labelled under §7–§9.
  - **The balance requirement is replaced** by the measured report of §11. It asked that the `ANSWER` rows
    be balanced against the other three modes in prompt length, `context_bucket` and source:
    - balance in source is impossible, because the other modes all come from When2Call;
    - enforcing length balance by subsampling would cut the strata below their sizing.
  - **The materialization clause is satisfied by repository code.** It asked for a content hash produced by
    repository code. This population's hash is written by `src/opengrad/verification/answer_strata.py`, and
    `P-CONF`'s own materialization is a later step.
- **Nothing else in 06 changes:** the C1 coverage rule, the C2 resolvability rule, the `n ≥ 200` floor, the
  one-shot discipline, the disjointness rules and the other partitions stay as they are.

## 3. Sources, pinned

| Pool | File | Revision | sha256 | Licence |
|---|---|---|---|---|
| N | `BFCL_v4_irrelevance.json` | ShishirPatil/gorilla `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` | `2b6ed4c2e992cdcf5f1678a701851f944bef7550ee026ed1ddb89efed5be01a6` | Apache-2.0 |
| N | `BFCL_v4_live_irrelevance.json` | same | `6559fda2beaceb609a2cd2e504c65b4a56cb448e1ef88fddfd199e163d163349` | Apache-2.0 |
| K (questions) | `nq_open/validation-00000-of-00001.parquet` | google-research-datasets/nq_open `5dd9790a83002ad084ddeb7c420dc716852c6f28` | `b074bed0bccb56fa1551a8ac1c9c51ce89bc11c7fbb6a9c713b2c33a98531e12` | CC-BY-SA-3.0 |
| K (tools) | `BFCL_v4_simple_python.json` | gorilla `6ea5797…` as above | `82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991` | Apache-2.0 |

- **Pinning and download.** The gorilla revision is the one `registry/benchmarks.yaml` already pins for
  `bfcl-v4`. The builder downloads each file at its revision and refuses a file whose digest differs.
- **`nq_open` is the Natural Questions development questions restricted to those with a short answer** of at
  most five tokens: 3,610 questions, each with its reference answers.
- **Licence correction.** The screening first recorded Natural Questions as CC-BY-4.0, which is the licence of
  the paper. The data cards state CC-BY-SA-3.0. Share-alike applies: the constructed items are distributed
  under CC-BY-SA-3.0 with attribution.

## 4. Pool N: the natural items

- **Eligible:** every item of the two BFCL files that has exactly one user turn, no system message, and at
  least one offered tool.
- **Excluded:** items with more than one user turn or with a system message. They are counted: the other modes'
  items are single questions without a system prompt, and a system prompt changes the policy being measured.
- **Duplicates removed.** An item whose user message, lowercased with punctuation replaced by spaces and
  whitespace collapsed, equals another item's is removed. The one kept is first in ascending item-id order.
  The same rule applies across the two pools, with pool N kept over pool K. Removals are counted.
- **Not subsampled:** every eligible item that survives §6 is labelled.

## 5. Pool K: the constructed items

- **Questions.** They are drawn from the 3,610 `nq_open` questions in ascending order of
  `sha256("opengrad-answer-strata-001-v1" | "K" | question)`.
  - A question is skipped if it carries a time cue that could make a stable fact current: the words
    `current`, `currently`, `latest`, `now`, `today`, `this year`, `recent`, `recently`, `new`, `newest`,
    `next`, `upcoming`, `still`.
  - A question is also skipped if it has fewer than four words, or if §6 flags it.
  - The first **850** that survive are kept.
- **Tools.** Each question is paired with k tool schemas, where k is 1 + (the first byte of
  `sha256(seed | "k" | question)` mod 3).
  - The tools come from the distinct functions of `BFCL_v4_simple_python.json`, in ascending order of
    `sha256(seed | "tool" | question | function name)`.
  - Skip a function whose name or description matches a general information route: `search`, `lookup`,
    `look_up`, `query`, `wiki`, `knowledge`, `encyclop`, `fact`, `trivia`, `news`, `web`, `google`, `browse`,
    `retriev`, `question`, `answer`, `definition`, `dictionary`, `biograph`, `histor`, `database`.
  - Skip a function whose name or description names an entity domain Natural Questions asks about: `music`,
    `song`, `album`, `singer`, `movie`, `film`, `actor`, `celebrity`, `sport`, `game`, `team`, `player`,
    `book`, `author`, `artist`, `award`, `election`, `president`, `country`, `capital`, `population`,
    `geograph`, `city`, `state`.
  - Skip a function that shares a content word (four or more letters, lowercase, not a stop word) with the
    question.
- **The reference answers** stay on the item and are hidden from annotators (§7). They record that the
  question has a stable, human-annotated answer, which is what makes it knowledge-answerable.
- **Deliberate difference from the screening report.** The report proposed a no-tool model control to confirm
  each question is answerable. It is replaced by those human reference answers plus the three-model
  judgement of §8. What the stratum needs is that a direct answer is the right *kind* of response. Whether
  one particular model knows the fact is not the property being measured.

## 6. Exclusion screen (both pools)

An item is excluded before labelling if its user message is flagged against any user turn of OpenGrad's
normalized corpora. Those corpora are `data/processed/normalization-v1/{button, glaive, looptool, toolace, xlam,
when2call-sft, when2call-preference, when2call-mcq, when2call-llm-judge}`: every training source, plus the
When2Call evaluation splits already in use. An item is flagged by either of two tests:
- **(a) The contamination engine** `opengrad.contamination.levels`, at levels 1–4 with its default thresholds.
  Any level counts.
- **(b) The screening probe.** It uses word 8-grams: an 8-gram containment of at least 0.5 in one corpus, or
  an exact match after lowercasing and replacing punctuation with spaces.

Counts per test and per pool go in the manifest. Level 5 is not claimed: an excluded item is removed, never
adjudicated back in.

## 7. Annotation

- **One task, `answer-strata-v1`.** Items from both pools are mixed in the order
  `sha256(seed | "order" | item id)`.
- **What each annotator sees:** the user message and the offered tools, nothing else.
- **What is hidden:** the pool, the source dataset, the upstream id, the reference answers and every
  provenance field.
- **The annotators are the three of [34](34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md):** Gemini 3.8 Flash
  (High), gpt-5.6-sol and deepseek-v4.1-flash.
  - Each labels every item independently and blind, under `configs/annotation/answer-strata-v1.model-procedure.md`.
  - Each runs outside the repository with no file access (`scripts/run_external_annotation.py`).
- **No Claude model labels an item.** Claude designed this construction, and a designer labelling their own
  population is the dependence 28 item 3 guards against.

## 8. Rubric

For each item, decide what the assistant **should** do in reply to the user's message, given only the tools
listed with it. You are labelling the right response, not grading one; no response is shown.

- **`CALL`** — one of the offered tools is the right way to do what the user asks, and the message gives what
  the call needs, or reasonable defaults cover it.
- **`ANSWER`** — the right reply is prose from general knowledge or reasoning: facts, explanations,
  calculations, advice, writing, code or conversation. None of the offered tools is needed. A tool that is
  related but unnecessary does not make an item `CALL`.
- **`CLARIFY`** — the right reply asks for information only the user can give, such as which item, which
  account or a missing value. Proceeding without it would risk a wrong answer or action.
- **`UNSUPPORTED`** — the request needs something that neither the offered tools nor general knowledge
  provide, and no offered tool does it:
  - live or current information (today's weather, current prices, tonight's schedule, something near the
    user);
  - the user's own private records;
  - an action in the world.
- **`UNKNOWN`** — two modes remain genuinely plausible, or the message is not substantive or cannot be read
  without missing context. Set the matching `ambiguity_status`.

Boundary rules, in this order:

1. **`ANSWER` vs `CALL`.** If an offered tool would do exactly what is asked, the answer is `CALL`. If the
   answer is general knowledge and a tool is merely adjacent, the answer is `ANSWER`.
2. **`ANSWER` vs `UNSUPPORTED`.**
   - A stable fact is `ANSWER`: history, science, a definition, how to do something, who did what in a past
     year.
   - A fact that must be current is `UNSUPPORTED`, unless an offered tool provides it (then `CALL`): today's
     rate, a live score, a present roster.
3. **`CLARIFY` vs `UNSUPPORTED`.** If the user could supply what is missing, the answer is `CLARIFY`. If a
   capability is missing, it is `UNSUPPORTED`.
4. **Conversation.** A greeting or small talk is `ANSWER`, because a conversational reply is the right
   response.
5. **Language.** Judge a message in any language the same way.
6. **Your own knowledge does not matter.** Do not ask whether you personally know the answer. Ask whether a
   direct answer from general knowledge is the right kind of response. An obscure but stable fact is still
   `ANSWER`.

## 9. Reference labels and strata

- **Consensus rule.** An item's reference label is the label at least two of the three annotators give. A
  three-way split is `NO_CONSENSUS` and is excluded. The reference is `MODEL_REFERENCE` and provisional, as
  in 34.
- **The `ANSWER` strata set** is the items whose reference label is `ANSWER`, in two strata reported
  separately:
  - `ANSWER-natural` (pool N);
  - `ANSWER-constructed` (pool K).

  A pooled figure may be printed only beside the two separate ones, never instead of them.
- **Pool N's other consensus labels** (`CALL`, `CLARIFY`, `UNSUPPORTED`) are recorded as a by-product. This
  amendment adds none of them to `P-CONF` or `P-UNANS`; that would be a separate decision.
- **Pool K items reaching another consensus label** are recorded and counted as construction failures, and
  never relabelled.

## 10. Sizing and stop rules

Each stratum is judged against 06 on its own n:

| n | Status |
|---|---|
| n ≥ 601 | resolves 8 points |
| 385 ≤ n < 601 | resolves 10 points |
| 200 ≤ n < 385 | meets the floor only |
| n < 200 | `UNDER_POWERED` |

- **No redraw.** Neither pool is redrawn, topped up or re-screened to reach a size. A shortfall is reported,
  and fixing it needs a new amendment.
- **Every consensus `ANSWER` enters its stratum.** No item is dropped after its label is seen, except by the
  §6 screen, which runs before labelling.

## 11. Balance, measured rather than enforced

The manifest of the labelled set reports, for each stratum and for each of the other three modes of the
confirmatory partition:
- the prompt-length distribution, in characters and quartiles;
- the number of offered tools.

A difference is a stated limitation of any `ANSWER`-mode comparison. It is never corrected by reweighting
after scores exist.

## 12. Independence and limitations

- **Model reference, not human gold.** The labels are model judgments (`MODEL_REFERENCE`).
- **Public data.** BFCL and Natural Questions are public, so the base model may have seen their text. That
  would affect answer quality more than the decision to answer, but it cannot be ruled out.
- **The constructed stratum is a controlled construction, not a natural distribution.** Its questions are
  short factual ones and its tools come from one benchmark.
- **The natural stratum inherits BFCL's collection:** unmodified user traffic to one hosted endpoint, several
  languages, and labels an independent audit found noisy. That noise is why it is relabelled.

## 13. Not changed

- every threshold in [11](11-THRESHOLDS.md);
- the arms, the seeds and the confirmatory partition;
- the three-mode `P-CONF` items;
- `P-UNANS`, `P-DEV` and `P-SEALED`;
- every classifier, P-DET and C1 decision of v2–v8;
- training, which remains unauthorised.
