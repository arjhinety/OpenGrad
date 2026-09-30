# P-UNANS-v2: the procedure revision after the trial (44 §9)

**2026-09-30, recorded before any main-set label.** Amendment
[44](../../../docs/research/study-002/44-PUNANS-V2-AMENDMENT-DRAFT.md) §9 allows the labelling procedure to be
revised once after the trial set, to clarify §4's wording for the annotators. This is that revision. The owner
approved it as drafted.

## What changed

| | Procedure | sha256 | Used by |
|---|---|---|---|
| Before | `configs/annotation/punans-v2.model-procedure.md` | `9ad5e3cd1eb7d2ed00c65965fafb22433462c2e80ee4b7aaaec7de77cd15daed` | `punans-v2-trial` (unchanged) |
| After | `configs/annotation/punans-v2.model-procedure.rev1.md` | `2e7fd9c9fae13b5bf6b3b05f9c3f47dc7335455f76d59495e5c2235c3b72ffc4` | `punans-v2`, the main set |

- **The addition.** One bullet, "Where the line falls", saying what 44 §4 means in practice:
  - `UNKNOWABLE` asks for a specific answer that no one has now, so that the right reply is to decline;
  - a question that a careful reply would explain, weigh or discuss is `NOT_UNKNOWABLE`.

  It names the kinds of question that fall on each side. It quotes no trial or main-set question.
- **Unchanged:** the labels, the rubric (44 §4, served to the annotators), the 0.80 floor, the sources and the
  800 main-set questions. The trial's labels and its pinned procedure stay as they were.

## Why

The trial ([report](punans-v2-trial.report.json)) had raw agreement 0.810 but κ 0.560, below the first
attempt's 0.703. Every one of its 19 disagreements ran one way: Gemini said not unknowable where DeepSeek said
unknowable or undecided. So the models placed the line in different places.

With the owner's permission, Claude read the 19 split questions and both models' one-line reasons. It sorted
them by what the question does ([`punans-v2-trial.split-categories.jsonl`](punans-v2-trial.split-categories.jsonl),
ids and categories only):

| What the question does | Splits |
|---|---|
| Asks whether something can be achieved, or how a development will play out | 8 |
| Is hypothetical or philosophical | 3 |
| Rests on a false or unestablished premise | 3 |
| Asks whether undiscovered things of some kind exist | 3 |
| Asks for a private fact that the person concerned knows | 1 |
| Asks whether something can ever be determined | 1 |

- **DeepSeek's reading:** it treated an unsettled outcome as unknowable.
- **Gemini's reading:** it treated a question that a knowledgeable reply can discuss as not unknowable.
- **Which one §4 supports:** Gemini's. §4 lists open-ended questions and false premises as not unknowable, and
  says the right reply to an unknowable question is to decline.

The revision therefore follows the rubric, not one of the models. Only one split turns partly on a fact rather
than on the definition.

## What this does and does not show

- **Expected effect, a hypothesis:** DeepSeek moves toward Gemini, so agreement rises. The unknowable stratum may
  shrink toward Gemini's rate: Gemini called 62 of the 100 trial questions unknowable.
- **Not tested.** The revision was written from these 19 questions, so labelling them again would overstate its
  effect. The main set is its only test, and 44 §10's floor judges it there.
- **Harness.** The main set is labelled with cline 3.0.66 (`CLINE_NO_AUTO_UPDATE=1` for every start) and agy as
  installed. The runner records each attempt's version.
