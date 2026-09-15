# P-DET-v1 model annotation procedure

These are the complete instructions given to each model annotator subagent. Each subagent receives this
text verbatim, followed by one line naming its batch file. The file is pinned by SHA-256 in
`configs/annotation/pdet-v1.yaml` (`model_annotators`), and every recorded label names that hash. It is
authorized by `docs/research/study-002/28-PDET-MODEL-LABEL-AMENDMENT.md` (amendment `study_002_prereg_v2`).

---

You are annotating items from the P-DET-v1 population (OpenGrad Study 002). Your labels are recorded as
**model judgments**, under your model annotator id. They are never presented as human labels.

## What to read

Read exactly one file: the batch file named at the end of these instructions. It holds the rubric
excerpts (the same ones a human annotator sees on the annotation screen) and your items, each shown the
way the annotation screen shows it. Do not open, search for or run anything else: no other file in the
repository, no other batch, no web page. Everything you need is in the batch file. If it is long, read it
in parts until you have read all of it.

## How to decide

For every item, independently:

- Work through the decision tree (22 §3) in order, and consult the boundary discriminations (22 §2) when
  two readings compete, using the mode definitions (22 §1) and the instrument's fields and guidance (23 §2,
  §3). Annotate the intended behaviour of the assistant response in context -- what it actually does given
  the user message and the offered tools -- not its punctuation or its keywords, and not what it should
  have done.
- Decide each item on its own content. There is no expected distribution, no quota and no target. Do not
  balance classes, and do not let the labels you gave earlier items in the batch, or the order of the
  items, influence later ones.
- The metadata line (component, source, split, ids, duplicate count) is bookkeeping, not evidence for a
  label.
- If after the decision tree two modes remain genuinely plausible, or the response is not substantive, or
  the needed context is missing, or no tool was offered where the decision depends on one, label the item
  `UNKNOWN` with the matching `ambiguity_status`. Never guess to fill a class.

## What to return

Your final message must be a single JSON array and nothing else: no prose before or after it, no code
fence. It holds exactly one object per item in the batch, in batch order:

```
{"item_id": "<the item id, copied exactly>",
 "label": "CALL" | "DIRECT" | "CLARIFY" | "UNSUPPORTED" | "UNKNOWN",
 "ambiguity_status": "NONE" | "AMBIGUOUS_TWO_MODES" | "NON_SUBSTANTIVE" | "MISSING_CONTEXT" | "EMPTY_TOOLSET",
 "boundary_rule_cited": "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "none",
 "annotator_rationale": "<one sentence saying why>",
 "flag": true | false}
```

- `UNKNOWN` requires an `ambiguity_status` other than `NONE`. Every other label requires `NONE`.
- `boundary_rule_cited` names the boundary discrimination (22 §2) that decided the label, or `"none"`.
- `flag: true` marks an item you are unsure about, so it can be reviewed later.
- Use no other keys.
