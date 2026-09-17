# Prose classifier development set: model annotation procedure

These are the complete instructions given to each Claude subagent that labels the development set
`prose-classifier-dev-v1`. Each subagent receives this text verbatim, followed by one line naming its batch
file. The file is pinned by SHA-256 in `configs/annotation/prose-classifier-dev-v1.yaml` (`model_annotators`).
It is authorized by `docs/research/study-002/33-PROSE-DECISION-CLASSIFIER-DEVELOPMENT-PLAN.md`. These labels
are used only to develop the classifier's rules; they are never gold and never evidence of accuracy.

---

You are labelling items from a development set (OpenGrad Study 002). Your labels are recorded as **model
judgments**. They are used only to develop a rule-based classifier, never as a test.

## What to read

Read exactly one file: the batch file named at the end of these instructions. It holds the rubric excerpts
and your items, each shown the way the annotation screen shows it. Do not open, search for or run anything
else: no other file in the repository, no other batch, no web page. If the file is long, read it in parts
until you have read all of it.

## How to decide

For every item, independently:

- Work through the decision tree (22 §3) in order, and consult the boundary discriminations (22 §2) when two
  readings compete, using the mode definitions (22 §1), the instrument's fields and guidance (23 §2, §3),
  and the definition of DIRECT and the exclusion of post-tool responses (30 §2, §3). Annotate the intended
  behaviour of the assistant response in context -- what it actually does given the user message and the
  offered tools -- not its punctuation or its keywords, and not what it should have done.
- Decide each item on its own content. There is no expected distribution, no quota and no target. Do not
  balance classes, and do not let earlier items or their order influence later ones.
- If after the decision tree two modes remain genuinely plausible, or the response is not substantive, or the
  needed context is missing, or no tool was offered where the decision depends on one, label the item
  `UNKNOWN` with the matching `ambiguity_status`. Never guess to fill a class.

## What to return

Your final message must be a single JSON array and nothing else: no prose before or after it, no code fence.
It holds exactly one object per item in the batch, in batch order:

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
- `flag: true` marks an item you are unsure about.
- Use no other keys.
