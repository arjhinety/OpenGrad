# P-DET-COVERAGE-v1 model annotation procedure

These are the complete instructions given to each external model annotator of P-DET-COVERAGE-v1. The
annotator receives this text verbatim followed by one batch, in a single input, and has no file to open. The
file is pinned by SHA-256 in `configs/annotation/pdet-coverage-v1.yaml` and
`configs/annotation/pdet-coverage-v1-routing.yaml` (`model_annotators`), and every recorded label names that
hash. It is authorized by `docs/research/study-002/34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md`
(amendment `study_002_prereg_v5`). The runner is `scripts/run_external_annotation.py`.

---

You are annotating items from the P-DET-COVERAGE-v1 population (OpenGrad Study 002). Your labels are
recorded as **model judgments** under your model annotator id, and combined with two other models'
independent labels. They are never presented as human labels.

## What to read

Everything you need follows these instructions in this same input: the rubric excerpts (the same ones the
annotation screen shows) and your items, each shown the way the annotation screen shows it. Do not open,
search for or run anything: no file, no command, no web page. Read the whole input before answering.

## How to decide

For every item, independently:

- Work through the decision tree (22 §3) in order, and consult the boundary discriminations (22 §2) when
  two readings compete, using the mode definitions (22 §1), the instrument's fields and guidance (23 §2,
  §3), and this population's definition of DIRECT and its exclusion of post-tool responses (30 §2, §3).
  Annotate the intended behaviour of the assistant response in context -- what it actually does given the
  user message and the offered tools -- not its punctuation or its keywords, and not what it should have
  done.
- When an item shows a structured tool call as part of the response, judge whether the turn as a whole
  invokes an offered tool for the request. A structured call that plausibly invokes an offered tool,
  consistent with the request, is CALL.
- Decide each item on its own content. There is no expected distribution, no quota and no target. Do not
  balance classes, and do not let the labels you gave earlier items, or the order of the items, influence
  later ones.
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
- `flag: true` marks an item you are unsure about.
- Use no other keys.

---

The batch follows.
