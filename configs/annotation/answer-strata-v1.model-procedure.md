# ANSWER-STRATA-v1 model annotation procedure

These are the complete instructions given to each external model annotator of the ANSWER-STRATA-v1 candidate
population. The annotator receives this text verbatim, followed by one batch, in a single input, and has no file
to open.
- **Pinning.** The file is pinned by SHA-256 in `configs/annotation/answer-strata-v1.yaml` (`model_annotators`),
  and every recorded label names that hash.
- **Authority.** It is authorized by `docs/research/study-002/41-ANSWER-STRATA-AMENDMENT.md` (amendment
  `study_002_prereg_v9`).
- **Runner.** `scripts/run_external_annotation.py`.

---

You are annotating items for OpenGrad Study 002. Your labels are recorded as **model judgments** under your
model annotator id and combined with two other models' independent labels. They are never presented as human
labels.

## What to read

Everything you need follows these instructions in this same input: the rubric (the same text the annotation
screen shows) and your items. Each item is shown the way the annotation screen shows it. Do not open, search
for or run anything: no file, no command, no web page. Read the whole input before answering.

## How to decide

For every item, independently:

- **What you see.** Each item shows **one user message** and the **tools the assistant was offered**. There is
  no assistant response.
- **What you decide.** Decide what the assistant **should** do in reply, given only those tools. Apply the
  rubric's mode definitions and its six boundary rules in order.
- **What you do not decide.** Do not judge whether you personally know the answer. A direct answer from general
  knowledge is the right kind of response to a stable factual question even when the fact is obscure.
- **Each item on its own content.** There is no expected distribution, no quota and no target. Do not balance
  classes. Do not let the labels you gave earlier items, or the order of the items, influence later ones.
- **When it cannot be decided,** label the item `UNKNOWN` with the matching `ambiguity_status`. That covers two
  modes remaining genuinely plausible, a message that is not substantive, and a message that cannot be read
  without missing context. Never guess to fill a class.

## What to return

Your final message must be a single JSON array and nothing else: no prose before or after it, no code
fence. It holds exactly one object per item in the batch, in batch order:

```
{"item_id": "<the item id, copied exactly>",
 "label": "CALL" | "ANSWER" | "CLARIFY" | "UNSUPPORTED" | "UNKNOWN",
 "ambiguity_status": "NONE" | "AMBIGUOUS_TWO_MODES" | "NON_SUBSTANTIVE" | "MISSING_CONTEXT",
 "boundary_rule_cited": "1" | "2" | "3" | "4" | "5" | "6" | "none",
 "annotator_rationale": "<one sentence saying why>",
 "flag": true | false}
```

- `UNKNOWN` requires an `ambiguity_status` other than `NONE`. Every other label requires `NONE`.
- `boundary_rule_cited` names the boundary rule that decided the label, or `"none"`.
- `flag: true` marks an item you are unsure about.
- Use no other keys.

---

The batch follows.
