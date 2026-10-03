# First-reply review model annotation procedure (v1)

These are the complete instructions given to each external model annotator of Study 002's first-reply review:
a 100-item trial, the full review and a second sample. The annotator receives this text verbatim, followed by
one batch, in a single input, and has no file to open.
- **Pinning.** The file is pinned by SHA-256 in each task's configuration (`model_annotators`), and every recorded
  label names that hash.
- **Authority.** Study 002 amendments `study_002_prereg_v14` (document 46, section 5) and `study_002_prereg_v15`
  (document 47). Section 5 of document 46 allows one revision after the trial, recorded before any label of the
  full review.
- **Runner.** `scripts/run_external_annotation.py`. No labelling run starts without the study owner's word.

---

You are annotating items for OpenGrad Study 002. Your labels are recorded as **model judgments** under your
model annotator id and combined with one other model's independent label; an item is decided only where both
agree. They are never presented as human labels.

## What to read

Everything you need follows these instructions in this same input: the rubric (the same text the annotation
screen shows) and your items. Each item is shown the way the annotation screen shows it. Do not open, search
for or run anything: no file, no command, no web page. Read the whole input before answering.

## How to decide

For every item, independently:

- **What you see.** The tools the assistant was offered, the user's first message, and the assistant's first
  reply.
- **What you decide.** One thing only: what the first reply does, using the rubric's definitions. Choose one
  label:
  - `NOT_A_DECLINE`: the reply does not decline.
  - `DECLINE_JUSTIFIED`: it declines, and the request needs something the assistant lacks (current or private
    data, an action in the world, a tool not offered).
  - `DECLINE_UNJUSTIFIED`: it declines, but a capable assistant could answer correctly from general knowledge or
    reasoning, without the tools.
  - `UNKNOWN`: the item cannot be decided (see below).
- **What you do not decide.** Do not judge whether the reply is polite, well written or factually correct, and do
  not answer the user's request yourself.
- **Each item on its own content.** There is no expected distribution, no quota and no target. Do not balance
  classes. Do not let the labels you gave earlier items, or the order of the items, influence later ones.
- **When it cannot be decided,** label the item `UNKNOWN` with the matching `ambiguity_status`: two labels remain
  genuinely plausible (including a reply that declines although an offered tool could have done what was asked),
  the item is not substantive, or it cannot be read without missing context. Never guess to fill a class.

## What to return

Your final message must be a single JSON array and nothing else: no prose before or after it, no code
fence. It holds exactly one object per item in the batch, in batch order:

```
{"item_id": "<the item id, copied exactly>",
 "label": "NOT_A_DECLINE" | "DECLINE_JUSTIFIED" | "DECLINE_UNJUSTIFIED" | "UNKNOWN",
 "ambiguity_status": "NONE" | "AMBIGUOUS_TWO_MODES" | "NON_SUBSTANTIVE" | "MISSING_CONTEXT",
 "annotator_rationale": "<one sentence saying why>",
 "flag": true | false}
```

- `UNKNOWN` requires an `ambiguity_status` other than `NONE`. Every other label requires `NONE`.
- `flag: true` marks an item you are unsure about.
- Use no other keys.

---

The batch follows.
