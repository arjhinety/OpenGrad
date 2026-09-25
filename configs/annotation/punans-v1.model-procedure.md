# P-UNANS-v1 model annotation procedure

These are the complete instructions given to each external model annotator of the P-UNANS-v1 candidate
population. The annotator receives this text verbatim, followed by one batch, in a single input, and has no file
to open.
- **Pinning.** The file is pinned by SHA-256 in `configs/annotation/punans-v1.yaml` (`model_annotators`), and
  every recorded label names that hash.
- **Authority.** It is authorized by `docs/research/study-002/43-PUNANS-AMENDMENT-DRAFT.md` (amendment
  `study_002_prereg_v11`).
- **Runner.** `scripts/run_external_annotation.py`.

---

You are annotating items for OpenGrad Study 002. Your labels are recorded as **model judgments** under your
model annotator id and combined with one other model's independent label; an item is kept only where both
agree. They are never presented as human labels.

## What to read

Everything you need follows these instructions in this same input: the rubric (the same text the annotation
screen shows) and your items. Each item is shown the way the annotation screen shows it. Do not open, search
for or run anything: no file, no command, no web page. Read the whole input before answering.

## How to decide

For every item, independently:

- **What you see.** Each item shows **one question** and the **tools the assistant was offered**.
  The tools are there because every item in this study offers tools; decide about the question itself.
- **What you decide.** Whether the question can be answered, using the rubric's definitions, as of today. Choose
  one label:
  - `UNKNOWABLE`: no one can answer the question now (a future event, an unsolved problem, a fact no one can
    determine). If the "future" event has already happened and its outcome is known, it is `ANSWERABLE`.
  - `FALSE_PREMISE`: the question assumes something untrue, so it cannot be answered as asked.
  - `ANSWERABLE`: a correct answer exists and can be given from general knowledge, even if it is obscure.
  - `SUBJECTIVE`: the question asks for an opinion, a preference or a value judgment, so there is no single
    right answer to know.
  - `UNDERSPECIFIED`: the question lacks context the user could supply (which place, which person, which
    version), so the right reply is to ask.
  - `UNKNOWN`: the item cannot be decided (see below).
- **What you do not decide.** Do not judge whether you personally know the answer. An obscure but determinable
  fact is `ANSWERABLE`.
- **Each item on its own content.** There is no expected distribution, no quota and no target. Do not balance
  classes. Do not let the labels you gave earlier items, or the order of the items, influence later ones.
- **When it cannot be decided,** label the item `UNKNOWN` with the matching `ambiguity_status`. That covers two
  labels remaining genuinely plausible, a question that is not substantive, and a question that cannot be read
  without missing context. Never guess to fill a class.

## What to return

Your final message must be a single JSON array and nothing else: no prose before or after it, no code
fence. It holds exactly one object per item in the batch, in batch order:

```
{"item_id": "<the item id, copied exactly>",
 "label": "UNKNOWABLE" | "FALSE_PREMISE" | "ANSWERABLE" | "SUBJECTIVE" | "UNDERSPECIFIED" | "UNKNOWN",
 "ambiguity_status": "NONE" | "AMBIGUOUS_TWO_MODES" | "NON_SUBSTANTIVE" | "MISSING_CONTEXT",
 "annotator_rationale": "<one sentence saying why>",
 "flag": true | false}
```

- `UNKNOWN` requires an `ambiguity_status` other than `NONE`. Every other label requires `NONE`.
- `flag: true` marks an item you are unsure about.
- Use no other keys.

---

The batch follows.
