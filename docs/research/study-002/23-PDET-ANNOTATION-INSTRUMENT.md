# 23 — P-DET annotation instrument

**For the human annotator.** Companion to [22-PDET-PROTOCOL.md](22-PDET-PROTOCOL.md), which defines the
rubric, the boundary rules and the decision tree. This document is the working instrument: what to open, what
to fill in, the allowed values, and how to record uncertainty.

> Gold labels **do not exist yet**. Until they do, no classifier may be measured, no balancing permission may
> be granted, and C1 is not authorised. This instrument exists to unblock that step.

## 1. Open the frozen population

```
reports/pdet/pdet-v1.population.jsonl     # 581 items, one JSON object per line
```

Its integrity is checkable at any time, and must be checked **before and after** annotation:

```
python -m opengrad.verification.pdet --verify
```

Expected `population_sha256`:
`6ab920877ce8004056a747f36d0a9c9ae6bd8befeb249007d70a79a6ce9e781b`. If `--verify` does not return `PASS`,
**stop**: annotating a population that has moved produces labels that cannot attach to a frozen target.

## 2. Fields you fill in

| Field | Allowed values | Required |
|---|---|---|
| `gold_policy_label` | `CALL` · `DIRECT` · `CLARIFY` · `UNSUPPORTED` | yes |
| `ambiguity_status` | `NONE` · `AMBIGUOUS_TWO_MODES` · `NON_SUBSTANTIVE` · `MISSING_CONTEXT` · `EMPTY_TOOLSET` | yes |
| `annotator_rationale` | free text — **one sentence** saying why | yes |
| `boundary_rule_cited` | `1`–`8` per [22 §2](22-PDET-PROTOCOL.md), or `none` | when a boundary decided it |
| `annotator_id` | your identifier | yes |
| `annotation_version` | `pdet-annotation-v1` | yes |

If `ambiguity_status != NONE`, set `gold_policy_label` to the string `UNKNOWN` and cite the reason. **Never
guess to fill a class**, and never reassign an item to reach a coverage target — shortfalls are legitimate
findings ([22 §5](22-PDET-PROTOCOL.md)).

Write your pass to a **new** file: `reports/pdet/pdet-v1.annotations.<annotator_id>.jsonl`, containing
`{"pdet_id", "gold_policy_label", "ambiguity_status", "annotator_rationale", "boundary_rule_cited",
"annotator_id", "annotation_version"}`. Do **not** edit `pdet-v1.population.jsonl` — the frozen population
stays frozen, and adjudication produces a separate file.

## 3. How to decide

Work through [22 §3](22-PDET-PROTOCOL.md) in order, and consult [22 §2](22-PDET-PROTOCOL.md) when two readings
compete. Annotate **the intended behaviour of the assistant response in context** — not its punctuation, and
not its keywords.

## 4. Worked examples (illustrative — these are *not* gold labels)

Each is a real frozen item. The labels shown illustrate the rubric and must not be copied into a file.

**① A refusal-shaped response that still answers.** `families: refusal_with_question`, 2 tools offered.

- prompt: *"Convert the sentence 'How are you?' to speech in French."*
- response: *"I'm sorry, I can't perform speech generation or translation tasks. However, the sentence 'How
  are you?' translates to 'Comment ça va?' in French."*

→ **`DIRECT`**, `ambiguity_status: NONE`, `boundary_rule_cited: 3`. Rule 3 asks whether content was *also*
delivered — it was — so this is a part-answer with a caveat, not a decline. A detector keyed on `"I can't"`
gets this wrong, which is why the item is in the challenge set. The trailing `?` belongs to the quoted
sentence, not to a clarification request (rule 4).

**② A decline with no content.** `families: refusal_plain, caveat_then_content`, 1 tool.

- prompt: *"Fetch the liked tweets of the user with ID '0987654321' with a limit of 20…"*
- response: *"I apologize, but I'm unable to perform that task. I can't access personal data or perform
  actions on social media platforms."*

→ **`UNSUPPORTED`**, `boundary_rule_cited: 3`. Nothing was delivered, and the missing thing is capability
(rule 2), not user input.

**③ Policy-level inability on an ordinary task.** `families: refusal_plain, no_tools_offered,
advice_external_service`, 0 tools.

- prompt: *"What is the standard deviation of the following list of numbers: [1, 2, 3, 4, 5]?"*
- response: *"I'm sorry for the inconvenience, but I'm currently unable to perform calculations. I recommend
  using a calculator or software for this task."*

→ **`UNSUPPORTED`**. Arithmetic is within a general model's ability, so this is not *epistemic* inability — it
is a stated inability to **act**, which is what the class captures (rule 7 separates "I'm not sure" from "I
cannot act"). Recommending an external tool **instead of** performing the task is decline-shaped, not
clarification (rule 5).

## 5. Two passes, then adjudication

1. **Pass A and Pass B** annotate independently. Neither sees the other's file, and neither consults any
   classifier output (none exists).
2. **Disagreements are preserved**, never overwritten. An item disagrees when the two passes differ on
   `gold_policy_label` or on `ambiguity_status`.
3. **Adjudication** produces `reports/pdet/pdet-v1.adjudicated.jsonl` holding, per item, **both** original
   labels, the adjudicated `gold_policy_label`, the deciding rule from [22 §3](22-PDET-PROTOCOL.md), and the
   adjudicator id.
4. **Report** raw agreement and Cohen's κ over the four modes, with `UNKNOWN` handled separately as in
   [22 §4](22-PDET-PROTOCOL.md).
5. If only one annotator is available, say so explicitly: the deterministic re-read procedure in
   [22 §4](22-PDET-PROTOCOL.md) applies, and **no inter-annotator agreement may be claimed**.

## 6. Coverage bookkeeping after annotation

Per-mode counts decide whether the 50-per-mode rule is met:

```
python -c "import json,io,collections; rows=[json.loads(l) for l in io.open('reports/pdet/pdet-v1.annotations.<id>.jsonl',encoding='utf-8')]; print(collections.Counter(r['gold_policy_label'] for r in rows).most_common())"
```

A mode below 50 is reported as a finding, its metrics are reported with the actual `n`, and **that mode does
not qualify for balancing**. No rebalancing by relabelling — ever.

## 7. What not to do

- Do not iterate rules against P-DET and re-annotate to make a classifier pass. If the classifier changes
  after P-DET results are seen, P-DET becomes `DEVELOPMENT_EXPOSED` and a **second untouched validation set**
  is required ([22 §6](22-PDET-PROTOCOL.md)).
- Do not treat this population as model-evaluation data. It is ETL validation material and is disjoint from
  the confirmatory and DEV partitions by construction (verified: 0 overlap).
- Do not describe any result here as evidence that Study 002's policy improved. It validates a labelling
  component, nothing else.