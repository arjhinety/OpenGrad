# Candidate Evaluation

How a trained checkpoint is measured against B0, and why it needs its own path.

## Why `run_baseline` cannot do it

`run_baseline` validates its config against the frozen contract and refuses anything that is not
the pinned canonical model:

```python
if config["model_id"] != CANONICAL_MODEL_ID or config["model_revision"] != PINNED_MODEL_REVISION:
    raise ValueError("baseline config must pin the canonical model, model revision, ...")
```

That guard is what keeps B0 immutable, so relaxing it for a candidate would remove the property
the baseline exists to have. The candidate path is therefore separate rather than permissive.

## What is shared, and what is allowed to differ

`opengrad.evaluation.candidate.run_candidate_evaluation` calls the same
`measure_predictions` the baseline calls. That function was extracted from `run_baseline` for
exactly this reason: the engine, renderer, template hash, parser, generation settings, context
bucketing, and metrics live in one implementation, so a candidate and the baseline cannot drift
into two different measurements.

`validate_candidate_config` enforces the split. A candidate may change the model and nothing
else; altering any of these is refused:

* `renderer`, `template_hash`
* `seed`
* `generation`, including temperature, `top_p`, and `max_new_tokens`
* `runtime`, including the engine, its version, precision, and the model window
* `evaluations.behavioral_manifest`

Each of those would move the metric without moving the model, which is the failure mode this
check exists to prevent.

A candidate must also record its lineage — `parent_experiment_id`, `checkpoint_id`,
`checkpoint_step`, `base_model_id`, `base_model_revision` — so a delta is attributable to a
specific checkpoint rather than to "some later model". Outputs are required to live under the
parent run's directory: a candidate can never write the baseline evidence paths.

## Producing a curve

```bash
python scripts/evaluate_sft_checkpoints.py qwen35_2b_m0_sft_full_v3
python scripts/evaluate_sft_checkpoints.py qwen35_2b_m0_sft_full_v3 --steps 400,1200,2400
```

The driver generates one candidate config per checkpoint from the frozen baseline config — so
the invariant block is literally the baseline's, not a hand-copied approximation — and writes
`runs/<experiment>/eval/curve.json` with one measurement per checkpoint, each including its own
comparison against the baseline metrics.

The same command runs either path:

```bash
opengrad baseline --config configs/evaluation/candidates/<experiment>/checkpoint-400.yaml --json
```

dispatch is by the config's `status` field. Anything that is not `CANDIDATE_EVALUATION` still
goes to `run_baseline`.

## Using the curve honestly

The comparisons in `curve.json` are per-metric deltas with an explicit direction: `call_f1`,
`call_precision`, and `call_recall` are better higher, while `over_call_rate` and
`under_call_rate` are better lower. A metric present on only one side is skipped rather than
counted as a regression.

One caveat belongs on the record. Selecting a checkpoint *because* it scores well on this set is
selecting on the evaluation set, and the selected number is then optimistic. Treating the curve
as a diagnostic and reporting a pre-registered checkpoint (for example the final one) as the
primary result avoids that. The curve is reported in full precisely so a reader can see what the
selection was.
