# 19 — Study 004 roadmap: speculative decoding

## Why this is a separate study

`docs/research/STUDIES.md:78-80` scoped speculative decoding into Study 002 as *"Target-attached or native
speculative decoding for decode efficiency without a separate draft model. The benchmark suite has a
speculative tier; no runtime support exists and nothing has been measured."* Two facts in that sentence force
the split: **no runtime support exists**, and **nothing has been measured**. The measured status today is
`BLOCKED_MISSING_MTP_COMPONENT` (`reports/H200_BENCHMARK_RUN.md:225`).

A runtime question also cannot share a preregistration with a data-contract question, for the reason given in
[18](18-STUDY-003-ROADMAP.md): one amendment would invalidate the other's thresholds, and the two failure
modes differ — one loses a behaviour, the other risks losing accuracy for speed.

## Where Study 004 starts

The specification already exists and is specific: `docs/evaluation/SPECULATIVE_DECODING.md` defines the
quality-parity principle, five execution modes and the required metrics. Study 004 inherits it rather than
inventing a second vocabulary.

**The governing constraint**, quoted from that document: *"QUALITY MUST NEVER BE HIDDEN BY SPEEDUP."* It
states the problem as a constrained optimisation, `max(speedup) subject to Δ(quality) ≥ −ε`. Study 004's job
is to make `ε` a pre-registered number with a population behind it, and to make `Δ(quality)` a **four-mode**
measurement rather than a single headline score — because a decode change that preserves `call_f1` while
degrading direct answering is exactly the Study 001 failure with a new cause.

| Mode | Identifier | Status |
|---|---|---|
| `MODE_A_AR` | standard autoregressive baseline | partially measured; the reference |
| `MODE_B_MTP` | native multi-token prediction heads | `BLOCKED_MISSING_MTP_COMPONENT` |
| `MODE_C_DRAFT` | two-model speculative decoding | not implemented |
| `MODE_D_DSPARK` | DSpark-style speculative execution | not implemented |
| `MODE_E_EXPERIMENTAL` | future draft/verifier algorithms | not implemented |

Metrics to report, from the same document: `TTFT` (cold and warm), `ITL` mean/p50/p95, end-to-end latency,
output tokens/s, total tokens/s, tasks/s, proposed draft tokens, accepted tokens, **acceptance rate**, and
accepted tokens per verifier step. The registry already reserves the measuring apparatus: `speculative-replay`
(`TIER_E`, `evaluator_version: opengrad-speculative-eval-v1`) and `performance-microsuite` (`TIER_E`).

## Study 004's research questions (draft, to be pre-registered)

- **RQ4.1** For each implemented mode, what are the acceptance rate and the speedup at fixed quality on the
  tool-policy population?
- **RQ4.2** What is the largest speedup available subject to `Δ(quality) ≥ −ε` across **all four modes**, not
  only `call_f1`? `ε` is pre-registered per mode using [06](06-SPLIT-SPEC.md)'s resolvable-margin arithmetic.
- **RQ4.3** Does speculation change *decisions*? A verifier that accepts a draft token changes the sampled
  trajectory, so a decode-mode change is a behaviour change. This is what makes Study 004 a behavioural study
  rather than a systems benchmark.
- **RQ4.4** Is the quality delta reproducible across two providers? Speculative kernels are the most
  hardware-sensitive artifact in this repository, and [14](14-HETEROGENEITY-POLICY.md)'s check applies with
  more force here than anywhere in Study 002.

RQ4.3 is why this study needs Study 002's metric set: without a four-mode population and a
refusal-correctness metric, a decode change could improve throughput while moving the model's decisions, and
the measurement would show only the speedup — the same blindness that let a refusing checkpoint be promoted.

## Inherited rules

Study 004 adopts: the split and one-shot discipline ([06](06-SPLIT-SPEC.md)), the metric spec and the
ability/decision/format separation ([07](07-METRIC-SPEC.md)), the seed and repeat policy
([05](05-SEED-AND-REPRODUCIBILITY-POLICY.md)), the statistics plan including the resolvable-margin
requirement ([10](10-STATISTICS-PLAN.md)), retention ([12](12-ARTIFACT-RETENTION.md)), the hardware-agnostic
preflight and heterogeneity rules ([13](13-HARDWARE-AGNOSTIC-EXECUTION.md),
[14](14-HETEROGENEITY-POLICY.md)), and the validators ([15](15-PROVENANCE-VALIDATORS.md)).

Two additional rules follow from the subject:

- **Engine and hardware are named together, always.** `#34` already cost this project a claim — an engine
  comparison where *"Hardware also differs (H200 vs A100), and 3 of the 21 flips are tokenizer-caused"* — and
  `#43` is the supporting claim contradicting itself. A speculative-decoding comparison that changes engine
  and provider at once measures neither.
- **Tokenizer-caused flips are counted, not assumed away.**
  `results/benchmarks/h200/vllm_vs_llamacpp_agreement.json` already records 21 decision flips with `3`
  attributed to the tokenizer, plus 1,271 of 1,277 exact token-id matches. Study 004 reports that count for
  every comparison it makes.

## What Study 004 may not do

- It may not report a speedup without the quality delta from the same run, in the same table, at the same
  `n`. The constraint is a conjunction, not two sections of a paper.
- It may not use a quality metric that cannot see the `ANSWER` mode. That population is Study 002's first
  deliverable precisely so later studies can inherit it.
- It may not present `BLOCKED_MISSING_MTP_COMPONENT` as a partial success. A mode that cannot run is
  `NOT_EVALUABLE`, which is a status, not a score.
- It may not vary quantization or dtype inside a mode comparison. Those are part of `device_class`, and each
  combination is its own row.

## Sequencing

Study 004 is blocked on runtime support for at least one non-autoregressive mode, and on Study 002's
four-mode partition. Its first deliverable is a pre-registration whose mode table marks each mode
`LIVE`/`NOT_IMPLEMENTED` and whose `ε` values are derived from the resolvable margins of the populations it
will actually use. A study that pre-registers an `ε` its partitions cannot resolve would repeat the
threshold-before-the-thing-exists error that the scope split in [README](README.md) exists to avoid.