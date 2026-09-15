# 09 — Benchmark plan

## What the registry actually says

`registry/benchmarks.yaml` registers 23 benchmarks with `status: BEHAVIORAL_SCORED_EXTERNAL_NOT_EXECUTED`,
and states the situation in its own header:

> *"The behavioral held-out (when2call-eval) has executed scores: B0 and 12 candidate checkpoints were
> measured on it, and those results are published. Every **external** benchmark in this registry is still
> `FROZEN_NOT_EXECUTED` and has no score, so the registry as a whole is not 'bootstrap' any more — only its
> external tier is unmeasured."*

Its tiering is `TIER_A` (3: `bfcl-v4`, `tau3`, `acebench`), `TIER_B` (6: `ifbench`, `ifeval`, `livebench`,
`mmlu-pro`, `gsm8k`, `arc-challenge`), `TIER_C` (5: `mcpmark`, `agentbench-fc`, `terminal-bench`,
`tua-bench`, `openweights`), `TIER_D` (1: `gaia`), `TIER_E` (2: `performance-microsuite`,
`speculative-replay`), plus six entries without a tier (`when2call-eval`, `tau-bench-tau2`, `toolsandbox`,
`mcpmark-verified`, `toolathlon`, `internal-no-tool-regression`).

`ROADMAP.md:55-57` records what has actually run: *"External benchmark families were `FROZEN_NOT_EXECUTED`
at this step; IFEval, GSM8K and MMLU-Pro were later executed on Base → M0 → M1-v2 (step 16), and the other
14 remain frozen."*

## What Study 002 runs

| Benchmark | Tier | `evaluator_version` | Mode | Endpoint |
|---|---|---|---|---|
| `ifeval` | `TIER_B` | `opengrad-ifeval-eval-v1` | 0-shot | `prompt_level_strict_acc`, `prompt_level_loose_acc`, failure classes |
| `gsm8k` | `TIER_B` | `opengrad-gsm8k-eval-v1` | 0-shot, 8-shot, elicit | accuracy, `answer_rate`, `accuracy_given_answer` |
| `mmlu-pro` | `TIER_B` | `opengrad-mmlupro-eval-v1` | 5-shot @2048 | accuracy + truncation-adversarial interval |
| `when2call-eval` | untiered | behavioural evaluator | 0-shot | four per-mode accuracies, `call_f1`, `over_call_rate` |
| `internal-no-tool-regression` | untiered | behavioural evaluator | 0-shot | the `ANSWER`-mode guard (this study's addition) |

The three external instruments are the ones that already exposed the regression, and they are run because
their numbers are the *comparison basis*: an intervention on the corpus must be shown not to move them
adversely, and a study that changed the instruments would lose the ability to say whether it had made
things better or merely different.

Study 002 does not run `TIER_A` (`bfcl-v4`, `tau3`, `acebench`), `TIER_C`, `TIER_D` or `TIER_E`. The reason
is substantive, not budgetary. `acebench` would be a genuinely good instrument here — its splits include
`ambiguous`, `incomplete` and `impossible`, and its metrics include `clarification_rate` and
`refusal_accuracy` — but every `TIER_A` entry requires harness work (parser adapters, evaluator pinning,
environment construction) that is outside a corpus-mechanism study, and `bfcl-v4` and `tau3` are
contamination-sensitive with `prohibit_training: true` and headline metrics on tool-selection endpoints this
study does not claim to move.

**This is a limitation and is stated as one.** Study 002's external-validity claim is bounded by three
`TIER_B` instruments plus the behavioural partitions. Adding `acebench`'s `impossible` and `ambiguous`
splits is the single highest-value benchmark addition for a future study, because they measure refusal and
clarification correctness *externally* rather than in a population this study curated itself.

## Instrument discipline

Every run pins, from the registry: `id`, `commit_sha`, `evaluator_version`, `splits`, `metrics`,
`parser_requirements`, `contamination_sensitivity` and the generation budget. A number without its
`evaluator_version` is not reportable, because two evaluator versions are not comparable — and Study 001's
`#34` is the case in point: *"Hardware also differs (H200 vs A100), and 3 of the 21 flips are
tokenizer-caused"*, where an engine comparison was confounded by hardware and by three tokenizer-caused
flips.

Three further rules follow from the registry's own fields:

- **`prohibit_training: true` on every benchmark used.** No benchmark item, in any split, may appear in a
  training corpus. The `ANSWER` strata set ([06](06-SPLIT-SPEC.md)) is checked against this prohibition
  before use; `contamination_sensitivity: HIGH` is why.
- **`parser_requirements: deterministic-evaluator`** for `ifeval` means the scorer is deterministic and its
  determinism is verified by double-scoring the same generations file (`docs/EVALUATION.md:54-60`, and the
  measured failure: 0.4510 / 0.4492 / 0.4492 on one unchanged file).
- **The registry's metric names are the vocabulary.** `ifeval`'s four instruction-level names
  (`prompt_level_strict_acc`, `prompt_level_loose_acc`, `inst_level_strict_acc`, `inst_level_loose_acc`) are
  not interchangeable, and "IFEval strict" in a Study 002 report always means `prompt_level_strict_acc`.

## Generation budgets

| Benchmark | Budget | Rationale |
|---|---|---|
| `ifeval` | the frozen evaluator's own config | prompt-level scoring is length-sensitive; changing the budget is a protocol change |
| `gsm8k` | the frozen config, identical across arms and modes | answers are short; the 0-shot/8-shot comparison must not differ in budget |
| `mmlu-pro` | **2,048** | 768 was too small for 5-shot CoT over ten options, and it did not bite evenly |

The MMLU-Pro budget is 2,048 because the 768-token pass was **discarded**: Base truncated on 4,527 of
12,032 (37.6%) against M0's 869 (7.2%), and 99.6% of Base's unattempted examples (4,292 of 4,309) were
truncations rather than refusals, so the pass measured verbosity
(`reports/GENERAL_CAPABILITY_REGRESSION.md:122-137`). The pass was re-run and its cost counted in full. At
2,048 a residual imbalance remains — Base 2,627 (21.8%) against M0 759 (6.3%) — so MMLU-Pro is reported as
a truncation-adversarial interval, never as a point, and never as a "lower bound" without a derived bound.

Adopted as a standing rule: **a generation budget is part of the measurement.** It may be changed only
before any response is scored, uniformly across every stage, and the change is recorded as a pre-scoring
protocol amendment (`docs/EVALUATION.md:43-44`).

## Benchmark versus sentinel

A benchmark is a frozen external suite scored once on the confirmatory population. A sentinel
([08](08-SENTINEL-SPEC.md)) is an internal guard run frequently across checkpoints during selection. They
are never merged in one table, and a sentinel total never appears in a benchmark column. The 7-case
OpenWeights sentinel is the cautionary example: it reported 5/7 for two models with opposite behaviour, and
at n = 7 its worst-case resolvable margin is 74pp.

## Cost and honesty about scope

Each arm's benchmark pass records its container time and cost in the existing ledger format — `gpu`,
`gpu_hourly_usd`, `container_seconds`, `usd`, `arm`, `seed`, `tier`, `disposition` — and the total is
reported. The rate is recorded per entry rather than assumed, because the Study 001 diagnosis ran at
$4.54/GPU-hour on an H200 while the training anchor is $1.79/GPU-hour on an A100 80GB: the same work differs
by 2.54× across providers, and the per-arm evaluation estimate of ≈ $1–3 (so ≈ $35–105 across 35 runs) is
derived at the A100 rate from one measured input, a single MMLU-Pro @2048 stage run at 732.53
container-seconds. The basis, the assumption ladder and the rule that a planning envelope is not available
credit are in [`16-GPU-READINESS-GATE.md`](16-GPU-READINESS-GATE.md).

A pass that was superseded, or whose results were discarded, still appears with its cost: a cost ledger that
hides discarded work makes the next study's budget look cheaper than it is. That is exactly what the
768-token MMLU-Pro pass would have done had its cost not been counted — it is retained at $3.8864 and 3,081.7
container-seconds in `results/benchmarks/h200/capability_v1/cost_ledger.json`.