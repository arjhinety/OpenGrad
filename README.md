<div align="center">

<img src="assets/opengrad-icon.png" alt="OpenGrad icon" width="140" />

# OpenGrad

**Every gradient is a hypothesis. Every checkpoint is evidence.**

### → [Study 001: We improved the metric. Then the model stopped answering.][study-001]

<sub>The full finding, with every number recomputed from per-example records.</sub>

</div>

OpenGrad is an empirical research repository for capability–efficiency tradeoffs in small
open-weight language models. It provides reproducible infrastructure for controlled post-training,
regression-aware evaluation, and provenance-preserving publication — and it keeps negative results
on the record rather than editing them away. The first study is reliable tool use in Qwen3.5-2B.
OpenGrad is the premier frontier research repository of the [Experimental Intelligence](https://experimentalintelligence.org/) lab, fully maintained by
Arjhine Ty.

[![CI](https://github.com/arjhinety/OpenGrad/actions/workflows/ci.yml/badge.svg)](https://github.com/arjhinety/OpenGrad/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-052B42?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-052B42?style=flat-square)](LICENSE)
[![Research status](https://img.shields.io/badge/research-Building%20in%20Public%20%7C%20M1--v2%20DPO%20promoted-052B42?style=flat-square)](reports/M1_DPO_EVALUATION.md)
[![Hugging Face models](https://img.shields.io/badge/Hugging%20Face-models-FFD21E?style=flat-square)](https://huggingface.co/collections/arjhinety/opengrad-models-6aa3c7ea9ae58be5adbb113e)
[![Hugging Face datasets](https://img.shields.io/badge/Hugging%20Face-datasets-FFD21E?style=flat-square)](https://huggingface.co/collections/arjhinety/opengrad-datasets-and-evaluation-records-6aa3c7eb4514d9bc27e5d160)

---

## Current Research Status

| Item | Current state |
|---|---|
| Base model | `Qwen/Qwen3.5-2B` at revision `15852e8c…` — the only executed target so far |
| Canonical dataset | **Canonical-v2 final** — 173,237 records, 161,966 trainable, 4 sources, fingerprint `8ced403b…` |
| Latest completed stage | **General-capability diagnosis** across Base → M0 → M1-v2 on real IFEval, GSM8K and MMLU-Pro. It found a **general-capability regression associated with tool-policy post-training** that the tool-policy gate could not see. It appears after SFT (M0) and also after DPO applied directly to Base (M1-v1), so it is not specific to SFT |
| Promoted model | [`OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2`](https://huggingface.co/arjhinety/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2) — checkpoint 30, promoted under the parent-relative `tool_use_promotion_v4` gate. **Improved on tool policy over Base; within noise of its M0 parent; materially worse than Base on general capability. Not an unqualified improvement** |
| Studies | **Study 001 frozen 2026-09-13** at tag [`study-001`](https://github.com/arjhinety/OpenGrad/tree/study-001). **Study 002 in progress, no results yet:** refusal relabelling ([`ROADMAP.md`](ROADMAP.md) step 16, **BLOCKED_ON_PREFLIGHT**), M2 on-policy distillation (unexecuted scaffold) and speculative decoding. See [`docs/research/STUDIES.md`](docs/research/STUDIES.md) |
| Behavioral held-out | When2Call — 3,650 examples. **Contains no ANSWER examples**, which is why the regression escaped promotion |
| Largest current limitation | One 2B model, one lineage, no replicate. The promoted checkpoint refuses 100% of bare arithmetic questions |

Every experiment record, its validity, selected checkpoint, and published artifact: [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) (generated from the run artifacts).

## Start Here

| I want to… | Start here |
|---|---|
| Read the Study 001 write-up | [Study 001 results][study-001] — the narrative version of the finding |
| See the latest results | [Latest Results](#latest-results) · [`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md) |
| Download the promoted model | [Hugging Face model](https://huggingface.co/arjhinety/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2) |
| Browse every Study 001 model and dataset | [OpenGrad Study 001 collection](https://huggingface.co/collections/arjhinety/opengrad-study-001) — in reading order, with a note per item |
| Run or inspect OpenGrad | [Quick Start](#quick-start) |
| Check what is executed vs. planned | [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) · [`ROADMAP.md`](ROADMAP.md) |
| Reproduce an experiment | [Baseline reproduction protocol](docs/experiments/BASELINE_REPRODUCTION_PROTOCOL.md) |
| Understand the methodology | [Research program](docs/research/research-program.md) · [Methodology](docs/research/methodology.md) |
| Add a benchmark or backend | [Adding a benchmark](docs/evaluation/ADDING_A_BENCHMARK.md) · [Adding a backend](docs/evaluation/ADDING_A_BACKEND.md) |
| Plan or publish a Study 002 experiment | [Guardrails](docs/research/GUARDRAILS.md) — rules and checklists derived from the 100 claims Study 001 got wrong |
| Challenge a result | [CONTRIBUTING.md](CONTRIBUTING.md) · [Negative-result guide](docs/contributing/negative-result.md) |

**Contents:** [Current status](#current-research-status) · [Quick start](#quick-start) · [Latest results](#latest-results) · [How it works](#how-opengrad-works) · [Research program](#research-program) · [Datasets & evaluation](#datasets--evaluation) · [Limitations](#current-limitations) · [Reproducing](#reproducing-experiments) · [Repository structure](#repository-structure) · [Contributing](#contributing)

## Quick Start

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/). These commands are CPU-only: they
validate registries and fixtures and do not download models, run inference, or produce a benchmark
score.

```bash
git clone https://github.com/arjhinety/OpenGrad.git
cd OpenGrad
uv sync --extra dev

uv run opengrad-validate     # validate registries and schemas (the CI gate)
uv run opengrad-preflight    # capture the local environment and run fixture preflight
```

Useful inspection commands:

```bash
uv run opengrad status                                  # authoritative repository + experiment state
uv run opengrad experiment list                         # experiment records and lifecycle states
uv run opengrad results show                            # derived result index
uv run opengrad results validate-registry               # index vs. authoritative-artifact drift
uv run opengrad readiness configs/experiments/m0_sft_canonical_v2_final.yaml
```

`opengrad readiness` reports blocking gates that depend on local evidence and hardware
(materialized corpus, contamination adjudication, GPU probe, baseline artifacts), so a fresh clone
reports `FAIL` on those gates by design; the registry and configuration checks still pass.

> **Infrastructure validation is not an ML result.** Passing CPU tests validates contracts and
> fixtures, not model quality. The empirical results are in [Latest Results](#latest-results).

## Latest Results

The primary trajectory of Study 001. Scores are `call_f1` / `call_recall` on the pre-registered
confirmatory partition (n=1,277), single seed, no interval. On the full held-out set (n=3,650) B0
scores 0.6191 / 0.9722. The same results written as a narrative, with the charts:
[Study 001 results][study-001].

| Stage | Intervention | Primary result | Status |
|---|---|---|---|
| **B0** | Base `Qwen/Qwen3.5-2B` | 0.6264 / 0.9735 — from a degenerate policy that calls on 62% of no-call items | Baseline |
| **M0** | Canonical-v2 SFT (checkpoint 1800) | 0.7470 / 0.7594 | Selected; **not promoted** under `tool_use_promotion_v3` (recall-regression check against B0) |
| **M1** | M1-v2 DPO calibration (checkpoint 30) | **0.7548 / 0.7748** | **PROMOTED** under `tool_use_promotion_v4` (checks against the M0 parent) |
| **M2** | On-policy distillation | — | **Not executed** in Study 001 (mock-only scaffold); in scope for Study 002 |

**"Not promoted" → "PROMOTED" is a gate change, not a measured improvement.** M1-v2 is promoted
under a parent-relative gate (v4) introduced after M0 was evaluated; M0 also clears v4, and M1-v2
fails the v3 gate that rejected M0 (all four DEV checkpoints `REJECT`; DEV recall 0.7257 against
B0's 0.9715). The promotion reflects the gate change; the measured difference from M0 (+0.0078
call_f1, 7 of 453 calls, single seed) is within noise.

- Full intervention record, caveats, and published artifacts: [`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md)
- Promoted evaluation: [`reports/M1_DPO_EVALUATION.md`](reports/M1_DPO_EVALUATION.md)
- Derived result index: [`results/registry.jsonl`](results/README.md)
- Corrections to frozen reports and published artifacts: [**Errata**](reports/ERRATA.md)
- Claim audit, 93 findings across this repository, the Hugging Face cards and the project site, each with its resolution: [**claim-audit.pdf**](https://opengrad.arjhinety.com/studies/001/claim-audit.pdf)

Seven post-training interventions have been executed in total: one negative on corpus v1, one
partial recovery on corpus v2, the definitive final-v2 (selected, not promoted), two joint-removal
ablations (both negative), a rejected historical DPO attempt applied directly to Base, and the
promoted M1-v2. Negative results are not summarized away here; the full record is in
[`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md).

### The tool-policy gain came with a general-capability regression

Those `call_f1` numbers are real. They are also not the whole picture. A later diagnosis across the
checkpoint ladder, on **real upstream IFEval, GSM8K and MMLU-Pro**, found a regression the
tool-policy gate was structurally unable to detect.

| measure | Base | M0 — SFT | M1-v2 — promoted |
|---|---:|---:|---:|
| GSM8K zero-shot accuracy | **67.4%** | **0.0%** | **0.0%** |
| GSM8K zero-shot refusal rate | 0.0% | **100.0%** | **100.0%** |
| GSM8K 8-shot accuracy *(same questions)* | **70.4%** | 56.3% | 55.5% |
| IFEval prompt-level strict | **67.8%** | 45.1% | 45.8% |
| MMLU-Pro (5-shot, 12,032 items) | **49.0%** | 37.0% | 37.0% |

**Two separable failures, both first observed at M0 on the primary lineage:**

1. **A prompt-regime-conditioned refusal policy.** The promoted checkpoint declines every bare
   arithmetic question — yet with eight worked exemplars it answers all 1,319 of *the same
   questions* (answer rate 100%) and gets 55.5% of them right. It never refuses 5-shot MMLU-Pro.
   Refusal is conditioned on the request shape, not the subject.
2. **Genuine capability loss.** The 8-shot and MMLU-Pro gaps occur where refusal is ~0%, so they
   cannot be explained by declining to answer.

**The M0 → M1-v2 DPO step did not materially change either on this suite** — it moves every metric
by under 1pp, and 82% of GSM8K generations are byte-identical between the two.

**What the regression is associated with.** It is a general-capability regression associated with
tool-policy post-training on When2Call-derived data. It appears after SFT (M0) and also after DPO
applied directly to the base (M1-v1), so it is not specific to SFT. M1-v1 (`qwen35_2b_m1_dpo_v1`,
checkpoint 300) has no SFT parent (`parent_experiment_id: null`, reference = the initial policy,
preference data `when2call_pref_v1`), and it refuses **70.7%** (933/1,319) of the same zero-shot
GSM8K questions, against Base's 0%. Its 8-shot accuracy (70.4%) matches Base, its IFEval
prompt-level strict is 43.4%, and its MMLU-Pro at the corrected budget is unmeasured. Causation is
not established: one lineage, one seed, no replicate.

The original MMLU-Pro measurement used a 768-token budget and reported 38.0 / 36.8 / 36.9 —
apparently no regression at all. That was an artifact: Base hit the cap on 37.6% of items against
M0's 7.2%, so the benchmark was measuring verbosity. Re-run at 2048 tokens, the observed gap is
12.0pp — a point estimate. Base still hits the cap on 21.8% of items against M0's 6.3%, so the
truncation-adversarial interval is **6.4–33.1pp**: the gap stays positive however the truncated
items resolve, but its size is not pinned down. The superseded 768-token runs for these three
stages are retained, not deleted.

- Independent audit of every number above: [`reports/FINAL_CAMPAIGN_AUDIT.md`](reports/FINAL_CAMPAIGN_AUDIT.md)
- Full diagnosis: [`reports/GENERAL_CAPABILITY_REGRESSION.md`](reports/GENERAL_CAPABILITY_REGRESSION.md)
- Machine-readable verdict: [`results/final_campaign_verdict.json`](results/final_campaign_verdict.json)

## What OpenGrad Is

Small open-weight models can run locally and cheaply, but their capabilities trade off in ways a
single headline score hides. OpenGrad asks whether controlled post-training can improve a specific
behavior — when and how a model uses tools — while measuring what regresses, at what cost, and with
what reproducibility. A recipe is not better because one metric rises: every intervention is
measured, every regression matters, and every reported result must be reproducible.

- Why these questions: [From deployment problems to research questions](docs/research/motivation.md)
- How claims are made: [Methodology](docs/research/methodology.md) · [Reproducibility](docs/research/reproducibility.md)
- What is retained: [Negative results](docs/contributing/negative-result.md) · [Incident log](docs/INCIDENT_LOG.md)

## How OpenGrad Works

```mermaid
flowchart TD
    A[Source datasets] --> B[Canonicalization<br/>+ contamination audit]
    B --> C[Training<br/>SFT / DPO]
    C --> D[Evaluation<br/>frozen behavioral held-out]
    D --> E[Regression + promotion gate]
    E --> F[Published checkpoints<br/>+ reproducible reports]
```

Identity, semantics, and evidence are separated: `registry/` declares what things are, `src/opengrad/`
implements canonical data and evaluation contracts, `runs/` owns experiment state, and `reports/`
holds the written analyses. Deep architecture: [repository architecture](docs/architecture/repository.md)
and the [experiment foundation](docs/EXPERIMENT_FOUNDATION.md).

OpenGrad grew out of **OpenWeights**, an independent Android project that runs open-weight models on
consumer hardware: its deployment record exposed tool-use and efficiency problems that model
capability claims tend to hide. OpenGrad turns those observations into controlled model-level
experiments, and compatible checkpoints may later return to OpenWeights for device validation. The
projects are independently maintained. See [motivation](docs/research/motivation.md) and
[on-device testing](docs/evaluation/OPENWEIGHTS_ON_DEVICE_TESTING.md) — no device study has been
executed yet.

## Research Program

| RQ | Question |
|---|---|
| RQ1 — Reliable tool policy | Can controlled post-training improve **when and how** small models use tools, beyond the tool syntax already present in their instruct checkpoints? |
| RQ2 — Transfer | Do those improvements transfer into realistic agent runtimes and workload patterns? |
| RQ3 — Regression | What instruction-following, reasoning, calibration, latency, or robustness capabilities regress as tool reliability improves? |
| RQ4 — Distillation | When SFT reaches its ceiling, can on-policy distillation from a larger model improve the remaining decision-boundary failures? |
| RQ5 — Constrained inference | Can target-attached/native speculative decoding improve decode efficiency without the cost of a separate draft model? |
| RQ6 — Capability × efficiency | Do post-training gains survive quantization and optimized constrained-device inference? |

Full statements, the first study's scope, and the experimental decision pipeline:
[research program](docs/research/research-program.md). Stage-by-stage status: [`ROADMAP.md`](ROADMAP.md).

## Datasets & Evaluation

| Dataset release | Records | Role |
|---|---:|---|
| [Canonical-v2 final](https://huggingface.co/datasets/arjhinety/OpenGrad-ToolPolicy-Canonical-v2) | 173,237 (161,966 trainable) | **Current** training corpus |
| [Canonical-v1](https://huggingface.co/datasets/arjhinety/OpenGrad-ToolPolicy-Canonical-v1) | 213,951 | Historical; training corpus of the negative M0-v1 run (B0's record pins only the held-out manifest, not a corpus) |
| [Partial-v2 snapshot](https://huggingface.co/datasets/arjhinety/OpenGrad-ToolPolicy-Canonical-v2-M0-snapshot) | 103,036 | Historical; corpus behind the first successful M0 |

Evaluation layers, and what has actually run in each:

| Layer | Purpose | Current status |
|---|---|---|
| Behavioral held-out | Tool-use policy (when to call, answer, clarify, refuse) | **Executed** — When2Call, 3,650 examples |
| External capability | General regressions and transfer | **3 of 17 executed** — IFEval, GSM8K, MMLU-Pro (Tier B) on Base → M0 → M1-v2; the other 14 are frozen, not executed |
| Agent / runtime | Realistic transfer in agent loops | Planned |
| Systems | Latency, throughput, memory, spec decoding | Planned |

The three executed external benchmarks ran in the H200 capability-diagnosis campaign with the
upstream scorers (IFEval 541 prompts, GSM8K 1,319 questions in two arms, MMLU-Pro 12,032 items);
results are in [`results/benchmarks/h200/capability_v1/`](results/benchmarks/h200/capability_v1/)
and the verdict in [`results/final_campaign_verdict.json`](results/final_campaign_verdict.json).
The runs under `reports/benchmarks/` are different: they used the deterministic mock backend and
validate only the result contract. Details: [benchmark inventory and counting convention](docs/benchmarks/README.md) ·
[benchmark strategy](docs/evaluation/BENCHMARK_STRATEGY.md) · [dataset releases](docs/publishing/huggingface-datasets.md) ·
[supervision contract](reports/SUPERVISION_CONTRACT_REPORT.md).

## Current Limitations

Claims in frozen reports and published artifacts that were later found wrong are corrected in the
[**Errata**](reports/ERRATA.md) rather than edited in place. The
[**claim audit**](https://opengrad.arjhinety.com/studies/001/claim-audit.pdf) that found them lists every finding
and how it was resolved.

- **Scope:** the empirical record is one model family (Qwen3.5-2B). No cross-model replication has run.
- **External benchmarks:** 3 of the 17 Tier A–E benchmarks have real scores (IFEval, GSM8K, MMLU-Pro); the tool-use (Tier A), agent-transfer (Tier C), stretch and systems tiers are prepared but not executed.
- **On-policy distillation:** deliberately not run. M1-v2 exposed no calibration failure that teacher guidance would address, so M2 was closed as `NOT RUN / NOT JUSTIFIED` ([decision](reports/M2_DECISION.md)); the trainer is a mock-tested scaffold whose live path refuses to run, and the one recorded M2 run is mock-only (`INVALID`).
- **Quantization (GGUF):** executed. Nine PTQ rungs were built from a BF16 GGUF with a frozen
  importance matrix and scored on the 1,277-example confirmatory partition. Only **Q6_K (1.45 GiB)
  and Q8_0 (1.87 GiB)** pass `quantization_preservation_v1`; every rung at Q5_K_M and below fails.
  Under the pre-registered smallest-passing-format rule the recommended release is **Q6_K** (the
  closure manifest records `recommended_release_rung: Q6_K`); neither passing rung meets the
  stricter secondary release bar. The pass/fail line is inside run-to-run noise: an H200 vLLM BF16
  rerun of the unquantized reference itself fails the gate (recall 0.7638 < 0.7671 floor), and Q6_K
  clears the precision floor by one example (357/490 against 356.96 required).
  Strict engine tokenizer parity **FAILED** on 6/1,277 prompts (a stock-llama.cpp `\p{M}`
  pre-tokenizer difference, all Thai) and was not redefined — see
  [`QUANTIZATION_ENGINE_PARITY.md`](reports/QUANTIZATION_ENGINE_PARITY.md) and
  [`QUANTIZATION_PTQ_EVALUATION.md`](reports/QUANTIZATION_PTQ_EVALUATION.md).
- **Quantization (ExecuTorch):** CPU/XNNPACK fp32 and 8da4w exported and audited. 99.98% of the
  named-data weights are int4, but overall int4 coverage is **78.7%** of stored parameters: the
  embedding table stays fp32 and is 65.7% of the quantized artifact. **No behavioural verdict** —
  scoring is run externally. Snapdragon is `REJECTED_EXPORT`; MediaTek is `BLOCKED_PORT_INCOMPLETE`
  (SDK now obtained, Gated DeltaNet layer unimplemented).
- **Device and speculation:** integration or reservation only; no on-device study, optimization
  run, or speculative benchmark exists. Every throughput figure is a datacentre number, not a
  device number: the GGUF ladder's are A100-80GB, and the H200 vLLM saturation sweep and
  performance microsuite are in [`reports/H200_BENCHMARK_RUN.md`](reports/H200_BENCHMARK_RUN.md).
- **Reproducibility gaps:** the historical M1-v1 DPO best checkpoints no longer exist, and some scaffold-era runs retain only metadata.
- **Measurement coverage:** tool selection, argument validity, and schema validity are not computed by the current evaluator.

## Reproducing Experiments

The B0 baseline, the M0 SFT lineage (including both ablations), and the M1 DPO runs are recorded:
experiment records and ledgers for all of them, and committed resolved configs, seeds and dataset
hashes for every training run (B0's environment and predictions sit under
[`reports/baselines/`](reports/baselines/qwen35_2b_baseline/RESULT.md)). Recorded is not the
same as reproduced: the only repeat ever attempted, `qwen35_2b_m1_dpo_v1_restore` (M1-v1 re-run
with config, data, seed and environment pinned), did not reproduce the original trajectory and is
annotated `NON_REPRODUCIBLE_REPEAT`. No other run has been repeated, so every other result here is
a single, unreplicated run. The baseline workflow is specified in the
[baseline reproduction protocol](docs/experiments/BASELINE_REPRODUCTION_PROTOCOL.md): acquire the
accelerator, fetch the exact model revision, validate its native template and parser, run sanity
checks, execute the selected evaluations, compare revisions, and pass the reproduction gate. That
protocol is not executed by the CPU-only Quick Start commands.

Seeds, dataset hashes, model and tokenizer revisions, hardware, and failure analysis are recorded
per run under `runs/<experiment_id>/`; see [reproducibility](docs/research/reproducibility.md) and
the [results namespace](results/README.md).

## Repository Structure

```text
registry/       dataset, benchmark, model, runtime, and experiment contracts
src/opengrad/   canonical data, adapters/renderers, evaluation, gates, reporting
configs/        versioned experiment, data, evaluation, and inference definitions
runs/           authoritative experiment state (experiment.json, eval/, ledgers)
results/        derived, rebuildable result index
reports/        experiment analyses, baselines, releases, incident records
docs/           methodology and detailed technical documentation
integrations/   harness-facing integration (opengrad-mcp stdio server)
```

Full layout and boundaries: [repository architecture](docs/architecture/repository.md).

## Contributing

You do not need to agree with a result to contribute. Showing that a finding fails to reproduce is
valuable research: reproduce it with another seed or model family, challenge a dataset assumption,
compare inference implementations, or submit a failed reproduction. Start with
[CONTRIBUTING.md](CONTRIBUTING.md), the [reproduction PR guide](docs/contributing/reproduction-pr.md),
and the [negative-result guide](docs/contributing/negative-result.md). Do not commit checkpoints,
bulk datasets, credentials, or fabricated results.

Operate OpenGrad from an agent harness through the same documented `opengrad … --json` CLI; see the
[agent integration guide](docs/AGENT_INTEGRATION.md) and the
[MCP server](integrations/opengrad-mcp/README.md).

## Related Projects

- [OpenWeights](https://github.com/alpharomercoma/openweights) — independent on-device execution environment for compatible GGUF/llama.cpp and ExecuTorch artifacts; OpenGrad defines experiments and evidence, OpenWeights runs compatible artifacts.
- [OpenPapers](https://github.com/arjhinety/OpenPapers) — provenance-preserving scholarly retrieval used during active research; its findings are research inputs, not OpenGrad results.
- [Experimental Intelligence](https://experimentalintelligence.org/) — the research lab OpenGrad is the premier frontier research repository of, fully maintained by Arjhine Ty. It grew out of [Experimental Machines](https://experimentalmachines.org/), the same team's first effort, which benchmarks LLM inference on datacenter, laptop and phone hardware. When the team moved into ML research, that work became Experimental Intelligence.

OpenGrad records upstream datasets, models, benchmarks, and papers in its registries; those sources
remain subject to their own licenses and attribution requirements.

## Citation and License

Please cite the repository using [CITATION.cff](CITATION.cff) until a formal release DOI exists.
Contact: Arjhine Ty, Experimental Intelligence — [arjhine@experimentalmachines.org](mailto:arjhine@experimentalmachines.org).

The Hugging Face namespace was renamed from `arrochi112` to `arjhinety` on 2026-09-13. Old links
redirect, and committed records (run manifests, publication records, frozen reports) keep the name
they were written with.
Source code and documentation are licensed under [Apache-2.0](LICENSE).

---

<sub>Documentation map: this README is the landing page. Methodology and technical detail live in
[`docs/`](docs/), empirical analyses in [`reports/`](reports/), authoritative state in
[`runs/`](runs/) and [`registry/`](registry/), and the derived index in
[`results/registry.jsonl`](results/README.md). The last documentation restructure is recorded in
[`docs/README_INFORMATION_ARCHITECTURE_AUDIT.md`](docs/README_INFORMATION_ARCHITECTURE_AUDIT.md).</sub>

<!-- Study 001 site. Referenced from the header, Start Here, and Latest Results.
     Deployment host is provisional — change this one line when the domain moves. -->
[study-001]: https://opengrad.arjhinety.com/studies/001
