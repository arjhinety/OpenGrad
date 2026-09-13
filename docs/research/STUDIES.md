# Studies

OpenGrad's research is published as numbered studies. Each study has a permanent landing page at
`https://opengrad.arjhinety.com/studies/<id>`, and the site's `/` shows the latest frozen study.

## Versioning rules

- **Freezing.** A study is frozen when its findings are final. The freeze is an annotated git tag
  (`study-001`, `study-002`, …) in this repository, and a tag of the same name in `opengrad-site`
  on the commit that froze its page. The study's page links its evidence at the tag, not at
  `master`, so a reader lands on the evidence the page was written from.
- **After a freeze, nothing in the study changes silently.** A number later found wrong in a frozen
  study is corrected in [`reports/ERRATA.md`](../../reports/ERRATA.md), the same way frozen,
  hash-pinned reports are corrected. The study's checkpoints, corpora and evaluations are not
  modified, relabelled or replaced.
- **New work goes into the next study.** Any experiment started after a freeze belongs to the next
  study. It is pre-registered before it runs, and it is reported against the frozen study's
  checkpoints, not in place of them.

## Study 001 — tool-policy post-training on Qwen3.5-2B

**Status: FROZEN 2026-09-13** at tag [`study-001`](https://github.com/arjhinety/OpenGrad/tree/study-001).
Landing page: [opengrad.arjhinety.com/studies/001](https://opengrad.arjhinety.com/studies/001).

It covers the B0 baseline, the five SFT arms, both DPO lineages, the H200 general-capability
diagnosis, the GGUF PTQ ladder and the OpenWeights ParitySuite run. On the confirmatory partition
(n=1,277), call F1 rose from 0.6264 (B0) to 0.7470 with SFT (M0) and 0.7548 after DPO (M1-v2,
+0.0078 over M0, within noise). M1-v2 was promoted under the parent-relative `tool_use_promotion_v4`
gate, introduced after M0 was evaluated. It refuses all 1,319 zero-shot GSM8K questions, a
general-capability regression associated with tool-policy post-training on When2Call-derived data
that the promotion gate could not see. Causation is not established.

- Full intervention record: [`docs/EXPERIMENT_RESULTS.md`](../EXPERIMENT_RESULTS.md)
- Diagnosis: [`reports/GENERAL_CAPABILITY_REGRESSION.md`](../../reports/GENERAL_CAPABILITY_REGRESSION.md)
  and [`reports/FINAL_CAMPAIGN_AUDIT.md`](../../reports/FINAL_CAMPAIGN_AUDIT.md)
- Corrections: [`reports/ERRATA.md`](../../reports/ERRATA.md) and the
  [claim audit](https://opengrad.arjhinety.com/studies/001/claim-audit.pdf), 93 findings with their
  resolutions

## Study 002 — relabelling, on-policy distillation and speculative decoding

**Status: IN PROGRESS. No results yet.** Landing page:
[opengrad.arjhinety.com/studies/002](https://opengrad.arjhinety.com/studies/002).

Scope, each part pre-registered before any GPU time is spent:

1. **Dataset corrections, starting with refusal relabelling** ([`ROADMAP.md`](../../ROADMAP.md)
   step 16, `BLOCKED_ON_PREFLIGHT`). A heuristic detector flags 18,114 of Canonical-v2's 173,237
   records as refusal-shaped targets labelled ANSWER. The detector's precision is measured on a
   hand-labelled sample first. Relabelling goes into a new corpus version; Canonical-v2 stays as
   published. M0 is retrained with only the labels changed, and GSM8K zero-shot, IFEval, MMLU-Pro
   and the tool-policy metrics are re-measured against pre-registered thresholds. A null result is
   publishable.
2. **On-policy distillation (M2, RQ4).** Study 001 did not run it: the trainer seams exist, but the
   live training path is unimplemented and the one recorded M2 attempt is mock-only
   ([`reports/M2_DECISION.md`](../../reports/M2_DECISION.md)). It needs a live path, a teacher that
   fits the available hardware, and its own confirmatory evaluation.
3. **Speculative decoding integration (RQ5, ROADMAP step 14).** Target-attached or native
   speculative decoding for decode efficiency without a separate draft model. The benchmark suite
   has a speculative tier; no runtime support exists and nothing has been measured.

Cross-model replication (step 11) and joint capability-efficiency studies (step 15) stay on the
roadmap and are not assigned to Study 002.
