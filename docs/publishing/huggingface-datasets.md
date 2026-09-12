# Hugging Face dataset publication

OpenGrad uses a two-repository publication boundary.

GitHub is the canonical home for source code, schemas, manifests, provenance, audits, reproduction commands, and experiment definitions. Hugging Face is the distribution home for large canonical dataset artifacts, dataset cards, source attribution, release manifests, and checksums.

## Current releases

| Release | Records | Sources | Status | Hub revision | Publication record |
|---|---:|---:|---|---|---|
| `OpenGrad-ToolPolicy-Canonical-v2` | 173,237 canonical / **161,966 trainable** | 4 | **Current** | `66470c07ed0a79941f49a5cf67c1b3b1a7d8196e` | `reports/releases/toolpolicy-canonical-v2-publication.json` |
| `OpenGrad-ToolPolicy-Canonical-v1` | 213,951 | 6 | Historical (pinned by B0 and all earlier results) | `bb295d8a4ad64f7e8161044ad2fa34f873ede418` | `reports/releases/toolpolicy-canonical-v1-publication.json` |
| `OpenGrad-ToolPolicy-Canonical-v2-M0-snapshot` | 103,036 | 3 of 6 | Historical experiment snapshot | — | — |

Canonical-v2 final is the corpus the definitive M0 and M1 DPO ran on. Its corpus fingerprint is
`8ced403b996e563d6e279aee7fdb346fc829fe5ff6af9daf8ef47c0a4007e161` (the sha256 of the frozen
release manifest), proven reproducible by a delete-and-rebuild; the tracked definition is
`configs/releases/toolpolicy_canonical_v2_final.yaml`.

The partial-v2 snapshot is the corpus behind the first successful M0. It is distinguished from the
final v2 by name, fingerprint and source count, and is retained as historical evidence rather than
as the current corpus. **xLAM contributed zero gradients to it.**

## Supervision contracts

Canonical-v2 final declares exactly one rule per record, and the two contracts differ only in
whether a terminal call needs a future environment response. Everything else fails closed under
both: undeclared tools, invalid arguments, malformed calls, orphaned results, FIFO order violations.

| Contract | Trainable | What it supervises |
|---|---:|---|
| `COMPLETE_TRAJECTORY` | 105,876 | A full trajectory: every call answered by its result, ending in a terminal response |
| `CALL_PREDICTION` | 56,090 | Next-call prediction: the terminal tool call *is* the target, so no result is required |

Measured composition: `reports/SUPERVISION_CONTRACT_REPORT.md`. Contract semantics:
`docs/architecture/repository.md`.

## Source composition

| Dataset | Purpose in the program | v2 final support | Training eligibility | Provenance |
|---|---|---|---|---|
| xLAM / APIGen Function Calling 60k | Function selection and argument generation | 57,342 canonical, **56,090 trainable** under `CALL_PREDICTION` | SFT (corpus v1, v2 final) | Salesforce snapshot revision recorded; upstream access-gated, so v2 reconstructs from the published v1 derivative with per-record verification |
| Glaive Function Calling v2 | Additional function-calling coverage | 98,339 canonical, 97,112 trainable | SFT (corpus v1, v2 final) | HF snapshot revision recorded |
| ToolACE | Complex schemas, candidate tools, parallel/dependent calls, negatives | 11,051 canonical, 2,259 trainable (8,476 end on an unanswered call and stay quarantined) | SFT (corpus v1, v2 final) | Team-ACE source revision recorded |
| When2Call | Call/no-call decisions and answer quality | 6,505 records, yield 1.000 | SFT, preference, and evaluation remain separate | NVIDIA HF and GitHub sources recorded |
| BUTTON / BUTTONInstruct | Multi-turn compositional trajectories | Not included: upstream access-gated | Not trained on | Repository commit recorded |
| LoopTool-23k | Loop/tool trajectories requiring lineage audit | Not included: upstream not located | Not trained on | Source revision recorded; possible derivation overlap |

Source identity, pinned revisions, and per-source counts: `docs/data/normalization-sources.md`.
Capability support per source: `docs/data/source-support-matrix.md`. Corpus mixture hypotheses
(mixture-M0/M1/M2) and the training-phase disambiguation: `docs/data/tool-use-mixture-methodology.md`.

`Salesforce/APIGen-MT-5k` is excluded from the clean default because of possible τ-bench/τ² overlap.
If it is ever used it must carry the contaminated namespace, and its scores cannot be presented as
clean generalization (`configs/data/tool_calling/contamination.yaml`).


## Release scope

The release builder reads source manifests rather than hard-coding record counts. It writes ordinary Parquet with source-aware columns, including `source_dataset`, `source_record_id`, `source_revision`, `canonical_hash`, quality state, behavior metadata, and canonical tools/messages/metadata.

The intended source scope is six sources: xLAM, BUTTON, ToolACE, LoopTool, Glaive Function Calling v2, and When2Call SFT. Canonical-v2 final resolves four of them; BUTTON is access-gated upstream and the LoopTool source was not located. When2Call preference, MCQ, and LLM-judge artifacts are explicitly excluded. The frozen behavioral held-out (3,652 distinct items, 3,650 scored after two quarantines) is never embedded in a training or candidate release. Qwen-rendered text is also excluded; model-specific rendering remains a separate experiment artifact.

Tracked release configs are `configs/releases/toolpolicy_canonical_v1.yaml`, `..._v2.yaml`, and `..._v2_final.yaml`; dataset-card, licensing, and citation inputs live under `release/huggingface/`. Generated staging output belongs under `.release/` and is ignored.

The release includes xLAM under CC BY 4.0 as a normalized derivative, with attribution, APIGen citation, and modification disclosure. Its upstream access mode is gated, but downstream redistribution is explicitly permitted with attribution; the OpenGrad release is public and does not reproduce the upstream gate.

## Commands

Build local staging without uploading:

`uv run opengrad-data build-hf-release --release-config configs/releases/toolpolicy_canonical_v1.yaml --output .release/hf/toolpolicy-canonical-v1`

Validate staging:

`uv run opengrad-data validate-release --input .release/hf/toolpolicy-canonical-v1`

These commands do not invoke the Hugging Face upload CLI.

## Versioning

Release meaning is immutable. PATCH releases correct metadata or provenance without changing record semantics. MINOR releases add compatible sources/configurations or clearly documented compatible corrections. MAJOR releases change canonical schema or dataset meaning. Every experiment must pin the exact Hub revision and record the release manifest hash.

## Experiment lineage

Future model provenance must identify the upstream source, the exact OpenGrad canonical release revision, the experiment-specific mixture and config, experiment ID, checkpoint, and evaluation report. “Trained on OpenGrad” alone is insufficient.

## Upload gate

Before a future release update, recheck current upstream licenses, dataset cards, repository terms, gated conditions, attribution, and citation requirements. Validate the generated manifest, Parquet payloads, card, citations, license audit, checksums, counts, and absence of evaluation/preference leakage. Publications are recorded at Hub commits `bb295d8a4ad64f7e8161044ad2fa34f873ede418` (v1) and `66470c07ed0a79941f49a5cf67c1b3b1a7d8196e` (v2 final).
