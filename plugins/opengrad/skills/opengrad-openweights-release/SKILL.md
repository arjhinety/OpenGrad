---
name: opengrad-openweights-release
description: How to publish and derive OpenGrad artifacts correctly — Hugging Face dataset and model releases with publication records and cards, the authoritative publication gate (scripts/verify_publication.py) and repository hygiene scan, GGUF conversion, quantization and parity, ExecuTorch exports, the optimization producer layer (python -m opengrad.optimization), Modal remote jobs, and on-device OpenWeights testing. This skill should be used before uploading, tagging or announcing any artifact, for any change under release/, hf/, src/opengrad/publication/, src/opengrad/optimization/, scripts/modal/ or the GGUF/ExecuTorch/quantization scripts.
---

# OpenGrad open-weights release

GitHub is the canonical home for code, manifests, provenance and audits. Hugging Face distributes large
artifacts and cards (`docs/publishing/huggingface-datasets.md`). A release is not complete until its publication
record is committed and the publication gate passes.

## Publishing (any Hugging Face upload, card change or tag)

1. **Build and validate** the artifact. For corpora, use `opengrad-data build-hf-release` and
   `opengrad-data validate-release` (`opengrad-data-pipeline`).
2. **Cards** come from the templates in `hf/` and live under `release/huggingface/<repo>/`. They state lineage,
   the policy version that promoted the checkpoint, caveats (G12), and whether the checkpoint is text-only
   (every pre-`full-model-components-v1` checkpoint is).
3. **Upload only on the user's explicit instruction.** Remote jobs use `scripts/modal/upload_artifact.py`.
4. **Record** the Hub revision in a publication record under `reports/releases/`, and update the release table in
   `docs/publishing/huggingface-datasets.md` in the same commit (G16/G17).
5. **Run the gate:** `python scripts/verify_publication.py` (use `--offline` when the network is unavailable; it
   then reports BLOCKED, not PASS).
   - Exit 0 is PASS, 1 is FAIL (a provenance defect), and 2 is BLOCKED (a check could not run).
   - The census must show the required populations were examined. Implementation:
     `src/opengrad/publication/verify.py`.
6. **Hygiene:** `python scripts/repo/check_publication_hygiene.py` scans tracked text, data files (`.jsonl`)
   and tarball members for credentials (`ghp_`, `hf_`, `AKIA`, private keys), home-directory paths and
   assistant/prompt scaffolding language; CI runs it. Fix a finding. Only a finding that cannot be fixed
   (it sits in hash-pinned evidence, or is upstream content) goes into
   `scripts/repo/publication_hygiene_allowlist.yaml`, with an exact count, a reason and an ERRATA reference.
   Records that name files use `portable_path` (`src/opengrad/registry/provenance.py`), never an absolute path.
7. **Redistribution:** source terms are audited in `docs/publishing/source-redistribution-audit.md`. A source
   that may not be redistributed is not uploaded.
8. **Frozen studies:** a study's Hugging Face repositories are tagged at freeze, and later work goes to *new*
   repositories (`docs/research/STUDIES.md`).

Never commit or print tokens. If push protection flags a secret, verify it and let the user decide.

## GGUF, quantization and parity

- **Conversion:** llama.cpp `convert_hf_to_gguf.py` driven by `scripts/modal/gguf_study.py`. Supporting scripts:
  `scripts/run_gguf_ladder.py`, `scripts/prepare_imatrix_calibration.py`, `scripts/build_gguf_card.py` and
  `scripts/build_gguf_handoff.py`.
- **Parity and benchmarking:** `release/gguf/manifest.json`, `release/gguf/PARITY_NOTES.md`,
  `release/gguf/bench_gguf.py` and `release/gguf/score_generations.py`, with frozen prompts in
  `release/gguf/frozen_prompts_confirmatory_v1.jsonl`.
- **Preservation gate:** a quantized descendant must satisfy `quantization_preservation_v1`
  (`src/opengrad/promotion/quantization.py`); see `opengrad-promotion-gates`.
- **Pre-policy checkpoints** declare `mtp_num_hidden_layers: 1` without MTP tensors, so they convert with
  `--no-mtp`. A `full-model-components-v1` checkpoint carries MTP and vision. Its export must keep the MTP block
  and produce an mmproj, which is exactly the `gguf_export` check of the pre-training gate
  (`reports/training/model-components-validation.json`). Record the evidence there only after llama.cpp loads
  and generates.
- **Quantization studies and their closures:** `reports/QUANTIZATION_STUDY_PLAN.md` and
  `reports/PTQ_PHASE_CLOSURE.md`.

## ExecuTorch and on-device

- **Export:** `scripts/modal/executorch_export.py`, `scripts/build_executorch_handoff.py` and
  `scripts/publish_executorch_artifact.py`. Released bundles live in `release/executorch/`, and
  `integrations/executorch/` holds the integration.
- **On-device testing and parity grading:** `docs/evaluation/OPENWEIGHTS_ON_DEVICE_TESTING.md`,
  `scripts/grade_openweights_parity.py` and the benchmark config `configs/benchmarks/openweights.yaml`.
  OpenWeights is an independent downstream environment, not a dependency (`docs/openweights/README.md`).

## Optimization producer layer

Optimization **changes artifacts**; it never trains, executes or measures (`docs/optimization/README.md`).

```bash
python -m opengrad.optimization capability-matrix
python -m opengrad.optimization optimize --backend mock ...
```

- The source checkpoint is read-only (`ensure_distinct_output`), and results are pinned by `recipe_hash`.
- The ModelOpt backend is an optional extra (`docs/optimization/modelopt-integration.md`,
  `configs/optimization/qwen35_2b_fp8_ptq_smoke.yaml`).
- Optimized checkpoints are measured by the ordinary evaluation path against the same frozen contract.

## Remote execution (Modal)

`scripts/modal/` holds the remote jobs: `h200_eval.py`, `h200_capability.py`, `gguf_study.py`,
`executorch_export.py`, `upload_artifact.py` and `mediatek_sdk_probe.py`. They spend money and write remote
state, so run them only when the user asks. Preserve outputs before tearing down (`scripts/preserve_h200_state.py`).

## Keeping this skill current

Update this skill in the same commit when a release target, publication record location, conversion flag,
preservation policy version or remote job is added or changed.
