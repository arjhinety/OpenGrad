# Reproducing the M1-v2 quantization work

Everything below regenerates from pinned inputs. Where a step needs a GPU it names the one used.
**Nothing here should be re-run to "clean up" hashes** — the recorded hashes are the evidence.

## Pins

| | |
|---|---|
| model repo | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2` |
| revision | `f33d20308982f37deb459076f489e794d5521ee3` |
| checkpoint | `dpo-checkpoint-30` |
| weights sha256 | `903f9b115d418d1988c0d5e1dd4bda7e04809a54a6bc7e34431982fbee2262f6` |
| `config.json` sha256 | `88bf86c270d616198909ed1eefef8d8c21ac1fa13f62e947f20f8e1ebd02c211` |
| chat template hash | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| llama.cpp | `b10919` / `d3146f2b56c2db4711ac8391871c9e529d1946d7` |
| BF16 GGUF sha256 | `3cf73dc7f4deb303593ec01b8db4603356e35f1ab3e44edd9444496a04cc83de` |
| imatrix sha256 | `ab6dc65f8a5c02bf036d4233c199d4852fdad488ec018d9adc9d20033a09dbc1` |
| calibration corpus sha256 | `2b67da2b7046749ade395fc7d9587d7df9694e3b4d3cefeee5457aa75de4b400` |
| ExecuTorch commit | `df6147afadf106a0ef3a74f65d80b5c589e63e44` |
| GPU | A100-80GB (Modal) |

The `config.json` hash is enforced, not just recorded: `prepare` re-fetches the file if it does not
match, because an edited config produces a different model from identical weights. An earlier
revision of the study script mutated that file on the cached volume, which is exactly what the
guard now prevents.

## Environment

```bash
pip install modal
modal token new                       # once
export PYTHONIOENCODING=utf-8         # Windows: the Modal CLI cannot encode its own output otherwise
export PYTHONUTF8=1
export PYTHONPATH=src
```

The Modal CLI is a standalone executable, **not** importable — `python -m modal` fails.
`scripts/run_gguf_ladder.py` resolves the binary via `shutil.which`.

## GGUF branch (complete)

```bash
# 1. convert, verify block_count/nextn off the written artifact
modal run scripts/modal/gguf_study.py --stage prepare

# 2. tokenizer identity, and the parity gate (expect FAILURE: 1271/1277 — this is recorded, not fixed)
modal run scripts/modal/gguf_study.py --stage inspect-tokenizer
modal run scripts/modal/gguf_study.py --stage parity

# 3. isolate the divergence: engine held constant, only the tokenizer path varies
modal run scripts/modal/gguf_study.py --stage divergence-probe

# 4. importance matrix from the training-side calibration corpus
modal run scripts/modal/gguf_study.py --stage imatrix

# 5. quantize all nine rungs from the BF16 GGUF (never from a quantized source)
modal run scripts/modal/gguf_study.py --stage ladder

# 6. BF16 baseline: generate, fetch, score  (must precede every rung)
modal run scripts/modal/gguf_study.py --stage generate --artifact m1-v2-bf16.gguf --partition confirmatory
modal run scripts/modal/gguf_study.py --stage bench --artifact m1-v2-bf16.gguf
modal volume get --force opengrad-quant generations/m1-v2-bf16.confirmatory.jsonl \
  .workspace/quantization/generations/m1-v2-bf16.confirmatory.jsonl
python scripts/score_gguf_candidate.py --artifact m1-v2-bf16 \
  --generations .workspace/quantization/generations/m1-v2-bf16.confirmatory.jsonl

# 7. every rung: generate -> bench -> fetch -> score
python scripts/run_gguf_ladder.py --skip-existing

# 8. reports, figures, ledger, closure
python scripts/build_engine_parity_report.py
python scripts/build_ptq_evaluation_report.py
python scripts/build_quantization_visual.py
python scripts/update_quantization_ledger.py
python scripts/build_tokenizer_regression_fixtures.py
python scripts/close_ptq_phase.py
```

`modal volume get` needs the path **without** a leading slash (`generations/x`, not
`/generations/x`) even though `modal volume ls` accepts both.

### Verify a closure without rewriting it

```bash
python scripts/close_ptq_phase.py --verify     # fails if the manifest drifted from live evidence
```

## ExecuTorch branch

```bash
modal run scripts/modal/executorch_export.py --target cpu
modal run scripts/modal/executorch_export.py --target cpu --quantize 8da4w
modal run scripts/modal/executorch_export.py --target audit --artifact qwen3_5_2b_8da4w.pte
python scripts/audit_executorch_quantization.py          # must print reconciled: True
modal run scripts/modal/executorch_export.py --target qnn          # REJECTED_EXPORT, recorded
modal run scripts/modal/executorch_export.py --target mediatek-probe
```

## MediaTek

The NeuroPilot Express SDK is **proprietary and licensed**. It is not in this repository and must
not be committed, uploaded, or vendored. A user with legitimate access installs it locally:

```bash
tar -xzf <neuropilot-express-sdk>.tar.gz -C /opt/neuropilot
python3.10 -m pip install /opt/neuropilot/*/mtk_converter-*/mtk_converter-*.whl
python3.10 -m pip install /opt/neuropilot/*/mtk_neuron-*.whl
cp /opt/neuropilot/*/api/NeuronAdapter.h "$EXECUTORCH_ROOT/backends/mediatek/runtime/include/api/"
export EXECUTORCH_ROOT=/path/to/executorch ANDROID_NDK=/path/to/ndk-r26.3.11579264
./integrations/executorch/mediatek/export_qwen3_5.sh <checkpoint-dir>
```

Python **3.10** is required — the `mtk_converter` wheel is cp310. The export currently stops at the
unimplemented Gated DeltaNet layer by design; see [`MEDIATEK_PORT_STATUS.md`](MEDIATEK_PORT_STATUS.md).

## Tests

```bash
PYTHONPATH=src python -m pytest tests/evaluation -q
```

Two failures in `tests/evaluation/test_baseline_runner.py` are **pre-existing** Windows
path-separator issues in the frozen measurement path (`runner.py` compares
`str(path.relative_to(root))` against forward-slash literals). They predate this work, both files
are untouched by it, and they must not be attributed to the quantization phase.

## Known-good outputs

| check | expected |
|---|---|
| `close_ptq_phase.py` | `selection.recommended_release_rung: Q6_K` (Q6_K and Q8_0 pass the gate; no rung meets the secondary release bar, so the smallest passing rung is the fallback), 7 rungs `REJECTED_ACCURACY` |
| `audit_executorch_quantization.py` | `reconciled: True`, 99.98% of named-data weights int4 |
| tokenizer parity | 1271/1277 exact, 0 BOS insertions — **a FAIL, and expected** |
| divergence probe | 6 fixtures, 3 behaviour-changing |
| BF16 generate | 1277/1277, ~213s on A100 with 8 slots |

The committed closure manifest also carries per-rung role labels (`Q8_0 → RECOMMENDED_RELEASE`,
`Q6_K → MEMORY_OPTIMIZED_RELEASE`). Those labels were hard-coded in the script after the ladder was
scored and contradict the pre-registered rule (`release_selection_criteria_v1.json`: smallest
passing format wins). The recommended release is **Q6_K (1.45 GiB)**; Q8_0 (1.87 GiB) also passes.
Both pass inside run-to-run noise: the H200 vLLM BF16 rerun of the unquantized reference itself
fails the gate (recall 0.7638 < 0.7671 floor), and Q6_K clears precision by one example (357/490
against 356.96 required). The "99.98% int4" figure counts named-data weights only; overall int4
coverage is 78.7%, and the fp32 embedding is 65.7% of the `.pte`
(`results/quantization/executorch/quantization_audit_8da4w.json`).

If the BF16 generate takes tens of minutes with no progress output, the llama-server stdout pipe
has been reattached to an undrained `subprocess.PIPE`; see
`tests/evaluation/test_subprocess_pipe_invariants.py`.
