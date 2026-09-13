# OpenGrad M1-v2 Quantization Study — Handoff

**Status:** STOPPED BY USER before PTQ, runtime parity, QAD, export, or publication.

> **Superseded (2026-09-13).** This records the state at the interruption. The GGUF PTQ phase was
> resumed and is now `CLOSED`: all nine rungs were scored against the gate below; Q6_K and Q8_0
> pass, and the recommended release under the pre-registered smallest-passing rule is Q6_K
> (`manifests/quantization/ptq_phase_closure_v1.json`, `reports/PTQ_PHASE_CLOSURE.md`). QAD was
> not run (`reports/QAD_DECISION.md`).

**Study objective:** produce GGUF/llama.cpp and ExecuTorch/TorchAO descendants of the promoted
M1-v2 BF16 model, accepting only artifacts that pass a frozen behavioral-preservation gate.

## Authoritative reference reconstructed

| Field | Frozen value |
|---|---|
| M1 experiment | `m1_dpo_canonical_v2_final_v2` |
| Selected checkpoint | `dpo-checkpoint-30` |
| Published model | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2` |
| HF revision | `f33d20308982f37deb459076f489e794d5521ee3` |
| Base model revision | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Tokenizer revision | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Parent checkpoint | `m0_sft_canonical_v2_final::checkpoint-1800` |
| Parent weight SHA-256 | `7144579aeecec8b4de25f193ab63085efdf8d9d76b85ed915352291b0152277a` |
| Dataset revision | When2Call `0582f7749df63a96fdc3070932e83e72396ace53` |
| Frozen eval manifest SHA-256 | `8bb6ad2e37613c996476bb734b93ef792eae807ec2cccf6b200d96a47332c640` |
| DEV partition | 2,373 examples; `88a56821edfc8614285846e8bef60cf8aced124ce3b5a2d4054e8344618c9980` |
| Confirmatory partition | 1,277 examples; `d6d1e394a89b5ec8b9ed41ef233d752ff50f69abe41a6e06e34878c1088f32ba` |
| Renderer | `qwen3_5_2b_v1` |
| Template hash | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| Evaluator revision | `2d97c7d5a8de0b16a2e58e4376e231fe06ab16dc` |
| Existing promotion policy | `tool_use_promotion_v4` |

Exact frozen confirmatory metrics from
`runs/m1_dpo_canonical_v2_final_v2/eval/confirmatory/checkpoint-30/metrics.json`:

| Metric | BF16 M1-v2 |
|---|---:|
| `call_f1` | 0.7548387096774194 |
| `call_precision` | 0.7358490566037735 |
| `call_recall` | 0.7748344370860927 |
| `over_call_rate` | 0.1529126213592233 |
| `clarification_accuracy` | 0.7654986522911051 |
| `unsupported_accuracy` | 0.5386313465783664 |
| `parse_valid_rate` | 1.0 |

## Preservation gate already specified

The new policy implementation is `src/opengrad/promotion/quantization.py` and is intentionally
separate from the historical M1 policy. Policy name: `quantization_preservation_v1`.

Thresholds computed from the exact values above, before candidate evaluation:

```json
{
  "call_f1": 0.7472903225806452,
  "call_precision": 0.7284905660377358,
  "call_recall": 0.7670860927152318,
  "clarification_accuracy": 0.7578436657681941,
  "unsupported_accuracy": 0.5332450331125828,
  "over_call_rate_max": 0.1629126213592233,
  "parse_valid_rate_min": 0.99
}
```

The final artifact also needs the existing M1 tool-use promotion verdict to remain `PROMOTE`.
No candidate had been evaluated against this gate at handoff (nine have since; see the note above).

## Completed during the interrupted turn

- Verified the repository is on `master` at commit `17530e5`; pre-existing user changes were
  preserved.
- Verified Modal CLI authentication and the `arjhibe` workspace.
- Previously confirmed Modal can allocate an **NVIDIA A100 80GB PCIe**.
- Downloaded the immutable selected checkpoint into:
  `.workspace/quantization/source/m1-v2/dpo-checkpoint-30/`
- Verified the downloaded `model.safetensors` is 3,763,692,048 bytes with SHA-256:
  `903f9b115d418d1988c0d5e1dd4bda7e04809a54a6bc7e34431982fbee2262f6`.
- Confirmed the model config is `Qwen3_5ForCausalLM` with 24 layers, mixed linear/full attention,
  and tied embeddings.
- Added deterministic preservation-policy code and its tests:
  - `src/opengrad/promotion/quantization.py`
  - `tests/evaluation/test_quantization_policy.py`
- Added, but did not execute, the input-freezing script:
  - `scripts/prepare_quantization_inputs.py`

The model download is complete and no study GPU job is running. No GGUF, PTE, QAD checkpoint,
candidate metric, or publication artifact was produced.

## Not yet done

1. Run `scripts/prepare_quantization_inputs.py` to create:
   - `results/quantization/m1_v2_reference.json`
   - `results/quantization/quantization_preservation_v1.json`
   - `results/quantization/frozen_behavioral_eval_v1.jsonl`
   - `manifests/quantization/m1_v2_qad_recovery_v1.jsonl`
   - `manifests/quantization/m1_v2_qad_recovery_v1.json`
2. Add and freeze `reports/QUANTIZATION_STUDY_PLAN.md` and the quantization configs.
3. Run the new policy tests.
4. Establish HF-BF16, GGUF-BF16/F16, and ExecuTorch high-precision runtime parity.
5. Run GGUF PTQ in the order `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, using a training-side imatrix
   only. The smallest passing format wins.
6. Run ExecuTorch/TorchAO PTQ for the supported 8da4w configuration and inspect quantized and
   skipped layers.
7. Only if a branch fails PTQ preservation, run target-aware QAD with a frozen BF16 teacher and
   deployment-matching fake quantization. Do not train QAD pre-emptively.
8. Evaluate the actual GGUF and `.pte` artifacts on the frozen confirmatory IDs and benchmark
   size, memory, load time, TTFT, prefill, and decode throughput.
9. Record rejected candidates, lineage, tests, reports, and publication decisions.

## Resume commands

From the repository root:

```powershell
$env:PYTHONPATH = "src"
py scripts/prepare_quantization_inputs.py `
  --model-dir .workspace/quantization/source/m1-v2/dpo-checkpoint-30
py -m pytest -q tests/evaluation/test_quantization_policy.py
```

The local environment has Torch `2.10.0+cu126` and Transformers `5.14.1`, but no local
`vllm`, `torchao`, `executorch`, or llama.cpp binaries. Use Modal for the A100 work and record
the exact versions in every runtime artifact.

## Working-tree notes

At handoff, pre-existing user changes remain untouched:

- modified `release/huggingface/qwen35-2b-m1-dpo-canonicalv2-final-v2/README.md`
- untracked `release/huggingface/qwen35-2b-m1-dpo-canonicalv2-final-v2/qwengrad-dpo-results.png`
- existing untracked `.workspace/` content

The quantization source checkpoint and the three new study files are also untracked. Do not
delete or overwrite the promoted HF source; treat the downloaded checkpoint as an immutable input.
