# Model renderer matrix

| Exact checkpoint | Revision pinned | Template inspected | Renderer implemented | Snapshot tested | Training validated |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-2B` | YES: `15852e8c16360a2fea060d615a32b45270f8a8fc` | MEASURED: pinned tokenizer hash `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` | IMPLEMENTED | COMPLETE: eight golden fixtures | USED IN TRAINING: the M0 SFT runs (corpus v1, corpus v2, Canonical-v2 final, both minus-xLAM arms) and the DPO runs rendered through it; see each run's `rendering_report.json` |

Only registered exact checkpoints receive renderers. The canonical layer does not contain Qwen syntax. Rendering is lazy and uses tokenizer `apply_chat_template`; neural network weights are not loaded. The pre-GPU SFT render audit over the normalization-v1 sources produced 210,874 candidates (recorded only in the local, uncommitted `reports/artifacts/clean-sft-candidate-index.manifest.json`); 3,077 canonical-valid LoopTool records remain explicit exclusions because the pinned template requires a user query. The committed render measurement for the corpus M0 trained on is `runs/m0_sft_canonical_v2_final/rendering_report.json`: 173,237 records considered, 161,966 trainable.
