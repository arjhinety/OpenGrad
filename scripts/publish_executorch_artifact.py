#!/usr/bin/env python3
"""Publish one exported ExecuTorch artifact and add it to the QwenGrad-Executorch collection.

What this publishes is an *export*, not a result. The preservation gate needs behavioural evidence
and this study does not produce it for ExecuTorch, so every repo created here is labelled
`EXPORTED_PENDING_EVALUATION` and the README says plainly that a successful export is not evidence
of preserved behaviour. The card carries the frozen thresholds so whoever runs it on OpenWeights
compares against the same numbers rather than inventing their own.

The upload is the handoff package built by `scripts/build_executorch_handoff.py`: the `.pte`, the
tokenizer, the pre-tokenized frozen prompts, the vendored scorer, and the parity notes. Shipping
the prompts pre-tokenized is what removes the BOS ambiguity — the stock ExecuTorch metadata
declares `get_bos_id: 248045`, which is the `<|im_start|>` every prompt already begins with.

Usage:
    python scripts/publish_executorch_artifact.py --target cpu \\
        --repo experimentalmachines/QwenGrad-Qwen3.5-2B-M1-DPO-v2-ExecuTorch-CPU
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

COLLECTION_SLUG = "experimentalmachines/qwengrad-executorch-6aa5008c9414ee393594e3a8"
PARENT_MODEL = "arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2"
PARENT_REVISION = "f33d20308982f37deb459076f489e794d5521ee3"

TARGET_LABELS = {
    "cpu": "XNNPACK CPU",
    "qnn": "Qualcomm QNN / Hexagon HTP",
    "mediatek": "MediaTek NeuroPilot",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_card(target: str, manifest: dict, gate: dict, reference: dict) -> str:
    label = TARGET_LABELS.get(target, target)
    artifacts = manifest.get("artifacts") or []
    contract = manifest["measurement_contract"]
    status = manifest["status"]
    thresholds = gate["thresholds"]

    if artifacts:
        rows = "\n".join(
            f"| `{item['file']}` | {item.get('quantization') or 'fp32'} | "
            f"{item['bytes'] / (1024 ** 3):.2f} GiB | `{item['sha256'][:16]}…` |"
            for item in sorted(artifacts, key=lambda i: i["bytes"])
        )
        artifact_rows = (
            "| file | precision | size | sha256 |\n|---|---|---:|---|\n" + rows + "\n"
        )
    else:
        artifact_rows = "No `.pte` was produced — see the export record below.\n"

    return f"""---
license: apache-2.0
base_model: {PARENT_MODEL}
tags:
  - executorch
  - tool-calling
  - qwen3.5
  - quantization
  - opengrad
---

# QwenGrad Qwen3.5-2B M1-DPO-v2 — ExecuTorch ({label})

**Status: `{status}`** — this is an exported runtime artifact with **no behavioural verdict**.
A successful export is not evidence of preserved behaviour. Until it is scored against
`quantization_preservation_v1` on the frozen confirmatory partition, nothing here claims the
parent model's tool-calling policy survived.

## Lineage

| | |
|---|---|
| parent model | [`{PARENT_MODEL}`](https://huggingface.co/{PARENT_MODEL}) |
| parent revision | `{PARENT_REVISION}` |
| parent checkpoint | `{manifest['parent_checkpoint']}` |
| target | {label} |
| relationship | deployment adaptation (not a new capability stage) |

## Artifacts

{artifact_rows}
These were exported through a **custom OpenGrad path**, not a stock one-command ExecuTorch export:
the context window is raised from 2048 to 5760 to fit the frozen evaluation, and the Snapdragon
target additionally needs two source patches to ExecuTorch. Full details and the patch files are in
[`integrations/executorch/`](https://github.com/arjhinety/OpenGrad/tree/master/integrations/executorch).

Note on `8da4w`: upstream's `examples/models/qwen3_5/README.md` states that quantization is
*"intentionally deferred to a follow-up"*. It was attempted anyway and exported cleanly, 3.09×
smaller than fp32. The documentation is stale, not the capability.


## Measurement contract

The prompts in this repo were rendered **once** by OpenGrad's pinned renderer and must be used
verbatim. Re-applying a chat template produces a different measurement.

| | |
|---|---|
| renderer | `{contract['renderer']}` |
| tokenizer revision | `{contract['tokenizer_revision']}` |
| template hash | `{contract['template_hash']}` |
| partition | {contract['partition']} ({contract['examples']:,} examples) |
| decoding | greedy (`temperature 0.0`, `top_p 1.0`, `do_sample false`) |
| completion budget | {contract['generation']['max_new_tokens']} tokens |
| context window needed | {contract['engine_window']} |
| `add_bos_token` | **{contract['add_bos_token']}** |

### The BOS trap

The pinned tokenizer sets `add_bos_token=False` and `bos_token=None`. The stock ExecuTorch config
nevertheless declares `get_bos_id: {contract['bos_token_id_declared_by_pte_metadata']}` — which is
`<|im_start|>`, the token every rendered prompt **already starts with**. A runner that honours that
metadata emits `<|im_start|><|im_start|>system…` and silently measures a different model.

`{manifest['files']['prompts']}` therefore ships `token_ids` alongside the text. Feeding ids is the
safe path. Read `PARITY_NOTES.md` before running anything.

## The gate this must pass

Computed from the parent's exact BF16 confirmatory metrics **before** any candidate existed:

| metric | BF16 reference | required |
|---|---:|---:|
| `call_f1` | {reference['call_f1']:.6f} | ≥ {thresholds['call_f1']:.6f} |
| `call_precision` | {reference['call_precision']:.6f} | ≥ {thresholds['call_precision']:.6f} |
| `call_recall` | {reference['call_recall']:.6f} | ≥ {thresholds['call_recall']:.6f} |
| `clarification_accuracy` | {reference['clarification_accuracy']:.6f} | ≥ {thresholds['clarification_accuracy']:.6f} |
| `unsupported_accuracy` | {reference['unsupported_accuracy']:.6f} | ≥ {thresholds['unsupported_accuracy']:.6f} |
| `over_call_rate` | {reference['over_call_rate']:.6f} | ≤ {thresholds['over_call_rate_max']:.6f} |
| `parse_valid_rate` | {reference['parse_valid_rate']:.6f} | ≥ {thresholds['parse_valid_rate_min']:.2f} |

The first five are relative floors at 99% of the reference. `over_call_rate` uses an absolute
tolerance because it is an error rate near 0.15, where a relative floor turns a handful of examples
into an apparent collapse.

## How to score a run

```bash
python score_generations.py --generations my_run.jsonl --artifact {target}
```

`my_run.jsonl` is one object per example: `{{"example_id": "...", "raw": "...", "truncated": false}}`.
The scorer **refuses** a run whose generations do not reconcile exactly with the submitted prompts —
missing, duplicate, unknown, or errored rows are errors, not omissions. A shrinking denominator
would report a better score for having answered less.

It vendors OpenGrad's real parser (`opengrad_min/parser.py`, byte-identical to the source), which is
byte-compatible with OpenWeights' `ToolCallParser.parseTaggedXml`. Do not score with a runtime's own
tool extraction: llama.cpp's built-in parser does not recognise Qwen3.5's XML emission.

## Known limitations

- The Qwen3.5 ExecuTorch path is **fp32 + static shape** (`enable_dynamic_shape=False`), and
  `runner.native` falls back to sequential prefill for multi-token prompts. That is a throughput
  property, not a correctness one, but a full {contract['examples']:,}-example run is slow on one
  stream — shard it.
- The confirmatory partition is **internal**, not an untouched external benchmark.
- The evaluator does not measure tool-selection accuracy, argument validity, or schema validity.
  Their absence from the table above is not a zero.
- The frozen population contains **no ANSWER examples**, so `no_call_accuracy` is structurally 0.0
  and the ANSWER row of any confusion matrix is empty. That is a property of the data.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=["cpu", "qnn", "mediatek"])
    parser.add_argument("--repo", required=True, help="org/name for the model repo")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    package = ROOT / "release/executorch" / args.target
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"no handoff package at {package.relative_to(ROOT)}; "
            "run scripts/build_executorch_handoff.py first"
        )
    manifest = read_json(manifest_path)
    gate = read_json(ROOT / "results/quantization/quantization_preservation_v1.json")
    reference = {
        key: float(value)
        for key, value in read_json(ROOT / "results/quantization/m1_v2_reference.json")[
            "metrics"
        ].items()
    }

    card = build_card(args.target, manifest, gate, reference)
    (package / "README.md").write_text(card, encoding="utf-8")
    print(f"wrote {(package / 'README.md').relative_to(ROOT)}")

    if args.dry_run:
        print("--dry-run: not uploading")
        print(card[:1500])
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        folder_path=str(package),
        repo_type="model",
        commit_message=(
            f"ExecuTorch {TARGET_LABELS[args.target]} export of "
            f"{manifest['parent_checkpoint']} ({manifest['status']})"
        ),
    )
    print(f"uploaded -> https://huggingface.co/{args.repo}")

    try:
        api.add_collection_item(
            collection_slug=COLLECTION_SLUG,
            item_id=args.repo,
            item_type="model",
            note=f"{TARGET_LABELS[args.target]} — {manifest['status']}",
            exists_ok=True,
        )
        print(f"added to collection {COLLECTION_SLUG}")
    except Exception as exc:  # noqa: BLE001 - publishing succeeded; collection membership is recoverable
        print(f"WARNING: could not add to collection: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
