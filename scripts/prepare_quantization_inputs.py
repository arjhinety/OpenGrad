"""Freeze compact study inputs and the prospective quantization gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from opengrad.evaluation.runner import load_evaluation_examples
from opengrad.promotion.quantization import compute_preservation_thresholds


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2"
MODEL_REVISION = "f33d20308982f37deb459076f489e794d5521ee3"
TOKENIZER_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
CHECKPOINT = "dpo-checkpoint-30"
M1_EXPERIMENT = "m1_dpo_canonical_v2_final_v2"
EVAL_MANIFEST = Path("reports/evaluation/behavioral-heldout-v2.manifest.json")
PARTITION = Path("reports/evaluation/behavioral-heldout-v2-partition.json")
REFERENCE_METRICS = Path(
    "runs/m1_dpo_canonical_v2_final_v2/eval/confirmatory/checkpoint-30/metrics.json"
)
M1_PROMOTION = Path("reports/data/m1-dpo-canonical-v2-final-v2-promotion.json")
PREFERENCE = Path("data/processed/m1_calibration_preference_pairs_v1.jsonl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version(name: str) -> str | None:
    try:
        return str(importlib.import_module(name).__version__)
    except (ImportError, AttributeError):
        return None


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".workspace/quantization/source/m1-v2/dpo-checkpoint-30"),
    )
    args = parser.parse_args()
    model_dir = (ROOT / args.model_dir).resolve() if not args.model_dir.is_absolute() else args.model_dir

    manifest = json.loads((ROOT / EVAL_MANIFEST).read_text(encoding="utf-8"))
    partition = json.loads((ROOT / PARTITION).read_text(encoding="utf-8"))
    examples = load_evaluation_examples(ROOT, ROOT / EVAL_MANIFEST)
    dev_ids = set(partition["example_ids"]["dev"])
    confirmatory_ids = set(partition["example_ids"]["confirmatory"])
    if dev_ids & confirmatory_ids or len(dev_ids | confirmatory_ids) != len(examples):
        raise RuntimeError("frozen partition is not disjoint and complete")

    compact = []
    for example in examples:
        split = "dev" if example.example_id in dev_ids else "confirmatory"
        compact.append(
            {
                "example_id": example.example_id,
                "source": example.source,
                "question": example.question,
                "tools": example.tools,
                "expected_decision": example.expected_decision,
                "candidates": example.candidates,
                "metadata": example.metadata,
                "partition": split,
            }
        )

    compact_path = ROOT / "results/quantization/frozen_behavioral_eval_v1.jsonl"
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    with compact_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in compact:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    compact_fingerprint = sha256(compact_path)

    recovery_rows = []
    heldout_ids = dev_ids | confirmatory_ids
    with (ROOT / PREFERENCE).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            identity = str(row.get("canonical_id", ""))
            if identity in heldout_ids or str(row.get("example_id", "")) in heldout_ids:
                raise RuntimeError(f"QAD recovery row overlaps frozen held-out ID: {identity}")
            recovery_rows.append(
                {
                    "canonical_id": identity,
                    "prompt": row["prompt"],
                    "expected_decision": row.get("expected_decision", row.get("chosen_decision")),
                    "source_dataset": row.get("source_dataset"),
                    "pair_origin": row.get("pair_origin"),
                }
            )
    recovery_rows.sort(key=lambda row: hashlib.sha256(row["canonical_id"].encode()).hexdigest())
    validation_cut = max(1, round(len(recovery_rows) * 0.2))
    for index, row in enumerate(recovery_rows):
        row["split"] = "qad_validation" if index < validation_cut else "qad_train"

    recovery_path = ROOT / "manifests/quantization/m1_v2_qad_recovery_v1.jsonl"
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    with recovery_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in recovery_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    reference_artifact = json.loads((ROOT / REFERENCE_METRICS).read_text(encoding="utf-8"))
    routing = dict(reference_artifact["routing"])
    reference = {
        key: float(routing[key])
        for key in (
            "call_f1",
            "call_precision",
            "call_recall",
            "over_call_rate",
            "clarification_accuracy",
            "unsupported_accuracy",
        )
    }
    reference["parse_valid_rate"] = float(reference_artifact["parse_valid_rate"])
    source_hashes = {}
    for name in (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "checkpoint_metadata.json",
    ):
        candidate = model_dir / name
        if candidate.is_file():
            source_hashes[name] = {"sha256": sha256(candidate), "bytes": candidate.stat().st_size}

    write_json(
        ROOT / "results/quantization/m1_v2_reference.json",
        {
            "schema_version": 1,
            "artifact_kind": "BF16_REFERENCE",
            "study": "m1_v2_quantization",
            "experiment_id": M1_EXPERIMENT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "selected_checkpoint": CHECKPOINT,
            "source_model_path": str(model_dir.relative_to(ROOT)) if model_dir.is_relative_to(ROOT) else str(model_dir),
            "tokenizer_revision": TOKENIZER_REVISION,
            "parent_experiment_id": "m0_sft_canonical_v2_final",
            "parent_checkpoint": "m0_sft_canonical_v2_final::checkpoint-1800",
            "parent_weight_sha256": "7144579aeecec8b4de25f193ab63085efdf8d9d76b85ed915352291b0152277a",
            "hf_publication_revision": MODEL_REVISION,
            "evaluation": {
                "manifest": str(EVAL_MANIFEST).replace("\\", "/"),
                "manifest_sha256": sha256(ROOT / EVAL_MANIFEST),
                "compact_eval_sha256": compact_fingerprint,
                "population": len(compact),
                "dev_examples": len(dev_ids),
                "confirmatory_examples": len(confirmatory_ids),
                "dev_fingerprint": partition["dev"]["fingerprint"],
                "confirmatory_fingerprint": partition["confirmatory"]["fingerprint"],
                "renderer": "qwen3_5_2b_v1",
                "template_hash": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
                "evaluator_revision": reference_artifact["lineage"]["evaluator_revision"],
            },
            "metrics": reference,
            "generation": reference_artifact["generation"],
            "seed": 0,
            "runtime": {
                "recorded_backend": reference_artifact["backend"],
                "recorded_engine": reference_artifact["engine"],
                "study_hf_runtime": {
                    "python": platform.python_version(),
                    "torch": version("torch"),
                    "transformers": version("transformers"),
                },
            },
            "source_hashes": source_hashes,
            "promotion_policy": "tool_use_promotion_v4",
            "m1_promotion_artifact": str(M1_PROMOTION).replace("\\", "/"),
            "git_commit": git_sha(),
        },
    )
    write_json(
        ROOT / "results/quantization/quantization_preservation_v1.json",
        {
            "schema_version": 1,
            "policy_version": "quantization_preservation_v1",
            "frozen_before_candidate_evaluation": True,
            "reference": reference,
            "thresholds": compute_preservation_thresholds(reference),
            "requirements": {
                "positive_metrics": "candidate >= 99% of exact BF16 reference",
                "over_call_rate": "candidate <= exact BF16 reference + 0.01 absolute",
                "parse_valid_rate": ">= 0.99",
                "existing_tool_policy": "tool_use_promotion_v4 must remain satisfied",
            },
        },
    )
    write_json(
        ROOT / "manifests/quantization/m1_v2_qad_recovery_v1.json",
        {
            "schema_version": 1,
            "artifact_kind": "QAD_RECOVERY_CORPUS",
            "corpus_id": "m1_v2_qad_recovery_v1",
            "path": str(recovery_path.relative_to(ROOT)).replace("\\", "/"),
            "records": len(recovery_rows),
            "train_records": len(recovery_rows) - validation_cut,
            "validation_records": validation_cut,
            "sha256": sha256(recovery_path),
            "source_sha256": sha256(ROOT / PREFERENCE),
            "heldout_exclusion": {
                "dev_fingerprint": partition["dev"]["fingerprint"],
                "confirmatory_fingerprint": partition["confirmatory"]["fingerprint"],
                "overlap_count": 0,
            },
            "behavior_counts": {
                decision: sum(1 for row in recovery_rows if row["expected_decision"] == decision)
                for decision in sorted({str(row["expected_decision"]) for row in recovery_rows})
            },
            "source_counts": {
                source: sum(1 for row in recovery_rows if row["source_dataset"] == source)
                for source in sorted({str(row["source_dataset"]) for row in recovery_rows})
            },
        },
    )
    print(json.dumps({"eval_records": len(compact), "recovery_records": len(recovery_rows), "reference": reference}, indent=2))


if __name__ == "__main__":
    main()

