#!/usr/bin/env python3
"""Drive the full GGUF PTQ ladder: quantize, generate, bench, fetch, score — per rung.

Each rung is carried all the way to a scored verdict before the next begins, so a failure leaves a
complete record for every rung that finished rather than nine half-finished ones. Every step's
result is written under `results/quantization/gguf/`, and a rung that fails is recorded with its
error instead of being skipped silently — a ladder with a hole in it cannot answer "where is the
floor", which is the question the sub-Q4 rungs exist to answer.

The scoring step compares each rung to the **llama.cpp BF16 GGUF**, because that holds engine and
tokenizer constant and isolates quantization. The frozen `quantization_preservation_v1` gate is
still applied against the vLLM BF16 reference, unchanged.

Usage:
    python scripts/run_gguf_ladder.py                      # every rung
    python scripts/run_gguf_ladder.py --only Q4_K_M Q8_0   # a subset
    python scripts/run_gguf_ladder.py --skip-existing      # resume
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/quantization/gguf"
STUDY = "scripts/modal/gguf_study.py"
VOLUME = "opengrad-quant"

# The Modal CLI is installed as its own executable (uv tool), not as an importable module in this
# interpreter — `python -m modal` fails with ModuleNotFoundError. Resolve the real binary instead
# of assuming either form works.
MODAL = shutil.which("modal") or shutil.which(
    "modal", path=str(Path.home() / ".local/bin")
)
if MODAL is None:
    raise SystemExit("modal CLI not found on PATH or in ~/.local/bin")

# Ascending bits per weight, matching LADDER in the Modal script.
LADDER = ("Q2_K", "Q3_K_M", "IQ3_M", "IQ4_XS", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")

ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}


def run(command: list[str], *, label: str) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(command)}", flush=True)
    result = subprocess.run(command, env=ENV, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stdout[-4000:], flush=True)
        print(result.stderr[-4000:], flush=True)
        raise RuntimeError(f"{label} failed ({result.returncode})")
    return result


def modal_stage(stage: str, **kwargs) -> None:
    command = [MODAL, "run", STUDY, "--stage", stage]
    for key, value in kwargs.items():
        command += [f"--{key.replace('_', '-')}", str(value)]
    run(command, label=f"modal stage {stage}")


def fetch_generations(stem: str, partition: str) -> Path:
    """Copy the generations off the Modal volume so scoring happens against the tested module."""
    local_dir = ROOT / ".workspace/quantization/generations"
    local_dir.mkdir(parents=True, exist_ok=True)
    name = f"{stem}.{partition}.jsonl"
    target = local_dir / name
    if target.exists():
        target.unlink()
    run(
        # No leading slash: `modal volume get` resolves "generations/x" but reports
        # "No such file or directory" for "/generations/x", even though `ls` accepts both forms
        # elsewhere.
        [MODAL, "volume", "get", "--force", VOLUME, f"generations/{name}", str(target)],
        label=f"volume get {name}",
    )
    return target


def score(stem: str, partition: str, quantization: str | None, quantize_record: dict | None) -> None:
    command = [
        sys.executable, "scripts/score_gguf_candidate.py",
        "--artifact", stem,
        "--generations", str(fetch_generations(stem, partition)),
        "--partition", partition,
    ]
    if quantization:
        command += ["--quantization", quantization]
    if quantize_record:
        command += [
            "--artifact-bytes", str(quantize_record["artifact_bytes"]),
            "--artifact-sha256", quantize_record["artifact_sha256"],
        ]
    result = run(command, label=f"score {stem}")
    print(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--partition", default="confirmatory")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rungs = list(args.only) if args.only else list(LADDER)
    unknown = [r for r in rungs if r not in LADDER]
    if unknown:
        raise SystemExit(f"unknown rung(s): {unknown}; valid: {list(LADDER)}")

    baseline_score = OUT / f"score_m1-v2-bf16_{args.partition}.json"
    if not baseline_score.is_file():
        raise SystemExit(
            f"missing {baseline_score.relative_to(ROOT)} — the llama.cpp BF16 baseline must be "
            "scored before any rung, since it is the quantization-attribution reference"
        )

    outcomes = []
    for rung in rungs:
        stem = f"m1-v2-{rung}"
        score_path = OUT / f"score_{stem}_{args.partition}.json"
        if args.skip_existing and score_path.is_file():
            print(f"\n=== {rung}: already scored, skipping ===")
            outcomes.append({"rung": rung, "status": "SKIPPED_EXISTING"})
            continue

        print(f"\n{'=' * 70}\n=== {rung}\n{'=' * 70}")
        try:
            quantize_path = OUT / f"quantize_{rung}.json"
            if not (args.skip_existing and quantize_path.is_file()):
                modal_stage("quantize", quant_type=rung)
            record = json.loads(quantize_path.read_text(encoding="utf-8"))

            modal_stage("generate", artifact=f"{stem}.gguf", partition=args.partition)
            modal_stage("bench", artifact=f"{stem}.gguf")
            score(stem, args.partition, rung, record)

            verdict = json.loads(score_path.read_text(encoding="utf-8"))
            outcomes.append({
                "rung": rung,
                "status": verdict["gate_vs_frozen_vllm_reference"]["decision"],
                "bytes": record["artifact_bytes"],
            })
        except Exception as exc:
            # Recorded, not swallowed: a rung that failed is evidence about where the floor is.
            print(f"!!! {rung} FAILED: {type(exc).__name__}: {exc}", flush=True)
            outcomes.append({"rung": rung, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})

    summary = OUT / f"ladder_summary_{args.partition}.json"
    summary.write_text(json.dumps(outcomes, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {summary.relative_to(ROOT)}")
    for item in outcomes:
        size = f"{item['bytes'] / 1024**3:.2f} GiB" if item.get("bytes") else ""
        print(f"  {item['rung']:<8} {item['status']:<22} {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
