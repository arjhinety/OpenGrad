#!/usr/bin/env python3
"""Pull one stage's generations off the Modal volume and score every benchmark present. CPU.

Scoring is separated from generation on purpose: generations are expensive and immutable, scoring
is cheap and may be re-run when a scorer is corrected. Nothing here re-runs the GPU.

Usage:
    python scripts/fetch_and_score_capability.py --stage M1_DPO_CURRENT
    python scripts/fetch_and_score_capability.py --stage M0_SFT --skip-fetch
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "results/benchmarks/h200/capability_v1"
VOLUME = "opengrad-quant"

MODAL = shutil.which("modal") or shutil.which(
    str(Path.home() / "AppData/Roaming/uv/tools/modal/Scripts/modal.exe")
)

# Forcing UTF-8 is not cosmetic: the Modal CLI prints U+2713 in its progress output and dies with a
# charmap UnicodeEncodeError on a default Windows console, taking the whole invocation with it.
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

SCORERS = {
    "ifeval": ("scripts/score_ifeval.py", "ifeval_scores.json"),
    "gsm8k": ("scripts/score_gsm8k.py", "gsm8k_scores.json"),
    "mmlu_pro": ("scripts/score_mmlu_pro.py", "mmlu_pro_scores.json"),
    "sentinel": ("scripts/score_sentinel.py", "sentinel_scores.json"),
}


def run(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    print(f"\n$ {label}", flush=True)
    result = subprocess.run(cmd, env=ENV, text=True, capture_output=True,
                            encoding="utf-8", errors="replace", check=False)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        print(result.stderr.rstrip(), file=sys.stderr)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--skip-fetch", action="store_true")
    args = ap.parse_args()

    target = CAP / args.stage
    target.mkdir(parents=True, exist_ok=True)

    if not args.skip_fetch:
        if MODAL is None:
            print("modal CLI not found on PATH", file=sys.stderr)
            return 1
        # No leading slash: `modal volume get` resolves "h200/capability/<stage>" but rejects the
        # same path written with one.
        result = run(
            [MODAL, "volume", "get", "--force", VOLUME,
             f"h200/capability/{args.stage}", str(target.parent)],
            f"volume get {args.stage}",
        )
        if result.returncode != 0:
            print(f"fetch failed for {args.stage}", file=sys.stderr)
            return 1

    statuses = {}
    for job, (script, out_name) in SCORERS.items():
        gens = target / f"generations_{job}.jsonl"
        if not gens.exists():
            statuses[job] = "ABSENT"
            continue
        result = run(
            [sys.executable, str(ROOT / script), str(gens),
             "--stage", args.stage, "--out", str(target / out_name)],
            f"score {job}",
        )
        statuses[job] = "SCORED" if result.returncode == 0 else "SCORER_FAILED"

    print(f"\n{args.stage}:")
    for job, st in statuses.items():
        print(f"  {job:<10} {st}")
    return 0 if all(s in ("SCORED", "ABSENT") for s in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
