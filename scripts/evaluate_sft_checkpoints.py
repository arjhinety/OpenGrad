#!/usr/bin/env python3
"""Evaluate every checkpoint of an SFT run against the frozen held-out set.

Produces the training curve that a stopping decision needs: one measurement per checkpoint,
taken with the same engine, renderer, template, parser, generation settings, and manifest that
produced B0, so each point is comparable to the baseline and to the other points.

Usage:
    python scripts/evaluate_sft_checkpoints.py <experiment_id> [--steps 400,800,...] [--limit N]

Writes one candidate config per checkpoint under
``configs/evaluation/candidates/<experiment_id>/`` and the measurements under
``runs/<experiment_id>/eval/<checkpoint>/``. It never writes to the baseline evidence paths.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.evaluation.candidate import (
    run_candidate_evaluation,
    write_candidate_config,
)


def discover_checkpoints(experiment_id: str) -> list[tuple[int, Path]]:
    ckpt_dir = ROOT / "runs" / experiment_id / "checkpoints"
    found: list[tuple[int, Path]] = []
    # DPO names its checkpoints `dpo-checkpoint-N`, so match on the trailing step rather than on
    # a fixed prefix.
    for path in sorted(ckpt_dir.glob("*checkpoint-*")):
        match = re.search(r"checkpoint-(\d+)$", path.name)
        if match and (path / "config.json").is_file():
            found.append((int(match.group(1)), path))
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("--steps", default="", help="comma-separated checkpoint steps to evaluate")
    parser.add_argument(
        "--limit", type=int, default=None, help="evaluate only the first N examples"
    )
    parser.add_argument("--out", default=None, help="where to write the curve JSON")
    args = parser.parse_args()

    available = discover_checkpoints(args.experiment_id)
    if not available:
        print(f"no checkpoints under runs/{args.experiment_id}/checkpoints", file=sys.stderr)
        return 1
    if args.steps:
        wanted = {int(value) for value in args.steps.split(",") if value.strip()}
        selected = [(step, path) for step, path in available if step in wanted]
        missing = sorted(wanted - {step for step, _ in selected})
        if missing:
            print(f"requested checkpoints not found: {missing}", file=sys.stderr)
            return 1
    else:
        selected = available

    config_dir = ROOT / "configs" / "evaluation" / "candidates" / args.experiment_id
    curve: list[dict] = []
    for step, checkpoint in selected:
        config_path = config_dir / f"checkpoint-{step}.yaml"
        write_candidate_config(
            ROOT,
            checkpoint=checkpoint,
            checkpoint_id=f"checkpoint-{step}",
            checkpoint_step=step,
            parent_experiment_id=args.experiment_id,
            out_path=config_path,
        )
        print(f"--- evaluating checkpoint-{step} ({checkpoint})", flush=True)
        started = time.monotonic()
        result = run_candidate_evaluation(config_path, root=ROOT, limit=args.limit)
        result["evaluation_seconds"] = round(time.monotonic() - started, 1)
        curve.append(result)
        comparison = result.get("baseline_comparison", {}).get("metrics", {})
        summary = ", ".join(
            f"{key}={value['candidate']:.4f}({value['verdict'][:4]})"
            for key, value in comparison.items()
            if key in {"call_f1", "call_precision", "over_call_rate", "parse_valid_rate"}
        )
        print(f"    {summary}", flush=True)

    out_path = (
        Path(args.out) if args.out else ROOT / "runs" / args.experiment_id / "eval" / "curve.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge with any previously measured points. Evaluating a subset must not silently discard the
    # rest of the curve: checking one checkpoint at a time would otherwise leave a file that looks
    # like a one-point history of the run.
    existing: dict[int, dict] = {}
    if out_path.is_file():
        try:
            previous = json.loads(out_path.read_text(encoding="utf-8"))
            for previous_point in previous.get("points", []):
                existing[int(previous_point["checkpoint_step"])] = previous_point
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            existing = {}
    for point in curve:
        existing[int(point["lineage"]["checkpoint_step"])] = {
            "checkpoint_step": point["lineage"]["checkpoint_step"],
            "checkpoint_id": point["lineage"]["checkpoint_id"],
            "records": point["records"],
            "parse_valid_rate": point["parse_valid_rate"],
            "routing": point["routing"],
            "baseline_comparison": point.get("baseline_comparison"),
            "evaluation_seconds": point["evaluation_seconds"],
        }
    out_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": args.experiment_id,
                "baseline_run_id": "tool_calling/qwen35_2b/baseline",
                "limit": args.limit,
                "points": [existing[step] for step in sorted(existing)],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
