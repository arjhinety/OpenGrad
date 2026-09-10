"""Contamination screening CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="opengrad-contamination",
        description="Contamination screening for frozen evaluation namespaces",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    screen = sub.add_parser(
        "heldout-screen",
        help="measure contamination levels 1-4 between the held-out set and the training corpus",
    )
    screen.add_argument("--root", type=Path, default=Path.cwd())
    screen.add_argument(
        "--release-dir",
        type=Path,
        default=None,
        help="canonical training release directory (default .release/hf/toolpolicy-canonical-v1)",
    )
    screen.add_argument("--max-df", type=int, default=1000, help="prune shingles above this training document frequency")
    screen.add_argument("--min-shingles", type=int, default=25, help="minimum held-out prompt length for containment flagging")
    screen.add_argument("--jaccard", type=float, default=0.5)
    screen.add_argument("--containment", type=float, default=0.6)
    screen.add_argument("--edit-similarity", type=float, default=0.8)
    screen.add_argument("--json", action="store_true", help="print the full report")

    args = parser.parse_args()
    if args.command == "heldout-screen":
        from opengrad.contamination.heldout import screen_and_write

        report, path = screen_and_write(
            args.root.resolve(),
            release_dir=args.release_dir,
            max_df=args.max_df,
            min_shingles=args.min_shingles,
            jaccard_threshold=args.jaccard,
            containment_threshold=args.containment,
            edit_threshold=args.edit_similarity,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"status: {report['status']}")
            print(f"held-out records: {report['counts']['heldout_records']}")
            print(f"training records: {report['counts']['training_records']}")
            print(f"audit queue size: {report['audit_queue_size']}")
            print(f"report: {path}")
        return 0
    return 1
