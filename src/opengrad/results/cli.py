"""``opengrad results`` -- inspect and regenerate the derived experiment index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opengrad.results.registry import (
    REGISTRY_RELATIVE_PATH,
    load_registry,
    rebuild_registry,
    registry_exists,
    render_findings,
    render_registry_table,
    validate_registry,
)


def _root() -> Path:
    return Path.cwd()


def results_cli(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="opengrad results",
        description=(
            "Derived experiment index. results/registry.jsonl is a rebuildable projection of "
            "runs/<id>/experiment.json, runs/<id>/eval/ and runs/central_ledger.jsonl."
        ),
    )
    sub = parser.add_subparsers(dest="results_command")

    rebuild = sub.add_parser(
        "rebuild-registry",
        help="regenerate results/registry.jsonl from authoritative run artifacts",
    )
    rebuild.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    validate = sub.add_parser(
        "validate-registry",
        help="report differences between the registry and authoritative state",
    )
    validate.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    show = sub.add_parser("show", help="print the materialized registry summary")
    show.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    args_ns = parser.parse_args(args)
    root = _root()

    if args_ns.results_command == "rebuild-registry":
        summary = rebuild_registry(root)
        if args_ns.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"registry rebuilt: {summary['registry']}")
            print(f"  experiments : {summary['experiments']}")
            print(f"  with evals  : {summary['with_eval_artifacts']}")
            for status, count in summary["status_counts"].items():
                print(f"  {status:<12}: {count}")
            integrity = summary.get("integrity_findings") or []
            if integrity:
                print(f"  authoritative-state gaps: {len(integrity)} (unchanged by this rebuild)")
        return 0

    if args_ns.results_command == "validate-registry":
        findings = validate_registry(root)
        if args_ns.json:
            print(
                json.dumps(
                    {
                        "registry": REGISTRY_RELATIVE_PATH,
                        "status": "DRIFT" if findings else "OK",
                        "findings": [finding.to_dict() for finding in findings],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(render_findings(findings))
        return 1 if any(finding.kind == "DRIFT" for finding in findings) else 0

    if args_ns.results_command == "show":
        rows = load_registry(root)
        if args_ns.json:
            print(json.dumps({"registry": REGISTRY_RELATIVE_PATH, "rows": rows}, indent=2))
        elif not registry_exists(root):
            print(
                f"{REGISTRY_RELATIVE_PATH} does not exist; run `opengrad results rebuild-registry`"
            )
        else:
            print(render_registry_table(rows))
        return 0

    parser.print_help()
    return 0
