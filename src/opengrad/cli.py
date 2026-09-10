import argparse
import json
import sys
from pathlib import Path

from opengrad.benchmarks.cli import benchmark_cli
from opengrad.data.audit import coverage_report, load_records, render_human
from opengrad.data.canonical import ToolConversation
from opengrad.data.semantic import audit_records, validate_training_trajectory
from opengrad.env_capture import capture
from opengrad.evaluation.runner import run_baseline
from opengrad.registry.preflight import check
from opengrad.registry.validate import validate


def preflight(root: Path) -> int:
    checks = [*check(root).items(), ("Environment capture", bool(capture(root).get("python")))]
    print("OpenGrad Pre-Experiment Readiness\n")
    for name, ok in checks:
        print(f"{name:<24} {'PASS' if ok else 'FAIL'}")
    print("\nDATA_READY / BASELINE_PIPELINE_HARDENING")
    if all(ok for _, ok in checks):
        print("\nREADY_FOR_BASELINE_PIPELINE")
    else:
        print("\nNOT_READY_FOR_BASELINE_PIPELINE")
    return 0 if all(ok for _, ok in checks) else 1


def data_audit_cli() -> int:
    parser = argparse.ArgumentParser(prog="opengrad-data-audit")
    parser.add_argument("--records", required=True, help="JSON array or JSONL canonical records")
    parser.add_argument("--config", help="mixture config retained for audit provenance")
    parser.add_argument("--json", action="store_true", help="also emit machine-readable JSON")
    args = parser.parse_args()
    report = coverage_report(load_records(Path(args.records)))
    print(render_human(report))
    if args.json:
        print("\nJSON_REPORT")
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        return benchmark_cli(sys.argv[2:])

    parser = argparse.ArgumentParser(prog="opengrad")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("validate")
    sub.add_parser("preflight")
    sub.add_parser("benchmark", help="reproducible post-training benchmark system")
    data_audit = sub.add_parser("data-audit")
    data_audit.add_argument(
        "--records", required=True, help="JSON array or JSONL canonical records"
    )
    data_audit.add_argument("--config", help="mixture config retained for audit provenance")
    data_audit.add_argument("--json", action="store_true", help="also emit machine-readable JSON")
    corpus_audit = sub.add_parser(
        "audit-corpus", help="strict semantic audit of canonical training records"
    )
    corpus_audit.add_argument("--records", required=True, help="canonical JSONL records")
    env = sub.add_parser("env")
    env.add_subparsers(dest="env_command").add_parser("capture")
    baseline = sub.add_parser(
        "baseline", help="run the frozen baseline end-to-end (use --dry-run before GPU time)"
    )
    baseline.add_argument(
        "--config",
        default="configs/evaluation/tool_calling/qwen35_2b_baseline.yaml",
        help="frozen baseline YAML",
    )
    baseline.add_argument(
        "--dry-run", action="store_true", help="use the CPU deterministic backend"
    )
    baseline.add_argument("--limit", type=int, help="evaluate only the first N held-out examples")
    args = parser.parse_args()
    root = Path.cwd()
    if args.command == "validate":
        errors = validate(root)
        print("OK" if not errors else "\n".join(errors))
        return int(bool(errors))
    if args.command == "preflight":
        return preflight(root)
    if args.command == "data-audit":
        report = coverage_report(load_records(Path(args.records)))
        print(render_human(report))
        if args.json:
            print("\nJSON_REPORT")
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "audit-corpus":
        records: list[ToolConversation] = []
        failures: list[dict[str, object]] = []
        source_rows = 0
        reason_counts: dict[str, int] = {}
        with Path(args.records).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                source_rows += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError("record must be a JSON object")
                except (json.JSONDecodeError, TypeError) as exc:
                    code = (
                        "INVALID_JSON"
                        if isinstance(exc, json.JSONDecodeError)
                        else "INVALID_RECORD_TYPE"
                    )
                    reason_counts[code] = reason_counts.get(code, 0) + 1
                    failures.append({"line": line_number, "reasons": [code], "details": [str(exc)]})
                    continue
                try:
                    example = ToolConversation(
                        row["id"], row["source"], row["tools"], row["messages"], row["metadata"]
                    )
                    example.validate()
                    records.append(example)
                    issues = validate_training_trajectory(example)
                    if issues:
                        codes = [issue.code for issue in issues]
                        for code in codes:
                            reason_counts[code] = reason_counts.get(code, 0) + 1
                        failures.append(
                            {
                                "line": line_number,
                                "id": example.id,
                                "reasons": codes,
                                "details": [issue.message for issue in issues],
                            }
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    code = "INVALID_CANONICAL_RECORD"
                    reason_counts[code] = reason_counts.get(code, 0) + 1
                    failures.append(
                        {
                            "line": line_number,
                            "reasons": [code],
                            "details": [str(exc)],
                        }
                    )
        summary = audit_records(records)
        summary["records"] = source_rows
        summary["valid"] = source_rows - len(failures)
        summary["invalid"] = len(failures)
        merged_reasons: dict[str, int] = {}
        for code, count in summary["reason_counts"].items():
            merged_reasons[code] = merged_reasons.get(code, 0) + count
        for code, count in reason_counts.items():
            merged_reasons[code] = merged_reasons.get(code, 0) + count
        summary["reason_counts"] = dict(sorted(merged_reasons.items()))
        summary["failures"] = failures
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return int(bool(failures))
    if args.command == "env" and args.env_command == "capture":
        print(capture(root))
        return 0
    if args.command == "baseline":
        result = run_baseline(root / args.config, root=root, limit=args.limit, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    parser.print_help()
    return 0


def preflight_cli() -> int:
    return preflight(Path.cwd())
