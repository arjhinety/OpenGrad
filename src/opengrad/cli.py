import argparse
import json
import sys
from pathlib import Path

from opengrad.agent_cli import (
    handle_checkpoint_cli,
    handle_distill_cli,
    handle_doctor,
    handle_experiment_cli,
    handle_inspect_template,
    handle_preference_cli,
    handle_promote_reject,
    handle_rollout_cli,
    handle_train,
    handle_validate_data,
)
from opengrad.benchmarks.cli import benchmark_cli
from opengrad.benchmarks.reporting.comparator import compare_runs, render_comparison_markdown
from opengrad.benchmarks.runner import BenchmarkRunner
from opengrad.data.audit import coverage_report, load_records, render_human
from opengrad.data.canonical import ToolConversation
from opengrad.data.semantic import audit_records, validate_training_trajectory
from opengrad.env_capture import capture
from opengrad.evaluation.candidate import CANDIDATE_STATUS, run_candidate_evaluation
from opengrad.evaluation.runner import run_baseline
from opengrad.experiments.preflight import run_experiment_preflight
from opengrad.failures.analyzer import FailureAnalyzer, FailureItem
from opengrad.readiness import gpu_smoke, readiness, repository_status
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


def _evaluation_config_status(config_path: Path) -> str | None:
    """Read just the status field, so an unreadable config still fails in the runner."""
    import yaml

    try:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if isinstance(config, dict):
        return str(config.get("status")) if config.get("status") is not None else None
    return None


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        return benchmark_cli(sys.argv[2:])

    parser = argparse.ArgumentParser(prog="opengrad", description="OpenGrad Research Platform")
    sub = parser.add_subparsers(dest="command")

    validate_p = sub.add_parser("validate", help="validate repository registries")
    validate_p.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    status_p = sub.add_parser("status", help="show authoritative repository and experiment state")
    status_p.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    readiness_p = sub.add_parser("readiness", help="evaluate baseline and SFT readiness gates")
    readiness_p.add_argument(
        "config", nargs="?", help="optional baseline or experiment config path"
    )
    readiness_p.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    gpu_p = sub.add_parser("gpu-smoke", help="run the bounded real-model GPU boundary smoke")
    gpu_p.add_argument("config", nargs="?", help="optional frozen baseline evaluation config")
    gpu_p.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    # preflight
    pre_p = sub.add_parser("preflight", help="pre-experiment or config readiness check")
    pre_p.add_argument("config", nargs="?", help="optional experiment config YAML path")
    pre_p.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    sub.add_parser("benchmark", help="reproducible post-training benchmark system")

    # validate-data
    val_data = sub.add_parser(
        "validate-data", help="strict dataset trajectory and schema validation"
    )
    val_data.add_argument("records", help="JSON or JSONL dataset path")
    val_data.add_argument("--mode", default="sft", choices=["sft", "dpo"], help="dataset mode")
    val_data.add_argument("--json", action="store_true", help="emit JSON output")

    # inspect-template
    ins_temp = sub.add_parser(
        "inspect-template", help="inspect chat template rendering and loss masks"
    )
    ins_temp.add_argument("--record", help="optional path to conversation JSON record")
    ins_temp.add_argument("--thinking", action="store_true", help="enable thinking tokens")
    ins_temp.add_argument("--max-tokens", type=int, default=50, help="max tokens to display")
    ins_temp.add_argument("--json", action="store_true", help="emit JSON output")

    # train
    train_p = sub.add_parser("train", help="launch SFT, DPO, or on-policy distillation training")
    train_p.add_argument("config", help="experiment config YAML path")
    train_p.add_argument(
        "--dry-run", action="store_true", help="execute CPU mock training without GPU"
    )
    train_p.add_argument("--json", action="store_true", help="emit JSON output")

    # evaluate
    eval_p = sub.add_parser("evaluate", help="evaluate a model or checkpoint on a benchmark suite")
    eval_p.add_argument("target", help="checkpoint path or model identifier")
    eval_p.add_argument(
        "--suite", default="smoke", help="suite name (smoke, tool_use_core, full_post_training)"
    )
    eval_p.add_argument(
        "--dry-run", action="store_true", help="force CPU dry-run using mock backend"
    )
    eval_p.add_argument("--limit", type=int, help="limit tasks per benchmark")
    eval_p.add_argument("--json", action="store_true", help="emit JSON output")

    # compare
    comp_p = sub.add_parser("compare", help="compare baseline and candidate runs or checkpoints")
    comp_p.add_argument("baseline", help="baseline run directory")
    comp_p.add_argument("candidate", help="candidate run directory")
    comp_p.add_argument("--json", action="store_true", help="emit JSON output")

    # failures
    fail_p = sub.add_parser("failures", help="analyze and cluster failures in a run directory")
    fail_p.add_argument("run_dir", help="run directory containing failures.json")
    fail_p.add_argument("--json", action="store_true", help="emit JSON output")

    # checkpoint
    ckpt_p = sub.add_parser("checkpoint", help="inspect or list registered checkpoints")
    ckpt_sub = ckpt_p.add_subparsers(dest="checkpoint_command")
    ckpt_list = ckpt_sub.add_parser("list", help="list registered checkpoints")
    ckpt_list.add_argument("--status", help="filter by promotion status")
    ckpt_list.add_argument("--json", action="store_true", help="emit JSON output")
    ckpt_ins = ckpt_sub.add_parser("inspect", help="inspect checkpoint details")
    ckpt_ins.add_argument("checkpoint_id", help="checkpoint ID")
    ckpt_ins.add_argument("--json", action="store_true", help="emit JSON output")

    # experiment
    exp_p = sub.add_parser("experiment", help="manage and inspect experiment records")
    exp_sub = exp_p.add_subparsers(dest="experiment_command")
    exp_list = exp_sub.add_parser("list", help="list experiments")
    exp_list.add_argument("--json", action="store_true", help="emit JSON output")
    exp_show = exp_sub.add_parser("show", help="show experiment details")
    exp_show.add_argument("experiment_id", help="experiment ID")
    exp_show.add_argument("--json", action="store_true", help="emit JSON output")
    exp_diff = exp_sub.add_parser("diff", help="diff two experiments across configurations")
    exp_diff.add_argument("exp_a", help="baseline experiment ID")
    exp_diff.add_argument("exp_b", help="candidate experiment ID")
    exp_diff.add_argument("--json", action="store_true", help="emit JSON output")

    # promote / reject
    prom_p = sub.add_parser("promote", help="promote a checkpoint to validated production status")
    prom_p.add_argument("checkpoint_id", help="checkpoint ID")
    prom_p.add_argument("--reason", help="justification for promotion")
    prom_p.add_argument("--json", action="store_true", help="emit JSON output")

    rej_p = sub.add_parser("reject", help="reject a candidate checkpoint")
    rej_p.add_argument("checkpoint_id", help="checkpoint ID")
    rej_p.add_argument("--reason", help="justification for rejection")
    rej_p.add_argument("--json", action="store_true", help="emit JSON output")

    # doctor
    doc_p = sub.add_parser(
        "doctor", help="diagnose environment, tooling, and on-device testing surfaces"
    )
    doc_p.add_argument("--json", action="store_true", help="emit JSON output")

    # preference
    pref_p = sub.add_parser("preference", help="synthetic DPO preference generation and validation")
    pref_sub = pref_p.add_subparsers(dest="preference_command")
    pref_ins = pref_sub.add_parser("inspect", help="inspect preference records or sample pair")
    pref_ins.add_argument("--records", help="JSONL preference dataset path")
    pref_ins.add_argument("--limit", type=int, default=5, help="limit records to display")
    pref_ins.add_argument("--json", action="store_true", help="emit JSON output")
    pref_gen = pref_sub.add_parser("generate", help="generate synthetic candidate preference pairs")
    pref_gen.add_argument(
        "--output", default="data/processed/synthetic_dpo_pairs.jsonl", help="output path"
    )
    pref_gen.add_argument("--count", type=int, default=8, help="number of prompts to generate for")
    pref_gen.add_argument("--candidates", type=int, default=4, help="candidates per prompt")
    pref_gen.add_argument("--json", action="store_true", help="emit JSON output")
    pref_val = pref_sub.add_parser("validate", help="validate DPO preference pairs")
    pref_val.add_argument("records", help="path to DPO JSONL records")
    pref_val.add_argument("--json", action="store_true", help="emit JSON output")
    pref_bld = pref_sub.add_parser("build", help="build DPO mixture and manifest")
    pref_bld.add_argument("--json", action="store_true", help="emit JSON output")

    # distill
    dist_p = sub.add_parser(
        "distill", help="on-policy distillation teacher validation and training"
    )
    dist_sub = dist_p.add_subparsers(dest="distill_command")
    dist_tok = dist_sub.add_parser(
        "validate-teacher", help="verify tokenizer compatibility between student and teacher"
    )
    dist_tok.add_argument("--student", default="Qwen/Qwen3.5-2B", help="student model ID")
    dist_tok.add_argument("--teacher", default="Qwen/Qwen3.8-27B", help="teacher model ID")
    dist_tok.add_argument("--json", action="store_true", help="emit JSON output")
    dist_bld = dist_sub.add_parser(
        "build-prompts", help="extract eligible prompt states for on-policy rollouts"
    )
    dist_bld.add_argument(
        "--output", default="data/processed/toolpolicy_opd_prompts.jsonl", help="output path"
    )
    dist_bld.add_argument(
        "--profile", default="broad", choices=["broad", "residual"], help="prompt profile"
    )
    dist_bld.add_argument(
        "--count", type=int, default=10, help="number of prompt states to extract"
    )
    dist_bld.add_argument("--json", action="store_true", help="emit JSON output")
    dist_smk = dist_sub.add_parser(
        "smoke", help="run distillation smoke preflight checking VRAM and teacher gap"
    )
    dist_smk.add_argument("--json", action="store_true", help="emit JSON output")
    dist_trn = dist_sub.add_parser("train", help="launch on-policy distillation training")
    dist_trn.add_argument("config", help="distillation experiment config YAML path")
    dist_trn.add_argument("--dry-run", action="store_true", help="force CPU mock training")
    dist_trn.add_argument("--json", action="store_true", help="emit JSON output")

    # rollout
    ro_p = sub.add_parser("rollout", help="inspect on-policy rollout history and staleness stats")
    ro_sub = ro_p.add_subparsers(dest="rollout_command")
    ro_ins = ro_sub.add_parser("inspect", help="inspect rollout records")
    ro_ins.add_argument("--file", help="rollout history JSONL path")
    ro_ins.add_argument("--limit", type=int, default=5, help="limit records to display")
    ro_ins.add_argument("--json", action="store_true", help="emit JSON output")
    ro_stat = ro_sub.add_parser("stats", help="compute rollout acceptance and staleness statistics")
    ro_stat.add_argument("--file", help="rollout history JSONL path")
    ro_stat.add_argument("--json", action="store_true", help="emit JSON output")

    # Legacy CLI tools
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
    env_capture = env.add_subparsers(dest="env_command").add_parser("capture")
    env_capture.add_argument("--json", action="store_true", help="emit machine-readable JSON")
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
    baseline.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    args = parser.parse_args()
    root = Path.cwd()

    if args.command == "validate":
        errors = validate(root)
        if args.json:
            print(
                json.dumps(
                    {
                        "ok": not errors,
                        "errors": errors,
                        "code": None if not errors else "CONFIG_INVALID",
                        "message": "registry validation passed"
                        if not errors
                        else "registry validation failed",
                        "blocking": bool(errors),
                    }
                )
            )
        else:
            print("OK" if not errors else "\n".join(errors))
        return int(bool(errors))

    if args.command == "status":
        payload = repository_status(root)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "readiness":
        payload = readiness(root, root / args.config if args.config else None)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if payload["status"] in {"PASS", "WARN"} else 1

    if args.command == "gpu-smoke":
        payload = gpu_smoke(root, root / args.config if args.config else None)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if payload["status"] == "PASS" else 1

    if args.command == "preflight":
        if getattr(args, "config", None):
            res = run_experiment_preflight(args.config, root=root)
            if args.json:
                print(json.dumps(res.to_dict(), indent=2))
            else:
                print(res.render_summary())
            return 0 if res.overall_status in {"PASS", "WARN"} else 1
        return preflight(root)

    if args.command == "validate-data":
        return handle_validate_data(args, root)

    if args.command == "inspect-template":
        return handle_inspect_template(args, root)

    if args.command == "train":
        return handle_train(args, root)

    if args.command == "evaluate":
        runner = BenchmarkRunner(root)
        allowed_suites = {
            "smoke",
            "tool_use_core",
            "regression_core",
            "agent_transfer",
            "full_post_training",
            "speculative_decoding",
        }
        if args.suite not in allowed_suites:
            err = {
                "code": "SUITE_NOT_FOUND",
                "message": f"Suite must be one of: {', '.join(sorted(allowed_suites))}",
            }
            print(json.dumps(err) if args.json else f"Error: {err['message']}")
            return 1
        suite_path = root / "configs" / "benchmark_suites" / f"{args.suite}.yaml"
        if not suite_path.is_file():
            err = {"code": "SUITE_NOT_FOUND", "message": f"Suite not found: {args.suite}"}
            print(json.dumps(err) if args.json else f"Error: {err['message']}")
            return 1
        res_suite = runner.run_suite(suite_path, dry_run=args.dry_run, limit=args.limit)
        if args.json:
            print(json.dumps(res_suite, indent=2))
        else:
            print(
                f"Evaluated suite '{args.suite}' across {res_suite['benchmarks_run']} benchmarks:"
            )
            for b_id, b_res in res_suite["results"].items():
                acc = b_res.get("result", {}).get("overall_accuracy", 0.0)
                print(f"  - {b_id:<24} {acc:.1f}%")
        return 0

    if args.command == "compare":
        comp = compare_runs(Path(args.baseline), Path(args.candidate))
        if args.json:
            print(json.dumps(comp, indent=2))
        else:
            print(render_comparison_markdown(comp))
        return 0

    if args.command == "failures":
        r_dir = Path(args.run_dir)
        fail_file = r_dir / "failures.json"
        if not fail_file.exists():
            err = {"code": "FAILURES_NOT_FOUND", "message": f"File not found: {fail_file}"}
            print(json.dumps(err) if args.json else f"Error: {err['message']}")
            return 1
        fail_data = json.loads(fail_file.read_text(encoding="utf-8"))
        raw_items = fail_data.get("failures", [])
        analyzer = FailureAnalyzer()
        clusters = analyzer.cluster(
            [
                FailureItem(
                    benchmark=f.get("benchmark", r_dir.name),
                    sample_id=str(f.get("task_id", f.get("sample_id", f"s_{i}"))),
                    prompt=str(f.get("input", f.get("prompt", ""))),
                    expected=f.get("expected"),
                    actual=f.get("parsed_output", f.get("actual")),
                    score=float(f.get("score", 0.0)),
                    failure_category=str(f.get("failure_category", "unknown")),
                    checkpoint_id="unknown",
                    experiment_id="unknown",
                )
                for i, f in enumerate(raw_items)
            ]
        )
        if args.json:
            print(
                json.dumps(
                    {"total_clusters": len(clusters), "clusters": [c.to_dict() for c in clusters]},
                    indent=2,
                )
            )
        else:
            print(f"Failure Analysis: {len(raw_items)} failures in {len(clusters)} clusters\n")
            for c in clusters:
                print(f"- {c.category:<24} {c.count} failures ({c.percentage:.1f}%)")
        return 0

    if args.command == "checkpoint":
        return handle_checkpoint_cli(args, root)

    if args.command == "experiment":
        return handle_experiment_cli(args, root)

    if args.command in {"promote", "reject"}:
        return handle_promote_reject(args, root, args.command)

    if args.command == "doctor":
        return handle_doctor(args, root)

    if args.command == "preference":
        return handle_preference_cli(args, root)

    if args.command == "distill":
        return handle_distill_cli(args, root)

    if args.command == "rollout":
        return handle_rollout_cli(args, root)

    # Legacy CLI dispatch
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
                    failures.append({"line": line_number, "reasons": [code], "details": [str(exc)]})
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
        value = capture(root)
        print(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            if getattr(args, "json", False)
            else value
        )
        return 0

    if args.command == "baseline":
        config_path = root / args.config
        if not args.dry_run:
            gate = readiness(root, config_path)
            smoke_gate = next(
                (item for item in gate.get("gates", []) if item.get("name") == "gpu_boundary"), {}
            )
            if gate.get("ready_for_baseline") is not True or smoke_gate.get("status") != "PASS":
                error = {
                    "ok": False,
                    "code": "BASELINE_NOT_READY",
                    "message": "real baseline is blocked by OpenGrad readiness gates",
                    "blocking": True,
                    "readiness": gate,
                }
                print(json.dumps(error, ensure_ascii=False, indent=2, sort_keys=True))
                return 1
        # A candidate evaluation reuses this command rather than introducing a parallel one,
        # because it is the same measurement against the same frozen held-out set. The config's
        # status field decides which path runs; run_baseline keeps refusing anything that is not
        # the pinned canonical model, so B0 stays immutable.
        candidate_status = _evaluation_config_status(config_path)
        try:
            if candidate_status == CANDIDATE_STATUS:
                result = run_candidate_evaluation(
                    config_path, root=root, limit=args.limit, dry_run=args.dry_run
                )
            else:
                result = run_baseline(
                    config_path, root=root, limit=args.limit, dry_run=args.dry_run
                )
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            error = {
                "ok": False,
                "code": getattr(exc, "code", "BASELINE_FAILED"),
                "message": str(exc),
                "blocking": True,
            }
            print(json.dumps(error, ensure_ascii=False, indent=2, sort_keys=True))
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        # A completed dry run is a successful command: the status field carries
        # DRY_RUN vs EXECUTED. Exiting non-zero here made agent bridges report a
        # successful plumbing run as COMMAND_FAILED.
        return 0 if result.get("status") in {"EXECUTED", "DRY_RUN"} else 1

    parser.print_help()
    return 0


def preflight_cli() -> int:
    return preflight(Path.cwd())
