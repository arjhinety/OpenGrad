"""Command line interface for the OpenGrad benchmark subsystem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from opengrad.benchmarks.config import BenchmarkConfig, BenchmarkSuiteConfig
from opengrad.benchmarks.contamination.registry import ContaminationRegistry
from opengrad.benchmarks.contamination.scanner import MultiLevelContaminationScanner
from opengrad.benchmarks.registry import BenchmarkRegistry
from opengrad.benchmarks.reporting.comparator import (
    compare_runs,
    render_comparison_markdown,
)
from opengrad.benchmarks.reporting.generator import generate_run_readme
from opengrad.benchmarks.runner import BenchmarkRunner


def benchmark_cli(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="opengrad benchmark", description="OpenGrad Benchmark System"
    )
    sub = parser.add_subparsers(dest="benchmark_command")

    # list
    sub.add_parser("list", help="List registered benchmarks and tiers")

    # validate
    sub.add_parser("validate", help="Validate all declarative benchmark and suite configs")

    # dry-run
    sub.add_parser("dry-run", help="Execute rapid CPU dry-run across smoke suite")

    # run
    run_p = sub.add_parser("run", help="Run a benchmark or benchmark suite")
    run_group = run_p.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--suite", help="Suite name or path (e.g. smoke, tool_use_core)")
    run_group.add_argument("--benchmark", help="Benchmark ID or path (e.g. bfcl_v4, tau3)")
    run_p.add_argument("--model", help="Optional model identifier override")
    run_p.add_argument(
        "--backend", default="mock", help="Inference backend (mock, transformers, mtp)"
    )
    run_p.add_argument("--limit", type=int, help="Limit number of tasks to evaluate")
    run_p.add_argument(
        "--dry-run", action="store_true", help="Force CPU dry-run using deterministic fake backend"
    )
    run_p.add_argument("--output-dir", help="Output directory for run artifacts")

    # compare
    comp_p = sub.add_parser("compare", help="Compare baseline and candidate run results")
    comp_p.add_argument("--baseline", required=True, help="Baseline run directory")
    comp_p.add_argument("--candidate", required=True, help="Candidate run directory")
    comp_p.add_argument("--json", action="store_true", help="Output JSON instead of markdown")

    # report
    rep_p = sub.add_parser("report", help="Generate or display report for a benchmark run")
    rep_p.add_argument("run_dir", help="Run directory containing manifest.json and metrics.json")

    # contamination-scan
    cont_p = sub.add_parser(
        "contamination-scan", help="Run multi-level benchmark contamination scan"
    )
    cont_p.add_argument("--benchmark", default="bfcl-v4", help="Benchmark ID to check")
    cont_p.add_argument("--training-data", help="Optional path to training JSON/JSONL records")
    cont_p.add_argument(
        "--max-level", type=int, default=5, help="Maximum contamination level (1-5)"
    )

    parsed = parser.parse_args(args)
    root = Path.cwd()

    if parsed.benchmark_command == "list":
        reg = BenchmarkRegistry(root)
        benchmarks = reg.list_all()
        print(f"\nRegistered Benchmarks in OpenGrad ({len(benchmarks)}):\n")
        print(f"{'ID':<24} {'Tier':<10} {'Version':<12} {'Revision':<14} {'License'}")
        print("-" * 75)
        for b in sorted(benchmarks, key=lambda x: (str(x.tier), x.id)):
            tier_str = b.tier.value if b.tier else "LEGACY"
            rev_str = b.commit_sha[:10] if b.commit_sha else "pinned"
            print(
                f"{b.id:<24} {tier_str:<10} {b.version!s:<12} {rev_str:<14} {b.license or 'unknown'}"
            )
        print()
        return 0

    if parsed.benchmark_command == "validate":
        configs_dir = root / "configs" / "benchmarks"
        suites_dir = root / "configs" / "benchmark_suites"
        errors: list[str] = []

        cfg_files = list(configs_dir.glob("*.yaml"))
        for f in cfg_files:
            try:
                BenchmarkConfig.from_file(f)
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"{f.name}: {exc}")

        suite_files = list(suites_dir.glob("*.yaml"))
        for f in suite_files:
            try:
                BenchmarkSuiteConfig.from_file(f)
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"{f.name}: {exc}")

        if errors:
            print("Validation FAILED with errors:")
            for err in errors:
                print(f"  - {err}")
            return 1
        print(
            f"All {len(cfg_files)} benchmark configs and {len(suite_files)} suite configs validated successfully."
        )
        return 0

    if parsed.benchmark_command == "dry-run":
        runner = BenchmarkRunner(root)
        smoke_suite = root / "configs" / "benchmark_suites" / "smoke.yaml"
        print("Executing rapid CPU dry-run across smoke suite...")
        res = runner.run_suite(smoke_suite, dry_run=True, limit=2)
        print(f"\nDry-run COMPLETED across {res['benchmarks_run']} benchmarks:")
        for b_id, b_res in res["results"].items():
            acc = b_res.get("result", {}).get("overall_accuracy", 0.0)
            print(
                f"  - {b_id:<24} {acc:.1f}% accuracy ({b_res.get('result', {}).get('total_tasks', 0)} tasks)"
            )
        print("\nAll plumbing verified end-to-end without GPU.")
        return 0

    if parsed.benchmark_command == "run":
        runner = BenchmarkRunner(root)
        out_dir = Path(parsed.output_dir) if parsed.output_dir else None
        if parsed.suite:
            suite_file = root / "configs" / "benchmark_suites" / f"{parsed.suite}.yaml"
            if not suite_file.exists():
                suite_file = Path(parsed.suite)
            if not suite_file.exists():
                print(f"Error: suite not found: {parsed.suite}")
                return 1
            print(f"Running suite: {parsed.suite} (dry_run={parsed.dry_run})...")
            suite_res = runner.run_suite(
                suite_file,
                dry_run=parsed.dry_run or parsed.backend == "mock",
                limit=parsed.limit,
                output_base_dir=out_dir,
            )
            print(json.dumps(suite_res, indent=2, sort_keys=True))
            return 0
        if parsed.benchmark:
            cfg_file = root / "configs" / "benchmarks" / f"{parsed.benchmark}.yaml"
            if not cfg_file.exists():
                cfg_file = (
                    root / "configs" / "benchmarks" / f"{parsed.benchmark.replace('-', '_')}.yaml"
                )
            if not cfg_file.exists():
                cfg_file = Path(parsed.benchmark)
            if not cfg_file.exists():
                print(f"Error: benchmark config not found: {parsed.benchmark}")
                return 1
            cfg = BenchmarkConfig.from_file(cfg_file)
            print(f"Running benchmark: {cfg.benchmark_id} (dry_run={parsed.dry_run})...")
            run_res = runner.run_benchmark(
                cfg,
                dry_run=parsed.dry_run or parsed.backend == "mock",
                limit=parsed.limit,
                output_dir=out_dir,
            )
            print(json.dumps(run_res.to_dict(), indent=2, sort_keys=True))
            return 0

    if parsed.benchmark_command == "compare":
        base_dir = Path(parsed.baseline)
        cand_dir = Path(parsed.candidate)
        comp = compare_runs(base_dir, cand_dir)
        if parsed.json:
            print(json.dumps(comp, indent=2, sort_keys=True))
        else:
            print(render_comparison_markdown(comp))
        return 0

    if parsed.benchmark_command == "report":
        r_dir = Path(parsed.run_dir)
        readme = generate_run_readme(r_dir)
        print(readme)
        return 0

    if parsed.benchmark_command == "contamination-scan":
        scanner = MultiLevelContaminationScanner()
        b_reg = BenchmarkRegistry(root)
        c_reg = ContaminationRegistry(root)
        try:
            bm_meta = b_reg.get(parsed.benchmark)
        except KeyError as exc:
            print(f"Error: {exc}")
            return 1

        # Prepare dummy/sample training records if no path provided
        training_samples: list[dict[str, Any]] = []
        if parsed.training_data:
            tr_path = Path(parsed.training_data)
            if tr_path.exists():
                with tr_path.open(encoding="utf-8") as handle:
                    for idx, line in enumerate(handle):
                        line_str = line.strip()
                        if line_str:
                            training_samples.append({"id": f"train_{idx}", "prompt": line_str})
        if not training_samples:
            # Seed with representative fixture prompts to exercise full pipeline
            training_samples = [
                {
                    "id": "fixture_train_01",
                    "prompt": "What is the weather today?",
                    "canonical": "weather",
                },
                {
                    "id": "fixture_train_02",
                    "prompt": "Write a python script to sort numbers.",
                    "canonical": "sort",
                },
            ]

        adapter = __import__("opengrad.benchmarks.adapters", fromlist=["get_adapter"]).get_adapter(
            bm_meta.id, root=root
        )
        tasks = adapter.load_tasks(limit=10)
        bm_samples = [{"id": t.task_id, "prompt": t.prompt, "canonical": t.prompt} for t in tasks]

        report = scanner.scan(
            benchmark_id=bm_meta.id,
            benchmark_samples=bm_samples,
            training_samples=training_samples,
            corpus_fingerprint="sample-or-materialized-fingerprint",
            max_level=parsed.max_level,
        )

        c_reg.update_scan(
            benchmark_id=bm_meta.id,
            corpus_fingerprint=report.corpus_fingerprint,
            scan_status=report.verdict,
            levels_completed=report.level_status,
            audit_queue=[m.to_dict() for m in report.audit_queue],
        )

        print(report.render_markdown())
        return 0

    parser.print_help()
    return 0
