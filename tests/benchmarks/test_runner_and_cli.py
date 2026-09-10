from pathlib import Path

import pytest

from opengrad.benchmarks.backends.mock import DeterministicFakeBackend
from opengrad.benchmarks.cli import benchmark_cli
from opengrad.benchmarks.config import BenchmarkConfig
from opengrad.benchmarks.runner import BenchmarkRunner


def test_runner_executes_benchmark_end_to_end(tmp_path: Path) -> None:
    runner = BenchmarkRunner(Path.cwd())
    cfg = BenchmarkConfig.from_file("configs/benchmarks/bfcl_v4.yaml")

    res = runner.run_benchmark(
        cfg,
        dry_run=True,
        limit=2,
        output_dir=tmp_path / "test_run",
    )
    assert res.result["total_tasks"] == 2
    assert (tmp_path / "test_run" / "manifest.json").exists()
    assert (tmp_path / "test_run" / "predictions.jsonl").exists()


def test_runner_executes_suite_end_to_end(tmp_path: Path) -> None:
    runner = BenchmarkRunner(Path.cwd())
    suite_res = runner.run_suite(
        "configs/benchmark_suites/smoke.yaml",
        dry_run=True,
        limit=1,
        output_base_dir=tmp_path / "suite_out",
    )
    assert suite_res["benchmarks_run"] == 5
    assert "bfcl-v4" in suite_res["results"]


def test_adversarial_fixture_handling(tmp_path: Path) -> None:
    # Test runner with adversarial fake backend producing broken syntax
    backend = DeterministicFakeBackend(adversarial=True)
    runner = BenchmarkRunner(Path.cwd())
    cfg = BenchmarkConfig.from_file("configs/benchmarks/bfcl_v4.yaml")

    res = runner.run_benchmark(
        cfg,
        backend=backend,
        limit=5,
        output_dir=tmp_path / "adv_run",
    )
    assert res.result["total_tasks"] == 5
    # Should record failures without crashing
    failures = res.result.get("failure_breakdown", {})
    assert len(failures) > 0


def test_cli_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    assert benchmark_cli(["list"]) == 0
    captured = capsys.readouterr()
    assert "Registered Benchmarks" in captured.out

    assert benchmark_cli(["validate"]) == 0
    captured = capsys.readouterr()
    assert "validated successfully" in captured.out

    assert benchmark_cli(["dry-run"]) == 0
    captured = capsys.readouterr()
    assert "Dry-run COMPLETED" in captured.out
