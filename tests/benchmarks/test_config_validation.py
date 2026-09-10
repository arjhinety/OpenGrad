from pathlib import Path

import pytest

from opengrad.benchmarks.config import (
    BenchmarkConfig,
    BenchmarkSuiteConfig,
    GenerationParameters,
    check_run_compatibility,
)


def test_generation_parameters_missing_required_raises() -> None:
    with pytest.raises(ValueError, match="Missing research-critical generation parameter"):
        GenerationParameters.from_dict({"temperature": 0.0, "seed": 42})


def test_benchmark_config_validation_succeeds_on_valid() -> None:
    cfg_path = Path("configs/benchmarks/bfcl_v4.yaml")
    config = BenchmarkConfig.from_file(cfg_path)
    assert config.benchmark_id == "bfcl-v4"
    assert config.benchmark_revision == "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
    assert config.generation.temperature == 0.0
    assert config.max_context == 4096


def test_benchmark_config_missing_critical_param_raises() -> None:
    with pytest.raises(ValueError, match="missing research-critical parameter"):
        BenchmarkConfig.from_dict({
            "benchmark_id": "test",
            "model_id": "test",
        })


def test_suite_config_validation() -> None:
    suite_path = Path("configs/benchmark_suites/smoke.yaml")
    suite = BenchmarkSuiteConfig.from_file(suite_path)
    assert suite.suite_id == "smoke"
    assert "bfcl_v4" in suite.benchmarks


def test_run_compatibility_checker() -> None:
    cfg1 = BenchmarkConfig.from_file("configs/benchmarks/bfcl_v4.yaml")
    cfg2 = BenchmarkConfig.from_file("configs/benchmarks/bfcl_v4.yaml")
    assert check_run_compatibility(cfg1, cfg2) == []

    # Incompatible temperature
    incompat_dict = cfg2.to_dict()
    incompat_dict["generation"]["temperature"] = 0.7
    cfg_incompat = BenchmarkConfig.from_dict(incompat_dict)
    issues = check_run_compatibility(cfg1, cfg_incompat)
    assert any("Temperature mismatch" in s for s in issues)

    # Incompatible parser
    incompat_dict2 = cfg2.to_dict()
    incompat_dict2["parser"] = "other_parser"
    cfg_incompat2 = BenchmarkConfig.from_dict(incompat_dict2)
    issues2 = check_run_compatibility(cfg1, cfg_incompat2)
    assert any("Parser mismatch" in s for s in issues2)
