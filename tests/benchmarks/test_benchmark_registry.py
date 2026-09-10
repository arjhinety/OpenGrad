from pathlib import Path

import pytest

from opengrad.benchmarks.registry import BenchmarkRegistry, BenchmarkTier


def test_registry_loads_all_tiers() -> None:
    reg = BenchmarkRegistry(Path.cwd())
    benchmarks = reg.list_all()
    assert len(benchmarks) >= 15

    tier_a = reg.by_tier(BenchmarkTier.TIER_A)
    assert any(b.id == "bfcl-v4" for b in tier_a)
    assert any(b.id == "tau3" for b in tier_a)
    assert any(b.id == "acebench" for b in tier_a)

    tier_b = reg.by_tier(BenchmarkTier.TIER_B)
    assert any(b.id == "ifeval" for b in tier_b)
    assert any(b.id == "mmlu-pro" for b in tier_b)

    tier_c = reg.by_tier(BenchmarkTier.TIER_C)
    assert any(b.id == "mcpmark" for b in tier_c)
    assert any(b.id == "terminal-bench" for b in tier_c)

    tier_e = reg.by_tier(BenchmarkTier.TIER_E)
    assert any(b.id == "performance-microsuite" for b in tier_e)


def test_registry_pinned_revisions_and_licenses() -> None:
    reg = BenchmarkRegistry(Path.cwd())
    bfcl = reg.get("bfcl-v4")
    assert bfcl.commit_sha == "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
    assert bfcl.license == "Apache-2.0"
    assert bfcl.prohibit_training is True


def test_registry_unknown_benchmark_raises() -> None:
    reg = BenchmarkRegistry(Path.cwd())
    with pytest.raises(KeyError, match="Unknown benchmark"):
        reg.get("non_existent_benchmark_xyz")
