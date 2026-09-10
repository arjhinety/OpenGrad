"""Benchmark adapters registry and discovery."""

from __future__ import annotations

from pathlib import Path

from opengrad.benchmarks.adapters.acebench import ACEBenchAdapter
from opengrad.benchmarks.adapters.agentbench import AgentBenchFCAdapter
from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.adapters.bfcl import BFCLv4Adapter
from opengrad.benchmarks.adapters.gaia import GAIAAdapter
from opengrad.benchmarks.adapters.ifbench import IFBenchAdapter
from opengrad.benchmarks.adapters.ifeval import IFEvalAdapter
from opengrad.benchmarks.adapters.livebench import LiveBenchAdapter
from opengrad.benchmarks.adapters.mcpmark import MCPMarkAdapter
from opengrad.benchmarks.adapters.microsuite import MicrosuiteAdapter
from opengrad.benchmarks.adapters.mmlu_pro import MMLUProAdapter
from opengrad.benchmarks.adapters.openweights import OpenWeightsAdapter
from opengrad.benchmarks.adapters.regression import ARCChallengeAdapter, GSM8KAdapter
from opengrad.benchmarks.adapters.speculative_replay import SpeculativeReplayAdapter
from opengrad.benchmarks.adapters.tau3 import Tau3Adapter
from opengrad.benchmarks.adapters.terminal import TerminalBenchAdapter, TUABenchAdapter

ADAPTERS_MAP: dict[str, type[BenchmarkAdapter]] = {
    "bfcl-v4": BFCLv4Adapter,
    "bfcl_v4": BFCLv4Adapter,
    "tau3": Tau3Adapter,
    "tau-bench-tau2": Tau3Adapter,
    "acebench": ACEBenchAdapter,
    "ifbench": IFBenchAdapter,
    "ifeval": IFEvalAdapter,
    "livebench": LiveBenchAdapter,
    "mmlu-pro": MMLUProAdapter,
    "mmlu_pro": MMLUProAdapter,
    "gsm8k": GSM8KAdapter,
    "arc-challenge": ARCChallengeAdapter,
    "arc_challenge": ARCChallengeAdapter,
    "mcpmark": MCPMarkAdapter,
    "mcpmark-verified": MCPMarkAdapter,
    "agentbench-fc": AgentBenchFCAdapter,
    "agentbench_fc": AgentBenchFCAdapter,
    "terminal-bench": TerminalBenchAdapter,
    "terminal_bench": TerminalBenchAdapter,
    "tua-bench": TUABenchAdapter,
    "tua_bench": TUABenchAdapter,
    "gaia": GAIAAdapter,
    "openweights": OpenWeightsAdapter,
    "performance-microsuite": MicrosuiteAdapter,
    "performance_microsuite": MicrosuiteAdapter,
    "speculative-replay": SpeculativeReplayAdapter,
    "speculative_replay": SpeculativeReplayAdapter,
}


def get_adapter(benchmark_id: str, root: Path | None = None) -> BenchmarkAdapter:
    canonical = benchmark_id.replace("_", "-")
    cls = ADAPTERS_MAP.get(canonical) or ADAPTERS_MAP.get(benchmark_id)
    if cls is None:
        raise KeyError(
            f"No adapter registered for benchmark '{benchmark_id}'. Available adapters: {sorted(ADAPTERS_MAP.keys())}"
        )
    return cls(root=root)


__all__ = [
    "ADAPTERS_MAP",
    "ACEBenchAdapter",
    "ARCChallengeAdapter",
    "AgentBenchFCAdapter",
    "BFCLv4Adapter",
    "BenchmarkAdapter",
    "BenchmarkTask",
    "GAIAAdapter",
    "GSM8KAdapter",
    "IFBenchAdapter",
    "IFEvalAdapter",
    "LiveBenchAdapter",
    "MCPMarkAdapter",
    "MMLUProAdapter",
    "MicrosuiteAdapter",
    "OpenWeightsAdapter",
    "SpeculativeReplayAdapter",
    "TUABenchAdapter",
    "Tau3Adapter",
    "TerminalBenchAdapter",
    "get_adapter",
]
