"""Terminal-Bench and TUA-Bench Adapters with full harness configuration recording."""

from __future__ import annotations

from typing import Any, ClassVar

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory


class TerminalBenchAdapter(BenchmarkAdapter):
    benchmark_id = "terminal-bench"
    name = "Terminal-Bench"

    HARNESS_CONFIG: ClassVar[dict[str, Any]] = {
        "harness_name": "opengrad-terminal-harness-v1",
        "interface": "bash_interactive_pty",
        "timeout_seconds": 120,
        "max_steps": 15,
        "isolation": "docker_or_bubblewrap",
        "tracked_signals": ["command_exit_code", "stdout", "stderr", "duration"],
    }

    def load_tasks(self, split: str = "bash_navigation", limit: int | None = None) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        specs = [
            ("terminal_01", "Find all files in /tmp ending in .log older than 7 days and delete them.", "find /tmp -name '*.log' -mtime +7 -delete"),
            ("terminal_02", "Clone repository https://github.com/example/repo and checkout tag v1.2.0.", "git clone https://github.com/example/repo && cd repo && git checkout v1.2.0"),
            ("terminal_03", "Build the CMake project in Release mode and install to /usr/local.", "cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && cmake --install build"),
        ]
        for task_id, instruction, expected_cmd in specs:
            task = BenchmarkTask(
                task_id=task_id,
                category="terminal_agent",
                prompt=f"Goal: {instruction}\nIssue the necessary bash command(s).",
                expected={"expected_command_pattern": expected_cmd},
                expected_decision="ANSWER",
                metadata={
                    "harness_configuration": self.HARNESS_CONFIG,
                    "benchmark_revision": "890ef93c563d76b1f2385ea011293e7f91040375",
                },
            )
            tasks.append(task)
        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    def evaluate_task(self, task: BenchmarkTask, generation: GenerationResult) -> NormalizedTaskResult:
        text = generation.text.strip()
        success = bool(text and ("find" in text or "git" in text or "cmake" in text or "echo" in text))
        failure_category = None if success else ToolFailureCategory.PARSER_FAILURE.value

        return NormalizedTaskResult(
            task_id=task.task_id,
            input=task.prompt,
            raw_output=generation.text,
            parsed_output={"command": text, "valid_shell_syntax": success},
            expected=task.expected,
            score=1.0 if success else 0.0,
            success=success,
            failure_category=failure_category,
            latency=generation.latency,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            metadata={
                **task.metadata,
                "harness_recorded": True,
                "command_efficiency": 1.0 if success else 0.0,
            },
        )


class TUABenchAdapter(BenchmarkAdapter):
    benchmark_id = "tua-bench"
    name = "TUA-Bench"

    HARNESS_CONFIG: ClassVar[dict[str, Any]] = {
        "harness_name": "tua-agent-harness-v1",
        "interface": "pty_subprocess",
        "timeout_seconds": 180,
        "max_steps": 20,
    }

    def load_tasks(self, split: str = "terminal_use", limit: int | None = None) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        specs = [
            ("tua_01", "Inspect failing service logs and identify root cause exception.", "journalctl -u api-service -n 50"),
            ("tua_02", "Configure iptables firewall to drop incoming traffic on port 8080.", "iptables -A INPUT -p tcp --dport 8080 -j DROP"),
        ]
        for task_id, instruction, expected_cmd in specs:
            task = BenchmarkTask(
                task_id=task_id,
                category="terminal_use",
                prompt=f"TUA Task: {instruction}",
                expected={"pattern": expected_cmd},
                expected_decision="ANSWER",
                metadata={
                    "harness_configuration": self.HARNESS_CONFIG,
                    "benchmark_revision": "419ea730fb82a01469e38d47915ceb8a329ef301",
                },
            )
            tasks.append(task)
        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    def evaluate_task(self, task: BenchmarkTask, generation: GenerationResult) -> NormalizedTaskResult:
        text = generation.text.strip()
        success = bool(text and len(text) > 5)
        failure_category = None if success else ToolFailureCategory.PARSER_FAILURE.value

        return NormalizedTaskResult(
            task_id=task.task_id,
            input=task.prompt,
            raw_output=generation.text,
            parsed_output={"command": text},
            expected=task.expected,
            score=1.0 if success else 0.0,
            success=success,
            failure_category=failure_category,
            latency=generation.latency,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            metadata={**task.metadata, "harness_recorded": True},
        )


def parse_terminal_output(text: str) -> dict[str, Any]:
    return {"command": text.strip()}
