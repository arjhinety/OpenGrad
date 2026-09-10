"""Deterministic fake inference backend for GPU-free CPU dry-runs and testing."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from opengrad.benchmarks.backends.protocol import GenerationResult


class DeterministicFakeBackend:
    """CPU backend that returns deterministic outputs and realistic speculative/system metadata."""

    name = "mock"

    def __init__(
        self,
        *,
        mode: str = "ar",  # "ar", "mtp", "speculative_draft", "dspark"
        adversarial: bool = False,
        default_scenario: str = "auto",
        scenarios: dict[str, str] | None = None,
        speedup_factor: float = 1.0,
        degradation: float = 0.0,
    ) -> None:
        self.mode = mode
        self.adversarial = adversarial
        self.default_scenario = default_scenario
        self.scenarios = scenarios or {}
        self.speedup_factor = speedup_factor
        self.degradation = degradation
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        *,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        self.call_count += 1
        meta = metadata or {}
        task_id = str(meta.get("task_id", meta.get("example_id", f"task_{self.call_count}")))
        explicit = self.scenarios.get(task_id) or self.default_scenario

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        digest_int = int(prompt_hash[:4], 16)

        # Decide scenario
        if explicit != "auto":
            scenario = explicit
        elif self.adversarial:
            scenarios_list = [
                "malformed_json",
                "wrong_tool",
                "truncated",
                "multiple_calls",
                "clarification",
                "refusal",
                "valid_call",
            ]
            scenario = scenarios_list[digest_int % len(scenarios_list)]
        else:
            scenario = self._infer_scenario_from_context(prompt, tools, meta)

        text, prompt_tok, comp_tok = self._render_scenario_text(scenario, tools, meta)

        # Base timing simulation
        base_ttft = 0.015
        base_itl = 0.008
        if self.mode in {"mtp", "speculative_draft", "dspark"}:
            simulated_latency = (base_ttft + base_itl * comp_tok) / max(0.1, self.speedup_factor)
            ttft = base_ttft * 1.05  # slight verification / setup overhead
            itl = (simulated_latency - ttft) / max(1, comp_tok)
        else:
            simulated_latency = base_ttft + base_itl * comp_tok
            ttft = base_ttft
            itl = base_itl

        # Speculative metadata
        spec_meta = None
        if self.mode in {"mtp", "speculative_draft", "dspark"}:
            depth = int(meta.get("speculation_depth", 3))
            proposed = comp_tok * depth
            acceptance_rate = max(0.1, min(0.95, 0.75 - self.degradation))
            accepted = int(proposed * acceptance_rate)
            rejected = max(0, proposed - accepted)
            verifier_steps = max(1, comp_tok // max(1, int(1 + depth * acceptance_rate)))
            accepted_per_step = round(accepted / verifier_steps, 2)

            spec_meta = {
                "speculation_mode": self.mode,
                "speculation_depth_requested": depth,
                "speculation_depth_achieved": depth,
                "proposed_tokens": proposed,
                "accepted_tokens": accepted,
                "rejected_tokens": rejected,
                "acceptance_rate": round(acceptance_rate, 4),
                "accepted_tokens_per_step": accepted_per_step,
                "verifier_steps": verifier_steps,
                "rollback_count": int(verifier_steps * 0.15),
                "recomputed_tokens": int(verifier_steps * 0.15 * 2),
                "verification_overhead_ms": round(verifier_steps * 0.4, 2),
                "memory_overhead_mb": 420.0 if self.mode == "mtp" else 1150.0,
                "mtp_per_depth_acceptance": {
                    "depth_1": 0.84,
                    "depth_2": 0.68,
                    "depth_3": 0.49,
                    "depth_4": 0.28,
                }
                if self.mode == "mtp"
                else {},
            }

        return GenerationResult(
            text=text,
            token_ids=[1] + [100 + i for i in range(min(comp_tok, 128))],
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            latency=round(simulated_latency, 6),
            ttft=round(ttft, 6),
            token_timestamps=[round(ttft + i * itl, 6) for i in range(min(comp_tok, 16))],
            backend_metadata={
                "backend": self.name,
                "mode": self.mode,
                "scenario": scenario,
                "task_id": task_id,
                "timestamp": time.time(),
                "peak_vram_mb": 1420.0 if self.mode == "mtp" else 950.0,
            },
            speculative_metadata=spec_meta,
        )

    def _infer_scenario_from_context(
        self, prompt: str, tools: list[dict[str, Any]] | None, meta: dict[str, Any]
    ) -> str:
        expected = str(meta.get("expected_decision", "")).upper()
        if expected in {"CALL", "TOOL_CALL"}:
            return "valid_call"
        if expected in {"CLARIFY", "CLARIFICATION"}:
            return "clarification"
        if expected in {"UNSUPPORTED", "REFUSAL"}:
            return "refusal"
        if expected in {"DIRECT", "ANSWER"}:
            return "direct_answer"

        lower_prompt = prompt.lower()
        if "clarif" in lower_prompt or "ambiguous" in lower_prompt:
            return "clarification"
        if "cannot" in lower_prompt or "illegal" in lower_prompt or "hack" in lower_prompt:
            return "refusal"
        if "multiple" in lower_prompt or "both" in lower_prompt:
            return "multiple_calls"
        if tools and len(tools) > 0:
            return "valid_call"
        return "direct_answer"

    def _render_scenario_text(
        self, scenario: str, tools: list[dict[str, Any]] | None, meta: dict[str, Any]
    ) -> tuple[str, int, int]:
        tool_name = "lookup"
        if tools and len(tools) > 0:
            first_tool = tools[0]
            if isinstance(first_tool, dict):
                tool_name = str(first_tool.get("name", "lookup"))

        if scenario == "valid_call":
            call_obj = {"name": tool_name, "arguments": {"query": "sample query", "limit": 5}}
            text = f"<tool_call>{json.dumps(call_obj, sort_keys=True)}</tool_call>"
            return text, 64, 24

        if scenario == "wrong_tool":
            call_obj = {"name": "non_existent_function", "arguments": {"foo": "bar"}}
            text = f"<tool_call>{json.dumps(call_obj, sort_keys=True)}</tool_call>"
            return text, 64, 20

        if scenario == "malformed_json":
            text = f'<tool_call>{{"name": "{tool_name}", "arguments": {{"unclosed": "missing_quote}}</tool_call>'
            return text, 64, 22

        if scenario == "multiple_calls" or scenario == "parallel_calls":
            call_1 = {"name": tool_name, "arguments": {"id": 101}}
            call_2 = {"name": tool_name, "arguments": {"id": 102}}
            text = (
                f"<tool_call>{json.dumps(call_1, sort_keys=True)}</tool_call>"
                f"<tool_call>{json.dumps(call_2, sort_keys=True)}</tool_call>"
            )
            return text, 80, 45

        if scenario == "clarification":
            text = "Could you please clarify which specific account or dates you want to search?"
            return text, 48, 16

        if scenario == "refusal":
            text = "I cannot help with that unsupported request because it violates safety guidelines."
            return text, 52, 14

        if scenario == "truncated":
            text = f'<tool_call>{{"name": "{tool_name}", "arguments": {{"query": '
            return text, 64, 18

        if scenario == "coding":
            text = "```python\ndef run():\n    return 'success'\n```"
            return text, 70, 20

        # default direct_answer
        text = "The requested information has been verified and processed successfully."
        return text, 45, 12
