"""Synthetic DPO preference generation pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengrad.benchmarks.backends.protocol import InferenceBackend
from opengrad.preferences.deterministic_judge import DeterministicJudge
from opengrad.preferences.openai_judge import OpenAIJudge
from opengrad.preferences.schema import PreferenceCandidate, PreferencePair


@dataclass
class GenerationSummary:
    total_prompts: int
    total_candidates: int
    pairs_generated: int
    deterministic_pairs: int
    openai_pairs: int
    rejected_pairs: int
    output_file: str = ""
    breakdown_by_behavior: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_prompts": self.total_prompts,
            "total_candidates": self.total_candidates,
            "pairs_generated": self.pairs_generated,
            "deterministic_pairs": self.deterministic_pairs,
            "openai_pairs": self.openai_pairs,
            "rejected_pairs": self.rejected_pairs,
            "output_file": self.output_file,
            "breakdown_by_behavior": self.breakdown_by_behavior,
        }


class SyntheticPreferenceGenerator:
    """Generates N student candidates and constructs hard preference pairs."""

    def __init__(
        self,
        backend: InferenceBackend,
        deterministic_judge: DeterministicJudge | None = None,
        openai_judge: OpenAIJudge | None = None,
    ) -> None:
        self.backend = backend
        self.deterministic_judge = deterministic_judge or DeterministicJudge()
        self.openai_judge = openai_judge or OpenAIJudge()

    def generate_pairs(
        self,
        prompts: list[dict[str, Any]],
        output_file: Path,
        *,
        num_candidates_per_prompt: int = 4,
    ) -> GenerationSummary:
        pairs: list[PreferencePair] = []
        det_count = 0
        ai_count = 0
        rejected_count = 0
        total_candidates = 0

        for p_idx, p in enumerate(prompts):
            p_id = str(p.get("id", f"p_{p_idx}"))
            prompt_text = str(p.get("prompt", ""))
            tools = list(p.get("tools") or [])
            expected_dec = str(p.get("expected_decision", "CALL"))
            expected_tool = p.get("expected_tool")

            # 1. Generate N candidates with varying sampling temperatures
            candidates: list[PreferenceCandidate] = []
            temps = [0.0, 0.4, 0.7, 0.9][:num_candidates_per_prompt]
            for c_idx, temp in enumerate(temps):
                total_candidates += 1
                gen_res = self.backend.generate(
                    prompt_text,
                    generation_config={"temperature": temp, "do_sample": temp > 0},
                    tools=tools,
                    metadata={"task_id": f"{p_id}_c{c_idx}", "expected_decision": expected_dec},
                )
                cand = PreferenceCandidate(
                    candidate_id=f"{p_id}_c{c_idx}",
                    response_text=gen_res.text,
                    seed=42 + c_idx,
                    temperature=temp,
                )
                candidates.append(cand)

            # 2. Try pairing best vs worst deterministically (Stage 1)
            pair = None
            if len(candidates) >= 2:
                # If mock generation produced identical texts across temps, create a contrastive negative
                if candidates[0].response_text == candidates[-1].response_text:
                    contrastive_text = (
                        "I cannot help with that unsupported request."
                        if expected_dec == "CALL"
                        else '<tool_call>{"name": "unnecessary_lookup", "arguments": {}}</tool_call>'
                    )
                    candidates[-1] = PreferenceCandidate(
                        candidate_id=f"{p_id}_neg",
                        response_text=contrastive_text,
                        seed=999,
                        temperature=1.0,
                    )

                pair = self.deterministic_judge.judge_pair(
                    candidates[0],
                    candidates[-1],
                    prompt=prompt_text,
                    prompt_id=p_id,
                    expected_decision=expected_dec,
                    expected_tool=expected_tool,
                    available_tools=[t["name"] for t in tools if "name" in t],
                )

            # 3. If ambiguous, try OpenAI judge (Stage 2)
            if not pair and len(candidates) >= 2:
                pair = self.openai_judge.judge_pair(
                    candidates[0], candidates[-1], prompt=prompt_text, prompt_id=p_id, tools=tools
                )
                if pair:
                    ai_count += 1
            elif pair:
                det_count += 1

            # 4. Quality gate: ensure chosen != rejected and valid
            if pair:
                try:
                    pair.validate()
                    pairs.append(pair)
                except ValueError:
                    rejected_count += 1
            else:
                rejected_count += 1

        # Write out to JSONL
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

        return GenerationSummary(
            total_prompts=len(prompts),
            total_candidates=total_candidates,
            pairs_generated=len(pairs),
            deterministic_pairs=det_count,
            openai_pairs=ai_count,
            rejected_pairs=rejected_count,
            output_file=str(output_file),
            breakdown_by_behavior={"tool_policy": len(pairs)},
        )
