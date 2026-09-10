"""Student rollout generation and policy staleness tracking for on-policy distillation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opengrad.benchmarks.backends.protocol import InferenceBackend
from opengrad.distillation.prompts import PromptState


@dataclass
class DistillationRollout:
    rollout_id: str
    experiment_id: str
    prompt_state_id: str
    student_checkpoint: str
    student_step: int
    student_response: str
    student_generation_config: dict[str, Any] = field(default_factory=dict)
    student_tokens: list[int] = field(default_factory=list)
    teacher_id: str = "Qwen/Qwen3.8-27B"
    teacher_revision: str = "pinned"
    distillation_mode: str = "token_kl"  # "token_kl", "top_k", "sequence_kd"
    policy_staleness_steps: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "experiment_id": self.experiment_id,
            "prompt_state_id": self.prompt_state_id,
            "student_checkpoint": self.student_checkpoint,
            "student_step": self.student_step,
            "student_response": self.student_response,
            "student_generation_config": self.student_generation_config,
            "student_tokens": self.student_tokens,
            "teacher_id": self.teacher_id,
            "teacher_revision": self.teacher_revision,
            "distillation_mode": self.distillation_mode,
            "policy_staleness_steps": self.policy_staleness_steps,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DistillationRollout:
        return cls(
            rollout_id=str(data["rollout_id"]),
            experiment_id=str(data["experiment_id"]),
            prompt_state_id=str(data["prompt_state_id"]),
            student_checkpoint=str(data["student_checkpoint"]),
            student_step=int(data.get("student_step", 0)),
            student_response=str(data["student_response"]),
            student_generation_config=dict(data.get("student_generation_config") or {}),
            student_tokens=list(data.get("student_tokens") or []),
            teacher_id=str(data.get("teacher_id", "Qwen/Qwen3.8-27B")),
            teacher_revision=str(data.get("teacher_revision", "pinned")),
            distillation_mode=str(data.get("distillation_mode", "token_kl")),
            policy_staleness_steps=int(data.get("policy_staleness_steps", 0)),
            created_at=str(data.get("created_at", "")),
        )


class OnPolicyRolloutGenerator:
    """Generates on-policy rollouts from the active student model."""

    def __init__(self, backend: InferenceBackend) -> None:
        self.backend = backend

    def generate(
        self,
        prompt_states: list[PromptState],
        experiment_id: str,
        student_checkpoint: str,
        student_step: int = 0,
        output_file: Path | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> list[DistillationRollout]:
        cfg = generation_config or {"temperature": 0.7, "max_output_tokens": 512, "do_sample": True}
        rollouts: list[DistillationRollout] = []

        for idx, ps in enumerate(prompt_states):
            # Render prefix into prompt string
            prompt_str = f"{ps.system_prompt}\n\n"
            for m in ps.conversation_prefix:
                role = m.get("role", "user").capitalize()
                prompt_str += f"{role}: {m.get('content', '')}\n"
            prompt_str += "Assistant:"

            gen_res = self.backend.generate(
                prompt_str,
                generation_config=cfg,
                tools=ps.tools,
                metadata={"prompt_state_id": ps.prompt_state_id},
            )

            rollout = DistillationRollout(
                rollout_id=f"ro_{experiment_id}_{ps.prompt_state_id}",
                experiment_id=experiment_id,
                prompt_state_id=ps.prompt_state_id,
                student_checkpoint=student_checkpoint,
                student_step=student_step,
                student_response=gen_res.text,
                student_generation_config=cfg,
                student_tokens=gen_res.token_ids,
                policy_staleness_steps=0,
            )
            rollouts.append(rollout)

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w", encoding="utf-8") as f:
                for ro in rollouts:
                    f.write(json.dumps(ro.to_dict(), ensure_ascii=False) + "\n")

        return rollouts
