"""On-policy distillation trainer backend with decoupled rollout generation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from opengrad.training.protocol import (
    TrainerBackend,
    TrainingMetadata,
    TrainingRunResult,
)
from opengrad.training.teacher import MockTeacherProvider, TeacherProvider


@dataclass
class RolloutRecord:
    rollout_id: str
    prompt_id: str
    student_checkpoint: str
    teacher_model: str
    prompt: str
    student_output: str
    teacher_feedback: str
    score: float
    accepted: bool
    filter_reason: str | None = None
    seed: int = 42
    iteration: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "prompt_id": self.prompt_id,
            "student_checkpoint": self.student_checkpoint,
            "teacher_model": self.teacher_model,
            "prompt": self.prompt,
            "student_output": self.student_output,
            "teacher_feedback": self.teacher_feedback,
            "score": round(self.score, 4),
            "accepted": self.accepted,
            "filter_reason": self.filter_reason,
            "seed": self.seed,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
        }


class RolloutProvider(Protocol):
    """Protocol for generating student policy rollouts.

    Designed to be reused by future RL (GRPO/PPO/RLOO) without restructuring.
    """

    def generate_rollouts(
        self,
        prompts: list[dict[str, Any]],
        checkpoint_id: str,
        *,
        temperature: float = 0.7,
        seed: int = 42,
    ) -> list[dict[str, Any]]: ...


class MockRolloutProvider:
    """Deterministic rollout generator for CPU testing."""

    def generate_rollouts(
        self,
        prompts: list[dict[str, Any]],
        checkpoint_id: str,
        *,
        temperature: float = 0.7,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        results = []
        for idx, p in enumerate(prompts):
            p_id = str(p.get("id", f"p_{idx}"))
            text = str(p.get("prompt", "Sample prompt"))
            results.append(
                {
                    "prompt_id": p_id,
                    "prompt": text,
                    "student_output": f'<tool_call>{{"name": "lookup", "arguments": {{"q": "{p_id}"}}}}</tool_call>',
                    "student_checkpoint": checkpoint_id,
                }
            )
        return results


class OnPolicyDistillationTrainerBackend(TrainerBackend):
    name = "on_policy_distillation"

    def __init__(
        self,
        rollout_provider: RolloutProvider | None = None,
        teacher_provider: TeacherProvider | None = None,
    ) -> None:
        self.rollout_provider = rollout_provider or MockRolloutProvider()
        self.teacher_provider = teacher_provider or MockTeacherProvider()

    def train(
        self,
        experiment_id: str,
        config: dict[str, Any],
        output_dir: Path,
        *,
        dry_run: bool = False,
        experiment: dict[str, Any] | None = None,
        root: Path | None = None,
    ) -> TrainingRunResult:
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        rollouts_dir = output_dir / "rollouts"
        rollouts_dir.mkdir(parents=True, exist_ok=True)

        iterations = int(config.get("iterations", 3))
        prompts_per_iter = int(config.get("prompts_per_iteration", 5))
        min_score = float(config.get("min_teacher_score", 0.7))

        training_meta = TrainingMetadata(
            micro_batch_size=int(config.get("per_device_train_batch_size", 2)),
            gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
            world_size=1,
            effective_global_batch_size=8,
            learning_rate=float(config.get("learning_rate", 1e-5)),
            max_steps=iterations * 5,
            warmup_steps=2,
            tokens_per_update=8 * 2048,
            estimated_total_tokens=iterations * 5 * 8 * 2048,
        )

        (output_dir / "distillation_metadata.json").write_text(
            json.dumps(training_meta.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if dry_run or config.get("backend") == "mock":
            start_time = time.monotonic()
            rollout_records: list[RolloutRecord] = []
            accepted_count = 0
            student_ckpt = "initial_student"

            for iter_idx in range(1, iterations + 1):
                sample_prompts = [
                    {
                        "id": f"prompt_iter{iter_idx}_{i}",
                        "prompt": f"Task query {i} for iteration {iter_idx}",
                    }
                    for i in range(prompts_per_iter)
                ]

                # Step 1: Generate rollouts from student policy
                rollouts = self.rollout_provider.generate_rollouts(
                    sample_prompts, checkpoint_id=student_ckpt
                )

                # Step 2: Score & supervise with TeacherProvider
                for r in rollouts:
                    teacher_res = self.teacher_provider.generate_feedback(
                        r["prompt"], r["student_output"]
                    )
                    is_accepted = teacher_res.score >= min_score
                    record = RolloutRecord(
                        rollout_id=f"ro_{experiment_id}_{r['prompt_id']}",
                        prompt_id=r["prompt_id"],
                        student_checkpoint=student_ckpt,
                        teacher_model=teacher_res.model_id,
                        prompt=r["prompt"],
                        student_output=r["student_output"],
                        teacher_feedback=teacher_res.text,
                        score=teacher_res.score,
                        accepted=is_accepted,
                        filter_reason=None
                        if is_accepted
                        else f"Score {teacher_res.score:.2f} < {min_score:.2f}",
                        iteration=iter_idx,
                    )
                    rollout_records.append(record)
                    if is_accepted:
                        accepted_count += 1

                # Update student checkpoint ID for next iteration
                student_ckpt = f"student_iter_{iter_idx}"

            # Save versioned rollouts artifact
            rollouts_file = rollouts_dir / "rollout_history.jsonl"
            with rollouts_file.open("w", encoding="utf-8") as f:
                for rec in rollout_records:
                    f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

            # Final checkpoint
            final_ckpt = ckpt_dir / "distill-checkpoint-final"
            final_ckpt.mkdir(parents=True, exist_ok=True)
            (final_ckpt / "checkpoint_metadata.json").write_text(
                json.dumps(
                    {
                        "experiment_id": experiment_id,
                        "algorithm": "on_policy_distillation",
                        "iterations_completed": iterations,
                        "rollouts_generated": len(rollout_records),
                        "rollouts_accepted": accepted_count,
                        "status": "CANDIDATE",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            elapsed = time.monotonic() - start_time
            return TrainingRunResult(
                experiment_id=experiment_id,
                algorithm="on_policy_distillation",
                final_checkpoint_path=str(final_ckpt),
                checkpoints_created=[str(final_ckpt)],
                total_steps=iterations * 5,
                total_tokens_seen=accepted_count * 1024,
                final_loss=0.32,
                elapsed_seconds=round(elapsed, 2),
                algorithm_diagnostics={
                    "total_rollouts": len(rollout_records),
                    "accepted_rollouts": accepted_count,
                    "acceptance_rate": round(accepted_count / max(1, len(rollout_records)), 4),
                    "teacher_model": self.teacher_provider.model_id,
                },
            )

        raise NotImplementedError(
            "Live on-policy distillation is deliberately unimplemented: M2 was closed as NOT RUN / NOT JUSTIFIED and needs its own design freeze first (reports/M2_DECISION.md). Use dry_run=True or backend='mock' to exercise the pipeline."
        )
