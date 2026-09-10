"""Supervised Fine-Tuning (SFT) trainer backend."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from opengrad.training.protocol import (
    TrainerBackend,
    TrainingMetadata,
    TrainingRunResult,
)


class SFTTrainerBackend(TrainerBackend):
    name = "sft"

    def train(
        self,
        experiment_id: str,
        config: dict[str, Any],
        output_dir: Path,
        *,
        dry_run: bool = False,
    ) -> TrainingRunResult:
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # 1. Compute batch & token accounting (Section 10)
        micro_batch = int(config.get("micro_batch_size", config.get("per_device_train_batch_size", 2)))
        grad_accum = int(config.get("gradient_accumulation_steps", 4))
        world_size = int(config.get("world_size", 1))
        effective_global_batch = micro_batch * grad_accum * world_size
        lr = float(config.get("learning_rate", 2e-5))
        max_steps = int(config.get("max_steps", 50))
        warmup_steps = int(config.get("warmup_steps", 5))
        seq_len = int(config.get("max_seq_length", 2048))

        tokens_per_update = effective_global_batch * seq_len
        est_tokens = tokens_per_update * max_steps

        training_meta = TrainingMetadata(
            micro_batch_size=micro_batch,
            gradient_accumulation_steps=grad_accum,
            world_size=world_size,
            effective_global_batch_size=effective_global_batch,
            learning_rate=lr,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            tokens_per_update=tokens_per_update,
            estimated_total_tokens=est_tokens,
        )

        (output_dir / "training_metadata.json").write_text(
            json.dumps(training_meta.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if dry_run or config.get("backend") == "mock":
            # Deterministic CPU mock execution
            start_time = time.monotonic()
            steps = min(max_steps, 5)
            metrics_history = []
            final_loss = 1.45
            for step in range(1, steps + 1):
                loss = max(0.2, 1.8 - (step / steps) * 0.9)
                metrics_history.append({
                    "step": step,
                    "loss": round(loss, 4),
                    "learning_rate": lr,
                    "tokens_seen": step * tokens_per_update,
                })
                final_loss = loss

            # Create mock checkpoint artifact
            final_ckpt = ckpt_dir / f"checkpoint-{steps}"
            final_ckpt.mkdir(parents=True, exist_ok=True)
            (final_ckpt / "checkpoint_metadata.json").write_text(
                json.dumps({
                    "experiment_id": experiment_id,
                    "step": steps,
                    "tokens_seen": steps * tokens_per_update,
                    "loss": final_loss,
                    "status": "CANDIDATE",
                }, indent=2) + "\n",
                encoding="utf-8",
            )

            elapsed = time.monotonic() - start_time
            return TrainingRunResult(
                experiment_id=experiment_id,
                algorithm="sft",
                final_checkpoint_path=str(final_ckpt),
                checkpoints_created=[str(final_ckpt)],
                total_steps=steps,
                total_tokens_seen=steps * tokens_per_update,
                final_loss=final_loss,
                elapsed_seconds=round(elapsed, 2),
                metrics_history=metrics_history,
                algorithm_diagnostics={
                    "tokens_per_update": tokens_per_update,
                    "effective_global_batch": effective_global_batch,
                    "assistant_only_loss": True,
                },
            )

        # Real GPU training branch
        raise NotImplementedError(
            "Live GPU training requires accelerator execution. Use dry_run=True for pre-GPU testing."
        )
