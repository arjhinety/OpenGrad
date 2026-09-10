"""Direct Preference Optimization (DPO) trainer backend."""

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


class DPOTrainerBackend(TrainerBackend):
    name = "dpo"

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
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        beta = float(config.get("beta", 0.1))
        lr = float(config.get("learning_rate", 5e-6))
        max_steps = int(config.get("max_steps", 40))
        micro_batch = int(config.get("per_device_train_batch_size", 1))
        grad_accum = int(config.get("gradient_accumulation_steps", 4))
        world_size = int(config.get("world_size", 1))
        effective_global_batch = micro_batch * grad_accum * world_size
        tokens_per_update = effective_global_batch * 2048

        training_meta = TrainingMetadata(
            micro_batch_size=micro_batch,
            gradient_accumulation_steps=grad_accum,
            world_size=world_size,
            effective_global_batch_size=effective_global_batch,
            learning_rate=lr,
            max_steps=max_steps,
            warmup_steps=int(config.get("warmup_steps", 5)),
            tokens_per_update=tokens_per_update,
            estimated_total_tokens=tokens_per_update * max_steps,
        )

        (output_dir / "dpo_training_metadata.json").write_text(
            json.dumps({**training_meta.to_dict(), "beta": beta}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if dry_run or config.get("backend") == "mock":
            start_time = time.monotonic()
            steps = min(max_steps, 5)
            metrics_history = []
            for step in range(1, steps + 1):
                reward_margin = 0.2 + (step / steps) * 0.6
                loss = max(0.1, 0.68 - (step / steps) * 0.3)
                metrics_history.append(
                    {
                        "step": step,
                        "loss": round(loss, 4),
                        "reward_margin": round(reward_margin, 4),
                        "chosen_reward": round(0.4 + (step / steps) * 0.3, 4),
                        "rejected_reward": round(0.2 - (step / steps) * 0.1, 4),
                        "accuracy": round(0.55 + (step / steps) * 0.35, 4),
                    }
                )

            final_ckpt = ckpt_dir / f"dpo-checkpoint-{steps}"
            final_ckpt.mkdir(parents=True, exist_ok=True)
            (final_ckpt / "checkpoint_metadata.json").write_text(
                json.dumps(
                    {
                        "experiment_id": experiment_id,
                        "algorithm": "dpo",
                        "step": steps,
                        "beta": beta,
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
                algorithm="dpo",
                final_checkpoint_path=str(final_ckpt),
                checkpoints_created=[str(final_ckpt)],
                total_steps=steps,
                total_tokens_seen=steps * tokens_per_update,
                final_loss=metrics_history[-1]["loss"],
                elapsed_seconds=round(elapsed, 2),
                metrics_history=metrics_history,
                algorithm_diagnostics={
                    "beta": beta,
                    "final_reward_margin": metrics_history[-1]["reward_margin"],
                    "preference_accuracy": metrics_history[-1]["accuracy"],
                },
            )

        if experiment is None:
            raise ValueError(
                "real DPO requires the full experiment configuration (model, datasets, "
                "checkpointing, reproducibility); it is not inferred from the trainer block"
            )
        if root is None:
            raise ValueError("real DPO requires the repository root")
        from opengrad.training.dpo_live import run_real_dpo

        return run_real_dpo(
            experiment_id=experiment_id,
            experiment=experiment,
            trainer_config=config,
            output_dir=output_dir,
            root=Path(root),
        )
