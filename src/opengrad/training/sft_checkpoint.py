"""SFT checkpoints: finding the one a run resumes from, and writing one with its full lineage.

Split out of `sft.py` on 2026-09-24 with no change in behaviour; `sft` re-exports `_save_checkpoint`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opengrad.training.sft_runner import (
    checkpoint_lineage,
)


def _find_resume_checkpoint(ckpt_dir: Path) -> Path | None:
    """Newest checkpoint carrying optimizer state, or None when there is nothing to resume."""
    candidates = sorted(
        (path for path in ckpt_dir.glob("checkpoint-*") if (path / "training_state.pt").is_file()),
        key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else 0,
    )
    return candidates[-1] if candidates else None


def _save_checkpoint(
    model: Any,
    optimizer: Any,
    tokenizer: Any,
    output_dir: Path,
    world: dict[str, Any],
    settings: Any,
    experiment: dict[str, Any],
    dataset_ids: list[str],
    dataset_hashes: dict[str, str],
    git_commit: str,
    *,
    suffix: str = "",
    lineage_parent: str | None = None,
    model_components: dict[str, Any] | None = None,
) -> Path:
    """Persist a resumable checkpoint with the lineage the registry requires.

    ``save_pretrained`` writes every carried component, including the attached ``mtp.*`` layer,
    under the base checkpoint's tensor names.
    """
    import torch

    step = world["optimizer_step"]
    name = f"checkpoint-{step}" + (f"-{suffix}" if suffix else "")
    path = output_dir / "checkpoints" / name
    path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "optimizer_step": step,
            "examples_seen": world["examples_seen"],
            "supervised_tokens_seen": world["supervised_tokens_seen"],
            "epoch": world["epoch"],
            "image_batches_seen": world.get("image_batches_seen", 0),
            "mtp_loss_steps": world.get("mtp_loss_steps", 0),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng": __import__("random").getstate(),
        },
        path / "training_state.pt",
    )
    (path / "checkpoint_metadata.json").write_text(
        json.dumps(
            checkpoint_lineage(
                experiment,
                settings,
                step=step,
                tokens_seen=world["supervised_tokens_seen"],
                examples_seen=world["examples_seen"],
                git_commit=git_commit,
                dataset_manifest_ids=dataset_ids,
                dataset_hashes=dataset_hashes,
                parent_checkpoint=lineage_parent,
                model_components=model_components,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    # Optimizer state is roughly twice the size of the weights, and only the newest checkpoint
    # can be resumed from, so older copies are pure storage cost. Measured on this model: a
    # checkpoint is ~11 GiB with state and ~3.8 GiB without, which is the difference between
    # being able to keep a usable evaluation curve and filling the disk after three saves.
    for stale in (output_dir / "checkpoints").glob("checkpoint-*/training_state.pt"):
        if stale.parent != path:
            stale.unlink(missing_ok=True)
    return path
