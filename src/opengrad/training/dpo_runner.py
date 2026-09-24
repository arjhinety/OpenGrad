"""DPO pieces that can be tested without a GPU, plus the preference-data contract.

The parts of Direct Preference Optimization that are easy to get subtly wrong are the ones that
are pure arithmetic — the per-sequence log-probability over completion tokens only, and the
implicit-reward margin. Those live here so they can be tested directly, rather than being
buried in a training loop where a sign error would just show up as a slightly worse number.

Preference data is required to be explicit and identified. This repository's DPO configuration
currently pins a dataset identity that resolves to the *evaluation* source's revision and whose
only artifact on disk is four deterministic placeholder rows, so the real training path is
fail-closed: it refuses to start without a preference file it can hash and count. Training on
the evaluation source, or on placeholder rows, would produce a number that means nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opengrad.training.determinism import resolve_determinism
from opengrad.training.model_components import (
    ComponentSettings,
    ModelComponentError,
    resolve_component_settings,
)


class PreferenceDataError(ValueError):
    """The preference dataset is missing, malformed, or too small to train on."""


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    source: str = "unknown"

    def as_dict(self) -> dict[str, str]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "source": self.source,
        }


def preference_dataset_identity(path: Path) -> dict[str, Any]:
    """Content hash and record count of a preference file, for pinning in a config."""
    digest = hashlib.sha256()
    count = 0
    with Path(path).open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                count += 1
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "records": count,
    }


def load_preference_pairs(path: Path, *, min_records: int = 1) -> list[PreferencePair]:
    """Read preference pairs, refusing anything that cannot carry a real signal.

    A pair with an empty field, or with identical chosen and rejected completions, contributes
    no gradient direction and would only dilute the objective while looking like data.
    """
    path = Path(path)
    if not path.is_file():
        raise PreferenceDataError(f"preference dataset not found: {path}")
    pairs: list[PreferencePair] = []
    seen: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreferenceDataError(f"{path}:{number} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise PreferenceDataError(f"{path}:{number} is not a JSON object")
            missing = [key for key in ("prompt", "chosen", "rejected") if not row.get(key)]
            if missing:
                raise PreferenceDataError(f"{path}:{number} is missing {', '.join(missing)}")
            key = (str(row["prompt"]), str(row["chosen"]), str(row["rejected"]))
            if key in seen:
                continue
            if row["chosen"] == row["rejected"]:
                raise PreferenceDataError(
                    f"{path}:{number} has identical chosen and rejected completions"
                )
            seen.add(key)
            pairs.append(
                PreferencePair(
                    prompt=str(row["prompt"]),
                    chosen=str(row["chosen"]),
                    rejected=str(row["rejected"]),
                    source=str(row.get("preference_source") or row.get("source") or "unknown"),
                )
            )
    if len(pairs) < min_records:
        raise PreferenceDataError(
            f"preference dataset {path} has {len(pairs)} usable pairs, need at least {min_records}"
        )
    return pairs


def encode_pair(
    tokenizer: Any,
    prompt: str,
    completion: str,
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Tokenise prompt+completion and mark which positions are the completion.

    The completion mask is what makes the log-probability a preference signal rather than a
    language-modelling score of the prompt: only the tokens the policy actually chose count.
    Truncation is applied from the left of the completion, so a pair whose completion alone
    exceeds the window is refused rather than silently scored on a fragment.
    """
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    full_ids = list(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"])
    if len(prompt_ids) >= max_length:
        raise PreferenceDataError(
            f"prompt alone is {len(prompt_ids)} tokens, which leaves no room in a "
            f"{max_length}-token window for a completion"
        )
    if len(full_ids) > max_length:
        raise PreferenceDataError(
            f"prompt plus completion is {len(full_ids)} tokens, over the {max_length}-token "
            "window; truncating would score a fragment of the chosen response"
        )
    mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    return full_ids, mask


def pad_batch(
    encoded: list[tuple[list[int], list[int]]],
    pad_token_id: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    """Right-pad a batch of (ids, completion mask) pairs, ignoring padding in the loss."""
    width = max(len(ids) for ids, _ in encoded)
    input_ids, attention, mask = [], [], []
    for ids, completion in encoded:
        pad = width - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        attention.append([1] * len(ids) + [0] * pad)
        mask.append(completion + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        "completion_mask": torch.tensor(mask, dtype=torch.long, device=device),
    }


def sequence_logprob(logits: Any, input_ids: Any, completion_mask: Any) -> Any:
    """Sum of log P(token) over completion positions, per sequence.

    Causal shift: the distribution at position t predicts token t+1, so the logits and the
    targets are offset by one, and the mask is shifted with them. Getting this off by one is
    the classic silent bug here — it trains on the prompt's last token and skips the first
    completion token.
    """
    import torch.nn.functional as F

    shifted_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logprobs = F.log_softmax(shifted_logits.float(), dim=-1)
    gathered = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (gathered * completion_mask[:, 1:].float()).sum(dim=1)


def dpo_loss(
    policy_chosen: Any,
    policy_rejected: Any,
    reference_chosen: Any,
    reference_rejected: Any,
    beta: float,
) -> tuple[Any, Any, Any]:
    """DPO loss, implicit-reward margin, and preference accuracy.

    Returns ``(loss, margin, accuracy)``. The margin is the quantity DPO is actually trying to
    grow: how much more the policy prefers the chosen completion than the reference does,
    relative to the rejected one. Accuracy is the share of pairs where that margin is positive.
    """
    import torch.nn.functional as F

    if beta <= 0:
        raise ValueError("beta must be positive")
    chosen_reward = beta * (policy_chosen - reference_chosen)
    rejected_reward = beta * (policy_rejected - reference_rejected)
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin).mean()
    return loss, margin.detach().mean(), (margin.detach() > 0).float().mean()


@dataclass
class DPOSettings:
    beta: float
    learning_rate: float
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    warmup_steps: int
    max_seq_length: int
    precision: str
    seed: int
    save_steps: int
    max_checkpoints: int
    reference: str
    scheduler: str
    model_components: ComponentSettings
    #: `reproducibility.determinism` (`opengrad.training.determinism`); UNDECLARED when absent.
    determinism: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "learning_rate": self.learning_rate,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "max_seq_length": self.max_seq_length,
            "precision": self.precision,
            "seed": self.seed,
            "determinism": self.determinism,
            "save_steps": self.save_steps,
            "max_checkpoints": self.max_checkpoints,
            "reference": self.reference,
            "scheduler": self.scheduler,
            "model_components": self.model_components.to_dict(),
        }


SUPPORTED_REFERENCES = ("initial_policy", "explicit_checkpoint")


def resolve_dpo_settings(experiment: dict[str, Any], trainer: dict[str, Any]) -> DPOSettings:
    """Resolve the DPO contract, refusing anything ambiguous.

    ``reference`` is mandatory. DPO's objective is defined against a frozen reference model, and
    whether that is the initial policy or an explicit earlier checkpoint changes what the run
    measures, so guessing would change the experiment.
    """
    reference = trainer.get("reference")
    if reference not in SUPPORTED_REFERENCES:
        raise PreferenceDataError(
            f"trainer.reference must be explicitly one of {', '.join(SUPPORTED_REFERENCES)}; "
            f"got {reference!r}"
        )
    if reference == "explicit_checkpoint" and not trainer.get("reference_checkpoint"):
        raise PreferenceDataError("reference: explicit_checkpoint requires reference_checkpoint")
    beta = float(trainer.get("beta", 0.1))
    if beta <= 0:
        raise PreferenceDataError("trainer.beta must be positive")
    micro_batch = int(
        trainer.get("per_device_train_batch_size", trainer.get("micro_batch_size", 1))
    )
    grad_accum = int(trainer.get("gradient_accumulation_steps", 1))
    if micro_batch < 1 or grad_accum < 1:
        raise PreferenceDataError("batch sizes must be positive")
    # Components carried and how MTP trains. DPO defaults to `head_only`, so the MTP layer keeps
    # up with the moving policy without adding a likelihood term to the preference objective.
    try:
        model_components = resolve_component_settings(trainer, algorithm="dpo")
    except ModelComponentError as exc:
        raise PreferenceDataError(str(exc)) from exc
    try:
        determinism = resolve_determinism(experiment)
    except ValueError as exc:
        raise PreferenceDataError(str(exc)) from exc
    return DPOSettings(
        beta=beta,
        learning_rate=float(trainer.get("learning_rate", 5e-6)),
        micro_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        max_steps=int(trainer.get("max_steps", 40)),
        warmup_steps=int(trainer.get("warmup_steps", 5)),
        max_seq_length=int(trainer.get("max_seq_length", 2048)),
        precision=str(experiment.get("reproducibility", {}).get("precision", "bfloat16")),
        seed=int(experiment.get("reproducibility", {}).get("seed", 42)),
        save_steps=int((experiment.get("checkpointing") or {}).get("save_steps", 0) or 0),
        max_checkpoints=int((experiment.get("checkpointing") or {}).get("max_checkpoints", 2) or 0),
        reference=str(reference),
        scheduler=str(trainer.get("scheduler", "constant")),
        model_components=model_components,
        determinism=determinism,
    )
