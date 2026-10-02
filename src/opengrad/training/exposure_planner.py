"""Plan an SFT run's steps from a supervised-token budget, on CPU (`study_002_prereg_v14`, 46 §4).

Study 002 holds supervised tokens fixed across arms, not steps. Batches are token-budgeted
(`deterministic_batches`), so how many supervised tokens a step holds depends on the corpus, and Study 001's
"matched" arm saw 1.19x its reference's tokens because its steps were scaled instead (ERRATA §2).

Batch order is a pure function of sample lengths, the seed, the epoch and the two micro-batch budgets. This
module replays the training loop of `run_real_sft` without a model: it walks the same batches in the same
order, counts examples and supervised tokens per micro-batch exactly where the trainer counts them, and closes
an optimizer step every `gradient_accumulation_steps` micro-batches. At an epoch's end the trainer discards an
unfinished accumulation window (its gradients are zeroed) but has already counted its tokens, and the replay
does the same.

The plan's step for a target is the first optimizer step whose cumulative tally reaches it, so the run's
`max_steps` is the 100% step and the cosine schedule ends where training ends. The trainer's logged
`supervised_tokens_seen` at each planned step must equal the plan exactly: any difference is a defect in one of
the two, never noise.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from opengrad.training.sft_runner import deterministic_batches

#: 46 §4: checkpoints at these fractions of the budget, so DEV selection compares arms at equal exposure.
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
#: 46 §4 and 11's v14 note: logged tally within 1% of each target.
TOLERANCE = 0.01
#: A plan that needs more passes than this is a corpus too small for the budget, reported rather than looped.
MAX_EPOCHS = 50


class ExposurePlanError(ValueError):
    pass


@dataclass(frozen=True)
class Milestone:
    fraction: float
    target_tokens: int
    step: int
    supervised_tokens: int
    examples_seen: int
    within_tolerance: bool


@dataclass(frozen=True)
class ExposurePlan:
    seed: int
    micro_batch_tokens: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    budget_tokens: int
    records: int
    corpus_supervised_tokens: int
    max_steps: int
    passes: float
    epochs_started: int
    milestones: list[Milestone] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_exposure(
    lengths: Sequence[int],
    supervised: Sequence[int],
    *,
    seed: int,
    micro_batch_tokens: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    budget_tokens: int,
    fractions: Sequence[float] = CHECKPOINT_FRACTIONS,
) -> ExposurePlan:
    """Replay the batches and return the step at which each fraction of the budget is reached.

    `lengths[i]` is sample i's rendered length (`len(row["tokens"])`) and `supervised[i]` its supervised
    token count (`len(row["supervised"])`), in the cache's row order. `seed` is the trainer's
    `reproducibility.seed + trainer.shuffle_seed_offset`.
    """
    if len(lengths) != len(supervised) or not lengths:
        raise ExposurePlanError(
            "lengths and supervised counts must be non-empty and the same length"
        )
    if budget_tokens < 1 or gradient_accumulation_steps < 1:
        raise ExposurePlanError("the budget and gradient accumulation must be positive")
    if any(not 0 < f <= 1 for f in fractions) or list(fractions) != sorted(fractions):
        raise ExposurePlanError("fractions must be ascending, in (0, 1]")
    corpus_tokens = sum(supervised)
    if corpus_tokens <= 0:
        raise ExposurePlanError("the corpus holds no supervised token")
    targets = [math.ceil(f * budget_tokens) for f in fractions]
    milestones: list[Milestone] = []
    step = examples = tally = 0
    epoch = 0
    while len(milestones) < len(targets):
        if epoch >= MAX_EPOCHS:
            raise ExposurePlanError(
                f"the budget needs more than {MAX_EPOCHS} passes over a corpus of {corpus_tokens:,} tokens"
            )
        in_window = 0
        for batch in deterministic_batches(
            list(lengths),
            max_tokens=micro_batch_tokens,
            max_sequences=micro_batch_size,
            seed=seed,
            epoch=epoch,
        ):
            examples += len(batch)
            tally += sum(supervised[i] for i in batch)
            in_window += 1
            if in_window < gradient_accumulation_steps:
                continue
            in_window = 0
            step += 1
            while len(milestones) < len(targets) and tally >= targets[len(milestones)]:
                target = targets[len(milestones)]
                milestones.append(
                    Milestone(
                        fraction=float(fractions[len(milestones)]),
                        target_tokens=target,
                        step=step,
                        supervised_tokens=tally,
                        examples_seen=examples,
                        within_tolerance=abs(tally - target) <= TOLERANCE * target,
                    )
                )
            if len(milestones) == len(targets):
                break
        epoch += 1
    return ExposurePlan(
        seed=seed,
        micro_batch_tokens=micro_batch_tokens,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        budget_tokens=budget_tokens,
        records=len(lengths),
        corpus_supervised_tokens=corpus_tokens,
        max_steps=milestones[-1].step,
        passes=round(milestones[-1].examples_seen / len(lengths), 6),
        epochs_started=epoch,
        milestones=milestones,
    )


def check_logged(plan: ExposurePlan, logged: dict[int, int]) -> list[str]:
    """Compare a run's logged `supervised_tokens_seen` by step with its plan; every difference is a defect."""
    problems = []
    planned = {m.step: m for m in plan.milestones}  # two fractions can land on one step
    for milestone in planned.values():
        if milestone.step not in logged:
            problems.append(f"step {milestone.step}: no logged tally")
        elif logged[milestone.step] != milestone.supervised_tokens:
            problems.append(
                f"step {milestone.step}: logged {logged[milestone.step]:,} supervised tokens, "
                f"planned {milestone.supervised_tokens:,}"
            )
    return problems


def _trainer_settings(config_path: Path) -> dict[str, int]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    trainer = config.get("trainer", {})
    return {
        "seed": int(config.get("reproducibility", {}).get("seed", 42)),
        "shuffle_seed_offset": int(trainer.get("shuffle_seed_offset", 0)),
        "micro_batch_tokens": int(trainer["micro_batch_tokens"]),
        "micro_batch_size": int(trainer["micro_batch_size"]),
        "gradient_accumulation_steps": int(trainer.get("gradient_accumulation_steps", 1)),
    }


def main(argv: list[str] | None = None) -> int:
    from opengrad.training.preprocess import load_cached_samples

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True, help="rendered sample cache directory")
    parser.add_argument("--config", type=Path, required=True, help="the run's experiment config")
    parser.add_argument("--budget", type=int, required=True, help="supervised-token budget")
    parser.add_argument(
        "--seed", type=int, help="override reproducibility.seed; the offset still applies"
    )
    args = parser.parse_args(argv)
    settings = _trainer_settings(args.config)
    if args.seed is not None:
        settings["seed"] = args.seed
    settings["seed"] += settings.pop("shuffle_seed_offset")
    rows = load_cached_samples(args.cache)
    plan = plan_exposure(
        [len(row["tokens"]) for row in rows],
        [len(row["supervised"]) for row in rows],
        seed=settings["seed"],
        micro_batch_tokens=settings["micro_batch_tokens"],
        micro_batch_size=settings["micro_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        budget_tokens=args.budget,
    )
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
