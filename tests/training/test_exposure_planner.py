"""The CPU exposure planner (46 §4) replays the trainer's batches and counts tokens where the trainer does."""

from __future__ import annotations

import pytest

from opengrad.training import exposure_planner as ep
from opengrad.training.sft_runner import deterministic_batches


def _plan(n: int = 10, budget: int = 70, **kw):
    settings = {
        "seed": 0,
        "micro_batch_tokens": 20,
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 2,
    }
    settings.update(kw)
    return ep.plan_exposure([10] * n, [5] * n, budget_tokens=budget, **settings)


def test_a_worked_plan_counts_the_discarded_epoch_tail() -> None:
    # Ten equal samples of 10 tokens, 5 supervised: a 20-token micro-batch holds 2 samples (10 tokens
    # supervised), and 2 micro-batches make a step (20 tokens). Epoch 0 has 5 micro-batches: steps 1 and 2
    # close at 20 and 40; the fifth is counted (50) but its window is discarded at the epoch's end, as the
    # trainer zeroes it. Epoch 1's first two micro-batches close step 3 at 70.
    plan = _plan()
    rows = [(m.fraction, m.target_tokens, m.step, m.supervised_tokens) for m in plan.milestones]
    assert rows == [(0.25, 18, 1, 20), (0.5, 35, 2, 40), (0.75, 53, 3, 70), (1.0, 70, 3, 70)]
    assert plan.max_steps == 3 and plan.epochs_started == 2
    assert plan.milestones[-1].examples_seen == 14 and plan.passes == 1.4
    assert [m.within_tolerance for m in plan.milestones] == [False, False, False, True]


def test_the_replay_walks_the_trainers_batches() -> None:
    lengths = [5, 40, 12, 33, 7, 21, 9, 60, 14, 28, 3, 19]
    supervised = [2, 30, 5, 20, 4, 11, 6, 45, 9, 14, 1, 10]
    plan = ep.plan_exposure(
        lengths,
        supervised,
        seed=7,
        micro_batch_tokens=64,
        micro_batch_size=3,
        gradient_accumulation_steps=1,
        budget_tokens=sum(supervised),
        fractions=(1.0,),
    )
    batches = deterministic_batches(lengths, max_tokens=64, max_sequences=3, seed=7, epoch=0)
    assert plan.max_steps == len(batches) and plan.milestones[0].supervised_tokens == sum(
        supervised
    )


def test_check_logged_reports_every_difference() -> None:
    plan = _plan()
    logged = {1: 20, 2: 40, 3: 70}
    assert ep.check_logged(plan, logged) == []
    assert ep.check_logged(plan, {1: 20, 2: 41, 3: 70}) == [
        "step 2: logged 41 supervised tokens, planned 40"
    ]
    assert ep.check_logged(plan, {1: 20, 2: 40}) == ["step 3: no logged tally"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget": 0},
        {"n": 1, "budget": 10**9},
    ],
)
def test_impossible_plans_are_refused(kwargs) -> None:
    with pytest.raises(ep.ExposurePlanError):
        _plan(**kwargs)


def test_the_cli_refuses_a_cache_from_another_arm(tmp_path) -> None:
    import json

    c0 = {
        "model": {"model_id": "Qwen/Qwen3.5-2B", "model_revision": "r", "tokenizer_revision": "r"},
        "datasets": {
            "manifest_ids": ["canonical_v2_final"],
            "hashes": {"canonical_v2_final": "c0hash"},
        },
        "trainer": {"max_seq_length": 2048},
    }
    identity = {
        "corpus_manifest_sha256": "c0hash",
        "model_id": "Qwen/Qwen3.5-2B",
        "model_revision": "r",
        "tokenizer_revision": "r",
        "max_seq_length": 2048,
        "supervision_include": [],
        "exclude_sources": [],
    }
    (tmp_path / "sft_cache.json").write_text(json.dumps({"identity": identity}), encoding="utf-8")
    assert ep.cache_mismatches(tmp_path, c0) == []
    r1 = c0 | {"datasets": {"manifest_ids": ["r1"], "hashes": {"r1": "r1hash"}}}
    assert ep.cache_mismatches(tmp_path, r1) == [
        "corpus_manifest_sha256: cache has 'c0hash', config has 'r1hash'"
    ]
    c2 = c0 | {"datasets": c0["datasets"] | {"exclude_sources": ["when2call"]}}
    assert ep.cache_mismatches(tmp_path, c2) == [
        "exclude_sources: cache has [], config has ['when2call']"
    ]
    s1 = c0 | {"model": c0["model"] | {"model_id": "another-size"}}
    assert ep.cache_mismatches(tmp_path, s1)
    assert ep.cache_mismatches(tmp_path / "nowhere", c0)
