"""DPO arithmetic and preference-data contract tests.

The loss and log-probability code is tested directly, because a sign or off-by-one error there
survives every integration test: DPO still trains, and the number still moves, just in the
wrong direction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from opengrad.training.dpo_runner import (
    PreferenceDataError,
    PreferencePair,
    dpo_loss,
    encode_pair,
    load_preference_pairs,
    preference_dataset_identity,
    resolve_dpo_settings,
    sequence_logprob,
)

EXPERIMENT = {
    "experiment_id": "unit",
    "reproducibility": {"seed": 42, "precision": "bfloat16"},
    "checkpointing": {"save_steps": 10, "max_checkpoints": 2},
}


# --------------------------------------------------------------------------------------
# sequence_logprob
# --------------------------------------------------------------------------------------


def test_uniform_logits_give_minus_log_v_per_completion_token():
    """Pins the value, the shift, and the mask in one assertion."""
    batch, length, vocab = 1, 4, 4
    logits = torch.zeros(batch, length, vocab)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    completion_mask = torch.tensor([[0, 1, 1, 0]])

    result = sequence_logprob(logits, input_ids, completion_mask)
    # Uniform distribution -> every token costs log(vocab); two masked-in positions.
    assert result.item() == pytest.approx(-2 * math.log(vocab))


def test_prompt_positions_contribute_nothing():
    logits = torch.zeros(1, 4, 4)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    all_zero = sequence_logprob(logits, input_ids, torch.zeros(1, 4, dtype=torch.long))
    assert all_zero.item() == 0.0


def test_the_mask_is_shifted_with_the_targets():
    """The first position can never carry loss, because nothing predicts it.

    Position t predicts t+1. So a mask on the first token contributes nothing after the shift,
    while a mask on the second token contributes exactly one term. Getting the shift wrong
    reverses these two cases.
    """
    logits = torch.zeros(1, 4, 4)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    first_only = torch.tensor([[1, 0, 0, 0]])
    second_only = torch.tensor([[0, 1, 0, 0]])
    assert sequence_logprob(logits, input_ids, first_only).item() == 0.0
    assert sequence_logprob(logits, input_ids, second_only).item() == pytest.approx(-math.log(4))


# --------------------------------------------------------------------------------------
# dpo_loss
# --------------------------------------------------------------------------------------


def test_loss_falls_and_margin_grows_when_the_policy_prefers_the_chosen_completion():
    policy_chosen, policy_rejected = torch.tensor([0.0]), torch.tensor([-4.0])
    reference_chosen, reference_rejected = torch.tensor([0.0]), torch.tensor([0.0])
    loss, margin, accuracy = dpo_loss(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.5
    )
    assert margin.item() == pytest.approx(2.0)
    assert accuracy.item() == 1.0
    assert loss.item() == pytest.approx(-math.log(1 / (1 + math.exp(-2.0))))


def test_a_policy_that_prefers_the_rejected_completion_is_penalised():
    policy_chosen, policy_rejected = torch.tensor([-4.0]), torch.tensor([0.0])
    reference_chosen, reference_rejected = torch.tensor([0.0]), torch.tensor([0.0])
    loss, margin, accuracy = dpo_loss(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.5
    )
    assert margin.item() < 0
    assert accuracy.item() == 0.0
    assert loss.item() > math.log(2)


def test_reference_cancels_when_both_models_agree():
    """The objective is relative to the reference, so equal shifts change nothing."""
    loss_a, margin_a, _ = dpo_loss(
        torch.tensor([1.0]), torch.tensor([-1.0]), torch.tensor([0.0]), torch.tensor([0.0]), 0.1
    )
    loss_b, margin_b, _ = dpo_loss(
        torch.tensor([5.0]), torch.tensor([3.0]), torch.tensor([4.0]), torch.tensor([4.0]), 0.1
    )
    assert margin_a.item() == pytest.approx(margin_b.item())
    assert loss_a.item() == pytest.approx(loss_b.item())


def test_beta_must_be_positive():
    with pytest.raises(ValueError, match="beta must be positive"):
        dpo_loss(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), 0.0)


def test_accuracy_is_the_share_of_correct_pairs():
    # margins with beta=1 and a zero reference: +2, -2, +2, +2 -> three of four correct.
    policy_chosen = torch.tensor([1.0, -1.0, 1.0, 1.0])
    policy_rejected = torch.tensor([-1.0, 1.0, -1.0, -1.0])
    zeros = torch.zeros(4)
    _, _, accuracy = dpo_loss(policy_chosen, policy_rejected, zeros, zeros, beta=1.0)
    assert accuracy.item() == pytest.approx(0.75)


# --------------------------------------------------------------------------------------
# Preference data
# --------------------------------------------------------------------------------------


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_pairs_load_and_deduplicate(tmp_path):
    path = _write(
        tmp_path / "pairs.jsonl",
        [
            {"prompt": "a", "chosen": "yes", "rejected": "no"},
            {"prompt": "a", "chosen": "yes", "rejected": "no"},
            {"prompt": "b", "chosen": "x", "rejected": "y"},
        ],
    )
    pairs = load_preference_pairs(path)
    assert len(pairs) == 2
    assert isinstance(pairs[0], PreferencePair)


@pytest.mark.parametrize(
    "row,message",
    [
        ({"prompt": "", "chosen": "y", "rejected": "n"}, "missing prompt"),
        ({"prompt": "p", "chosen": "", "rejected": "n"}, "missing chosen"),
        ({"prompt": "p", "chosen": "y"}, "missing rejected"),
    ],
)
def test_incomplete_pairs_are_refused(tmp_path, row, message):
    path = _write(tmp_path / "pairs.jsonl", [row])
    with pytest.raises(PreferenceDataError, match=message.split()[-1]):
        load_preference_pairs(path)


def test_identical_completions_are_refused(tmp_path):
    """A pair with no difference carries no preference direction."""
    path = _write(tmp_path / "pairs.jsonl", [{"prompt": "p", "chosen": "same", "rejected": "same"}])
    with pytest.raises(PreferenceDataError, match="identical"):
        load_preference_pairs(path)


def test_too_few_pairs_is_refused(tmp_path):
    path = _write(tmp_path / "pairs.jsonl", [{"prompt": "p", "chosen": "y", "rejected": "n"}])
    with pytest.raises(PreferenceDataError, match="at least 2"):
        load_preference_pairs(path, min_records=2)


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(PreferenceDataError, match="not found"):
        load_preference_pairs(tmp_path / "absent.jsonl")


def test_malformed_json_is_refused(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(PreferenceDataError, match="not valid JSON"):
        load_preference_pairs(path)


def test_identity_is_a_content_hash_and_count(tmp_path):
    one = _write(tmp_path / "a.jsonl", [{"prompt": "p", "chosen": "y", "rejected": "n"}])
    same = _write(tmp_path / "b.jsonl", [{"prompt": "p", "chosen": "y", "rejected": "n"}])
    other = _write(tmp_path / "c.jsonl", [{"prompt": "p", "chosen": "y2", "rejected": "n"}])
    assert preference_dataset_identity(one)["sha256"] == preference_dataset_identity(same)["sha256"]
    assert preference_dataset_identity(one)["sha256"] != preference_dataset_identity(other)["sha256"]
    assert preference_dataset_identity(one)["records"] == 1


# --------------------------------------------------------------------------------------
# encode_pair
# --------------------------------------------------------------------------------------


class _StubTokenizer:
    """Whitespace tokenizer: enough to test the masking arithmetic without model artifacts."""

    def __call__(self, text, add_special_tokens=False):  # noqa: ARG002
        return {"input_ids": [len(word) for word in text.split()]}


def test_completion_mask_covers_exactly_the_completion():
    ids, mask = encode_pair(_StubTokenizer(), "a bb", " ccc", max_length=16)
    assert mask == [0, 0, 1]


def test_prompt_that_fills_the_window_is_refused():
    with pytest.raises(PreferenceDataError, match="leaves no room"):
        encode_pair(_StubTokenizer(), "a bb cc dd", " e", max_length=4)


def test_over_long_pair_is_refused_rather_than_truncated():
    with pytest.raises(PreferenceDataError, match="over the"):
        encode_pair(_StubTokenizer(), "a", " bb cc dd ee", max_length=4)


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("reference", [None, "policy", "latest", "explicit_checkpoint"])
def test_reference_must_be_explicit_and_valid(reference):
    trainer = {"type": "dpo", "reference": reference}
    with pytest.raises(PreferenceDataError):
        resolve_dpo_settings(EXPERIMENT, trainer)


def test_explicit_checkpoint_reference_must_name_a_checkpoint():
    with pytest.raises(PreferenceDataError, match="reference_checkpoint"):
        resolve_dpo_settings(EXPERIMENT, {"type": "dpo", "reference": "explicit_checkpoint"})


def test_settings_resolve_and_are_recorded():
    settings = resolve_dpo_settings(
        EXPERIMENT,
        {
            "type": "dpo",
            "reference": "initial_policy",
            "beta": 0.2,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_steps": 30,
        },
    )
    assert settings.beta == 0.2
    assert settings.micro_batch_size == 2
    assert settings.reference == "initial_policy"
    for key in ("beta", "reference", "seed", "precision", "max_seq_length", "max_checkpoints"):
        assert key in settings.to_dict()


def test_beta_must_be_positive_in_settings():
    with pytest.raises(PreferenceDataError, match="beta must be positive"):
        resolve_dpo_settings(
            EXPERIMENT, {"type": "dpo", "reference": "initial_policy", "beta": 0.0}
        )
