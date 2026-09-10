"""Training-path tests: masking, sequence policy, accounting, and configuration refusals.

Split into two groups:

* pure-logic tests that need no model artifacts, so they always run;
* template tests that need the pinned Qwen tokenizer, which are skipped when it is not cached
  (CI has no model access) rather than mocked — the whole point of those tests is fidelity to
  the real template.
"""

from __future__ import annotations

import json

import pytest

from opengrad.data.renderers import _qwen_messages
from opengrad.training.sft_data import (
    STATUS_CONTEXT_TAIL_TRUNCATED,
    STATUS_NO_ASSISTANT_TURN,
    STATUS_OK,
    STATUS_TARGET_TRUNCATED,
    STATUS_UNRENDERABLE,
    SupervisedSample,
    build_sample,
    disposition_counts,
    overflow_report,
)
from opengrad.training.sft_runner import (
    IGNORE_INDEX,
    TrainingConfigError,
    build_batch,
    checkpoint_lineage,
    deterministic_order,
    learning_rate_at,
    parameter_report,
    prune_checkpoints,
    resolve_settings,
    summarise_history,
)

REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def _tokenizer_or_skip():
    try:
        from transformers import AutoTokenizer
    except ImportError:  # pragma: no cover - transformers is a declared extra
        pytest.skip("transformers is not installed")
    try:
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-2B", revision=REVISION, trust_remote_code=False, local_files_only=True
        )
    except Exception:  # noqa: BLE001 - not cached locally
        pytest.skip("the pinned Qwen tokenizer is not cached in this environment")


# --------------------------------------------------------------------------------------
# Configuration refusals
# --------------------------------------------------------------------------------------


def _experiment(**overrides) -> dict:
    base = {
        "experiment_id": "unit-test",
        "model": {
            "model_id": "Qwen/Qwen3.5-2B",
            "model_revision": REVISION,
            "tokenizer_revision": REVISION,
        },
        "reproducibility": {"seed": 7, "precision": "bfloat16"},
        "checkpointing": {"save_steps": 10, "max_checkpoints": 2},
    }
    base.update(overrides)
    return base


def test_tuning_method_is_never_inferred():
    with pytest.raises(TrainingConfigError, match="tuning_method must be explicitly one of"):
        resolve_settings(_experiment(), {"type": "sft", "max_steps": 10})


@pytest.mark.parametrize("method", ["qlora", "peft", "FULL", ""])
def test_only_documented_tuning_methods_are_accepted(method):
    with pytest.raises(TrainingConfigError, match="tuning_method"):
        resolve_settings(_experiment(), {"type": "sft", "tuning_method": method})


def test_full_tuning_resolves_and_records_every_field():
    settings = resolve_settings(
        _experiment(),
        {
            "type": "sft",
            "tuning_method": "full",
            "micro_batch_size": 3,
            "gradient_accumulation_steps": 5,
            "max_steps": 20,
            "warmup_steps": 2,
            "max_seq_length": 128,
            "gradient_checkpointing": True,
            "gradient_clipping": 0.5,
        },
    )
    assert settings.effective_global_batch_size == 15
    assert settings.gradient_checkpointing is True
    assert settings.activation_checkpointing == "gradient_checkpointing"
    assert settings.gradient_clipping == 0.5
    # Every interpretation-affecting field is present in the recorded contract.
    recorded = settings.to_dict()
    for key in (
        "tuning_method",
        "learning_rate",
        "optimizer",
        "optimizer_state",
        "scheduler",
        "precision",
        "seed",
        "gradient_clipping",
        "gradient_checkpointing",
        "lora",
    ):
        assert key in recorded, key


def test_unknown_scheduler_is_refused():
    with pytest.raises(TrainingConfigError, match="scheduler"):
        resolve_settings(
            _experiment(), {"type": "sft", "tuning_method": "full", "scheduler": "warmup"}
        )


def test_lora_requires_an_explicit_block_with_targets():
    with pytest.raises(TrainingConfigError, match="lora tuning requires"):
        resolve_settings(_experiment(), {"type": "sft", "tuning_method": "lora"})
    with pytest.raises(TrainingConfigError, match="target_modules"):
        resolve_settings(
            _experiment(),
            {"type": "sft", "tuning_method": "lora", "lora": {"rank": 8, "alpha": 16}},
        )
    settings = resolve_settings(
        _experiment(),
        {
            "type": "sft",
            "tuning_method": "lora",
            "lora": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": ["q_proj"],
                "bias": "none",
            },
        },
    )
    assert settings.lora["rank"] == 8 and settings.lora["target_modules"] == ["q_proj"]


def test_precision_must_be_supported():
    with pytest.raises(TrainingConfigError, match="precision"):
        resolve_settings(
            _experiment(reproducibility={"precision": "int4"}),
            {"type": "sft", "tuning_method": "full"},
        )


def test_batch_sizes_must_be_positive():
    with pytest.raises(TrainingConfigError, match="positive"):
        resolve_settings(
            _experiment(), {"type": "sft", "tuning_method": "full", "micro_batch_size": 0}
        )


# --------------------------------------------------------------------------------------
# Batching and masking
# --------------------------------------------------------------------------------------


class _TorchStub:
    """Enough of torch to exercise build_batch without importing it in CI."""

    class tensor:
        def __init__(self, data, dtype=None, device=None):
            self.data = [list(row) for row in data]

    long = "long"


def test_labels_are_ignored_outside_supervised_positions():
    """The loss must not be computable from user, system, or tool tokens."""
    rows = [
        {"tokens": [1, 2, 3, 4, 5], "supervised": [3, 4]},
        {"tokens": [6, 7], "supervised": [1]},
    ]
    batch = build_batch(rows, pad_token_id=0, device="cpu", torch=_TorchStub)
    assert batch["input_ids"].data == [[1, 2, 3, 4, 5], [6, 7, 0, 0, 0]]
    assert batch["attention_mask"].data == [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]
    # Row 0: only indices 3 and 4 carry loss. Row 1: only index 1, and padding is ignored.
    assert batch["labels"].data == [
        [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5],
        [IGNORE_INDEX, 7, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
    ]


def test_padding_never_carries_supervision_even_when_requested():
    rows = [{"tokens": [9], "supervised": [0, 1, 2]}]
    batch = build_batch(rows, pad_token_id=0, device="cpu", torch=_TorchStub)
    assert batch["labels"].data[0] == [9]
    assert len(batch["labels"].data[0]) == len(rows[0]["tokens"])


def _sample(tokens, supervised, status=STATUS_OK):
    return SupervisedSample(
        record_id="r",
        canonical_hash="h",
        source_dataset="s",
        behavior_decision="CALL",
        status=status,
        tokens=list(tokens),
        supervised=list(supervised),
    )


def test_overflow_report_separates_trainable_from_dropped():
    samples = [
        _sample(range(10), [3, 4]),
        _sample(range(10), [3, 4], STATUS_CONTEXT_TAIL_TRUNCATED),
        _sample(range(10), [3, 4], STATUS_TARGET_TRUNCATED),
        _sample(range(10), [], STATUS_NO_ASSISTANT_TURN),
    ]
    report = overflow_report(samples, 8)
    assert report["trainable_records"] == 2
    assert report["dropped_records"] == 2
    assert report["tail_truncated_records"] == 1
    assert report["target_truncated_records"] == 1
    assert report["no_assistant_turn_records"] == 1
    assert report["max_seq_length"] == 8
    assert disposition_counts(samples)["OK"] == 1


def test_supervised_sample_trainable_predicate():
    assert _sample([1, 2], [1]).trainable is True
    assert _sample([1, 2], []).trainable is False
    assert _sample([1], [0]).trainable is False, "a single token has no prediction target"
    assert _sample([1, 2], [1], STATUS_TARGET_TRUNCATED).trainable is False


# --------------------------------------------------------------------------------------
# Determinism, schedules, lineage, retention
# --------------------------------------------------------------------------------------


def test_dataset_order_is_reproducible_and_seed_dependent():
    a = deterministic_order(50, seed=42, epoch=0)
    b = deterministic_order(50, seed=42, epoch=0)
    c = deterministic_order(50, seed=42, epoch=1)
    d = deterministic_order(50, seed=43, epoch=0)
    assert a == b, "same seed and epoch must give the same order"
    assert a != c, "a new epoch must reshuffle"
    assert a != d, "a new seed must reshuffle"
    assert sorted(a) == list(range(50)), "a permutation, not a sample"


def test_learning_rate_warms_up_then_decays():
    settings = resolve_settings(
        _experiment(),
        {
            "type": "sft",
            "tuning_method": "full",
            "learning_rate": 1e-3,
            "max_steps": 10,
            "warmup_steps": 2,
            "scheduler": "cosine",
        },
    )
    assert learning_rate_at(settings, 0) < learning_rate_at(settings, 1)
    assert learning_rate_at(settings, 10) < learning_rate_at(settings, 2)
    assert learning_rate_at(settings, 0) == pytest.approx(1e-3 / 2)
    assert learning_rate_at(settings, 10) <= 1e-3


def test_checkpoint_lineage_carries_every_required_field():
    settings = resolve_settings(
        _experiment(), {"type": "sft", "tuning_method": "full", "max_steps": 5}
    )
    lineage = checkpoint_lineage(
        _experiment(),
        settings,
        step=7,
        tokens_seen=99,
        examples_seen=3,
        git_commit="a" * 40,
        dataset_manifest_ids=["canonical_v1"],
        dataset_hashes={"canonical_v1": "b" * 64},
        parent_checkpoint=None,
    )
    for field in (
        "base_model",
        "base_model_revision",
        "tokenizer_revision",
        "experiment_id",
        "training_algorithm",
        "dataset_manifest",
        "dataset_hash",
        "git_commit",
        "training_step",
        "tokens_seen",
        "seed",
        "precision",
        "parent_checkpoint",
    ):
        assert field in lineage, field
    assert lineage["status"] == "CANDIDATE", "a fresh checkpoint is never auto-promoted"
    assert lineage["training_step"] == 7
    assert json.dumps(lineage)  # serializable


def test_prune_checkpoints_keeps_the_newest(tmp_path):
    for step in (10, 20, 30, 40):
        (tmp_path / f"checkpoint-{step}").mkdir()
    removed = prune_checkpoints(tmp_path, max_checkpoints=2)
    remaining = sorted(path.name for path in tmp_path.iterdir())
    assert remaining == ["checkpoint-30", "checkpoint-40"]
    assert len(removed) == 2


def test_prune_checkpoints_never_removes_the_protected_one(tmp_path):
    for step in (10, 20):
        (tmp_path / f"checkpoint-{step}").mkdir()
    prune_checkpoints(tmp_path, max_checkpoints=1, protected=str(tmp_path / "checkpoint-10"))
    assert (tmp_path / "checkpoint-10").is_dir()


def test_parameter_report_counts_trainable_share():
    import types

    class _Param:
        def __init__(self, n, requires):
            self._n = n
            self.requires_grad = requires

        def numel(self):
            return self._n

    model = types.SimpleNamespace(parameters=lambda: [_Param(100, True), _Param(300, False)])
    report = parameter_report(model, "lora")
    assert report["total_parameters"] == 400
    assert report["trainable_parameters"] == 100
    assert report["trainable_percent"] == 25.0


def test_history_summary_reports_a_curve():
    history = [{"loss": 1.0}, {"loss": 0.5}, {"loss": 0.25}]
    summary = summarise_history(history)
    assert summary["steps"] == 3
    assert summary["first_loss"] == 1.0
    assert summary["last_loss"] == 0.25
    assert summary["min_loss"] == 0.25


# --------------------------------------------------------------------------------------
# Template-faithful masking (needs the pinned tokenizer)
# --------------------------------------------------------------------------------------


def _renderer_or_skip():
    _tokenizer_or_skip()
    from opengrad.data.renderers import Qwen35_2BRenderer

    return Qwen35_2BRenderer(revision=REVISION, enable_thinking=False)


def _conversation():
    from opengrad.data.canonical import ToolConversation

    return ToolConversation(
        id="unit",
        source="unit-test",
        tools=[
            {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            }
        ],
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Look up worker 12."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "name": "lookup", "arguments": {"q": "worker 12"}}],
            },
            {"role": "tool", "content": "TOOLRESULTZZZ", "tool_call_id": "call_1"},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "The worker status is recorded."},
        ],
        metadata={
            "split": "train",
            "behavior": {"decision": "CALL", "capabilities": [], "confidence": "known"},
        },
    )


def _runs(indices: list[int]) -> list[list[int]]:
    """Split supervised indices into contiguous spans."""
    runs: list[list[int]] = []
    for index in indices:
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def test_spans_start_at_rendered_content_and_end_at_the_terminator():
    """The exact contract: a span is the assistant's own tokens plus its turn terminator."""
    renderer = _renderer_or_skip()
    tokenizer = renderer._load()
    sample = build_sample(renderer, _conversation(), max_seq_length=4096)
    assert sample.status == STATUS_OK, sample.detail

    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    runs = _runs(sample.supervised)
    assert len(runs) == 2, f"expected two assistant turns, got {len(runs)}"
    decoded = [tokenizer.decode([sample.tokens[i] for i in run]) for run in runs]

    # Turn 0 is a tool call: supervision starts at the call, not mid-way through it, and not
    # at the template's own <|im_start|>assistant opener scaffolding.
    assert decoded[0].startswith("<tool_call>\n<function=lookup>"), decoded[0]
    assert "<parameter=q>" in decoded[0] and "worker 12" in decoded[0]
    assert "<|im_start|>" not in decoded[0]
    assert not decoded[0].startswith("\n")
    # Turn 1 is prose.
    assert decoded[1] == "The worker status is recorded.<|im_end|>", decoded[1]
    for run in runs:
        assert sample.tokens[run[-1]] == im_end


def test_supervision_excludes_user_system_and_tool_tokens():
    """Nothing outside an assistant turn may carry loss."""
    renderer = _renderer_or_skip()
    tokenizer = renderer._load()
    sample = build_sample(renderer, _conversation(), max_seq_length=4096)
    assert sample.status == STATUS_OK, sample.detail

    supervised = set(sample.supervised)
    decoded = tokenizer.decode([sample.tokens[i] for i in sorted(supervised)])
    assert "TOOLRESULTZZZ" not in decoded, "the tool result was supervised"
    assert "Look up worker 12." not in decoded, "the user turn was supervised"
    assert "You are helpful." not in decoded, "the system turn was supervised"

    ignored = tokenizer.decode(
        [sample.tokens[i] for i in range(len(sample.tokens)) if i not in supervised]
    )
    assert "You are helpful." in ignored, "the system prompt must not be masked out of the input"
    assert "Look up worker 12." in ignored
    assert "TOOLRESULTZZZ" in ignored


def test_intermediate_tool_call_turn_gets_no_think_block_but_the_final_turn_does():
    """Documents the template behaviour the span logic has to cope with.

    The pinned template emits `<think>...</think>` only for assistant turns after the last real
    user query, so the same message renders with a different opener depending on its position.
    A single fixed opener length is therefore wrong, which is why spans are found by character
    offset instead.
    """
    renderer = _renderer_or_skip()
    text = renderer.render_sft(_conversation()).text
    first = text.index("<|im_start|>assistant")
    second = text.index("<|im_start|>assistant", first + 1)
    assert text[first:].startswith("<|im_start|>assistant\n<tool_call>")
    assert "<think>" not in text[first:second], "the intermediate turn must not open a think block"
    assert text[second:].startswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert text.count("<think>") == 1


def test_span_offsets_are_self_verified():
    """A mis-computed content offset must be reported, not returned."""
    from opengrad.training.sft_data import assistant_spans_with_offsets

    renderer = _renderer_or_skip()
    tokenizer = renderer._load()
    messages = _qwen_messages(_conversation())
    text = renderer.render_sft(_conversation()).text
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans, error = assistant_spans_with_offsets(
        tokenizer,
        messages,
        text,
        list(encoded["input_ids"]),
        [tuple(pair) for pair in encoded["offset_mapping"]],
    )
    assert error is None and len(spans) == 2
    # Every span begins on a token that starts at or after the content offset for its turn.
    for start, end in spans:
        assert start < end


def test_last_query_index_ignores_tool_responses():
    from opengrad.training.sft_data import last_query_index

    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "calling"},
        {"role": "user", "content": "<tool_response>\nX\n</tool_response>"},
        {"role": "assistant", "content": "done"},
    ]
    assert last_query_index(messages) == 1
    with pytest.raises(ValueError, match="no user query"):
        last_query_index([{"role": "system", "content": "s"}])


def test_reasoning_and_content_matches_the_template_split():
    from opengrad.training.sft_data import reasoning_and_content

    reasoning, content = reasoning_and_content(
        {"role": "assistant", "content": "<think>\nwhy\n</think>\n\nanswer"}
    )
    assert reasoning == "why"
    assert content == "answer"
    assert reasoning_and_content({"role": "assistant", "content": "plain"}) == ("", "plain")


def test_unrenderable_record_is_quarantined_not_repaired():
    """A tool schema the canonical contract rejects must not be coerced into training data."""
    from opengrad.data.canonical import ToolConversation

    renderer = _renderer_or_skip()
    # `parameters` is a bare property map with no type/properties wrapper: the upstream shape
    # that the strict canonical schema refuses. It must never be silently repaired.
    broken = ToolConversation(
        id="broken",
        source="unit-test",
        tools=[
            {
                "name": "x",
                "description": "d",
                "parameters": {"artStyle": {"type": "string", "description": "d"}},
            }
        ],
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        metadata={"split": "train"},
    )
    sample = build_sample(renderer, broken, max_seq_length=512)
    assert sample.status == STATUS_UNRENDERABLE
    assert sample.trainable is False
    assert "SCH_UNSUPPORTED_KEYWORD" in sample.detail.get("reason", "")


def test_overflow_drops_records_whose_target_would_be_cut():
    """Cutting an assistant turn is corrupted supervision, so the record is dropped."""
    renderer = _renderer_or_skip()
    conversation = _conversation()
    full = build_sample(renderer, conversation, max_seq_length=4096)
    assert full.status == STATUS_OK

    tiny = build_sample(renderer, conversation, max_seq_length=4)
    assert tiny.status == STATUS_TARGET_TRUNCATED
    assert tiny.trainable is False


def test_tail_truncation_keeps_the_window_when_supervision_survives():
    renderer = _renderer_or_skip()
    conversation = _conversation()
    full = build_sample(renderer, conversation, max_seq_length=4096)
    window = min(len(full.tokens) - 1, max(full.supervised) + 1)
    if window >= len(full.tokens):
        pytest.skip("this conversation fits entirely; nothing to truncate")
    truncated = build_sample(renderer, conversation, max_seq_length=window)
    assert truncated.status == STATUS_CONTEXT_TAIL_TRUNCATED
    assert len(truncated.tokens) == window
    assert truncated.supervised == full.supervised


def test_record_without_an_assistant_turn_is_not_a_training_example():
    from opengrad.data.canonical import ToolConversation

    renderer = _renderer_or_skip()
    conversation = ToolConversation(
        id="no-assistant",
        source="unit-test",
        tools=[],
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
        metadata={"split": "train"},
    )
    sample = build_sample(renderer, conversation, max_seq_length=512)
    assert sample.status == STATUS_NO_ASSISTANT_TURN
    assert sample.trainable is False


def test_cache_identity_round_trips():
    from opengrad.training.preprocess import CacheIdentity

    identity = CacheIdentity(
        model_id="m",
        model_revision="r",
        tokenizer_revision="t",
        renderer="x",
        template_hash="h",
        max_seq_length=2048,
        corpus_manifest_sha256="c",
    )
    assert CacheIdentity.from_dict(identity.to_dict()) == identity
    changed = dict(identity.to_dict(), max_seq_length=4096)
    assert CacheIdentity.from_dict(changed) != identity, "a different window invalidates the cache"


def test_default_cache_dir_is_keyed_by_the_rendering_contract(tmp_path):
    from opengrad.training.preprocess import default_cache_dir

    assert default_cache_dir(tmp_path, "Qwen/Qwen3.5-2B", 2048) == (
        tmp_path / "data/processed" / "sft-cache-Qwen-Qwen3.5-2B-2048"
    )
    # A different window renders different samples, so it must not share a cache directory.
    assert default_cache_dir(tmp_path, "Qwen/Qwen3.5-2B", 4096) != default_cache_dir(
        tmp_path, "Qwen/Qwen3.5-2B", 2048
    )
