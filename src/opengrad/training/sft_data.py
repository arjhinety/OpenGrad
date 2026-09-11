"""Model-rendered, assistant-masked SFT samples built from the canonical corpus.

The canonical corpus stores model-independent `messages`. Training must render them through
the model-family renderer, and the loss must fall only on assistant turns. This module is that
boundary, and it is deliberately the *only* place that decides what a training target is.

Assistant spans are computed by the template itself rather than by string matching: for each
assistant message we render the conversation up to that message and up to and including it, and
tokenize both. The template's own output therefore defines the span, which keeps this correct
if the pinned template ever changes. Two properties make that sound, and both are asserted in
tests:

* ``render(messages[:k])`` is a character prefix of ``render(messages)`` for ChatML-style
  templates, so a byte offset from a prefix is a byte offset in the full text;
* tokenizing a prefix yields a token prefix of tokenizing the full text, so span lengths are
  comparable.

A record whose assistant turn cannot be located this way is *quarantined*, never trained with
repaired or guessed supervision. A record with no assistant turn is not a training example at
all and is excluded with its own reason.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from opengrad.data.canonical import ToolConversation
from opengrad.data.renderers import Qwen35_2BRenderer, _qwen_messages
from opengrad.data.supervision import resolve_contract

# Sample disposition. Only OK and CONTEXT_TAIL_TRUNCATED are trainable.
STATUS_OK = "OK"
STATUS_CONTEXT_TAIL_TRUNCATED = "CONTEXT_TAIL_TRUNCATED"
STATUS_TARGET_TRUNCATED = "TARGET_TRUNCATED"
STATUS_NO_ASSISTANT_TURN = "NO_ASSISTANT_TURN"
STATUS_UNRENDERABLE = "UNRENDERABLE"
STATUS_UNMASKABLE = "UNMASKABLE"

TRAINABLE = frozenset({STATUS_OK, STATUS_CONTEXT_TAIL_TRUNCATED})


@dataclass
class SupervisedSample:
    record_id: str
    canonical_hash: str
    source_dataset: str
    behavior_decision: str
    status: str
    supervision_kind: str = ""
    """The contract this sample's target was validated under, from the canonical record."""
    tokens: list[int] = field(default_factory=list)
    supervised: list[int] = field(default_factory=list)
    """Token indices that carry loss (assistant spans only)."""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def trainable(self) -> bool:
        return self.status in TRAINABLE and len(self.tokens) > 1 and len(self.supervised) > 0

    def loss_mask(self) -> list[int]:
        mask = [0] * len(self.tokens)
        for index in self.supervised:
            mask[index] = 1
        return mask


def marker_ids(tokenizer: Any) -> tuple[int, int, list[int]]:
    """The template's own ChatML turn delimiters, read from the tokenizer.

    These are added tokens in the Qwen vocabulary, so they are single ids rather than text to
    be matched. Reading them from the tokenizer keeps this tied to the pinned artifact instead
    of a hard-coded number.
    """
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant = tokenizer.encode("assistant", add_special_tokens=False)
    if not isinstance(im_start, int) or im_start < 0:
        raise ValueError("tokenizer has no <|im_start|> token")
    if not isinstance(im_end, int) or im_end < 0:
        raise ValueError("tokenizer has no <|im_end|> token")
    if not assistant:
        raise ValueError("tokenizer cannot encode 'assistant'")
    return im_start, im_end, assistant


def _content_text(message: dict[str, Any]) -> str:
    """Message content as text, flattening the multimodal list form the template accepts."""
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def last_query_index(messages: list[dict[str, Any]]) -> int:
    """Index of the last genuine user query, mirroring the pinned template.

    The template scans backwards for the last ``user`` message whose content is not a
    ``<tool_response>`` wrapper, and the assistant branch keys off that index to decide whether
    to emit a ``<think>`` block. Duplicating the rule here is deliberate: assuming a fixed
    opener is wrong, because an intermediate tool-call turn renders without the block while the
    final turn renders with it.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue
        content = _content_text(message).strip()
        if not (content.startswith("<tool_response>") and content.endswith("</tool_response>")):
            return index
    raise ValueError("no user query found in messages")


def reasoning_and_content(message: dict[str, Any]) -> tuple[str, str]:
    """Split an assistant message the way the pinned template does.

    The template lifts any ``<think>`` region out of the content and re-emits it inside the
    fixed opener, so reproducing the split is what makes the content offset computable.
    """
    content = _content_text(message).strip()
    reasoning = ""
    explicit = message.get("reasoning_content")
    if isinstance(explicit, str):
        reasoning = explicit
    elif "</think>" in content:
        reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
        content = content.split("</think>")[-1].lstrip("\n")
    return reasoning.strip(), content


def assistant_spans_with_offsets(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    text: str,
    tokens: list[int],
    offsets: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], str | None]:
    """Locate assistant target spans using the tokenizer's character offsets.

    Character offsets are the reliable coordinate system here, and a fixed opener length is not.
    When the template renders this conversation, the ``<think>`` block appears only for the
    assistant turn after the last real user query, so the same message gets a different opener
    depending on its position. Prefix-rendering to measure that per turn appeared to cost a
    template render per message boundary and did not even agree with the full render at token
    level. Working in characters and mapping back through ``offset_mapping`` keeps every span
    anchored to the text that was actually produced.

    Each span is self-checked: the computed offset must land on that message's rendered content.
    A span that does not is reported unmaskable rather than returned, because a target that is
    off by a few tokens is corrupted supervision that no later stage would notice.
    """
    im_start, im_end, assistant = marker_ids(tokenizer)
    query_index = last_query_index(messages)

    turns = [index for index, message in enumerate(messages) if message.get("role") == "assistant"]
    if not turns:
        return [], STATUS_NO_ASSISTANT_TURN

    marker_positions = [
        position
        for position, token in enumerate(tokens)
        if token == im_start and tokens[position + 1 : position + 1 + len(assistant)] == assistant
    ]
    if len(marker_positions) != len(turns):
        return [], (
            f"{STATUS_UNMASKABLE}: found {len(marker_positions)} assistant markers for "
            f"{len(turns)} assistant messages"
        )

    spans: list[tuple[int, int]] = []
    for message_index, marker in zip(turns, marker_positions):
        after_role = offsets[marker + len(assistant)][1]
        reasoning, content = reasoning_and_content(messages[message_index])
        opener = f"\n<think>\n{reasoning}\n</think>\n\n" if message_index > query_index else "\n"
        content_start = after_role + len(opener)

        end = marker
        while end < len(tokens) and tokens[end] != im_end:
            end += 1
        if end >= len(tokens):
            return [], f"{STATUS_UNMASKABLE}: assistant turn is not closed by <|im_end|>"

        if content:
            if not text[content_start:].startswith(content):
                return [], (
                    f"{STATUS_UNMASKABLE}: computed content offset does not match the rendered "
                    f"content of assistant message {message_index}"
                )
        else:
            # A turn with no prose (tool call only) begins at the first token after the opener.
            content_start = offsets[marker + len(assistant)][1] + len(opener)

        start_token = next(
            (position for position in range(marker, end) if offsets[position][0] >= content_start),
            None,
        )
        if start_token is None or end <= start_token:
            continue
        spans.append((start_token, end + 1))

    if not spans:
        return [], STATUS_NO_ASSISTANT_TURN
    return spans, None


def build_sample(
    renderer: Qwen35_2BRenderer,
    conversation: ToolConversation,
    *,
    max_seq_length: int,
    canonical_hash: str = "",
    source_dataset: str = "",
    behavior_decision: str = "",
) -> SupervisedSample:
    """Render, mask, and apply the deterministic sequence-length policy.

    Order matters for correct classification: the full render happens first, so a record whose
    tool schema the canonical validator rejects is reported as ``UNRENDERABLE`` rather than
    being mistaken for a masking failure.

    Length policy, applied after masking so the target is never silently lost:

    * the whole sequence fits: keep it, status ``OK``;
    * it does not fit but every supervised token does: keep the first ``max_seq_length``
      tokens. All targets survive, only trailing non-target context is dropped
      (``CONTEXT_TAIL_TRUNCATED``);
    * it does not fit and any supervised token would be cut: drop. Training on a partially
      removed assistant turn is corrupted supervision, so the record is reported as
      ``TARGET_TRUNCATED`` instead.
    """
    sample = SupervisedSample(
        record_id=conversation.id,
        canonical_hash=canonical_hash,
        source_dataset=source_dataset or conversation.source,
        behavior_decision=behavior_decision,
        status=STATUS_OK,
    )

    # Resolve the contract up front so every exit path reports which rules judged this record.
    try:
        contract, assignment = resolve_contract(conversation.metadata)
    except (TypeError, ValueError) as exc:
        sample.status = STATUS_UNRENDERABLE
        sample.detail["reason"] = f"SUPERVISION_INVALID: {exc}"
        return sample
    sample.supervision_kind = contract.kind.value
    sample.detail["supervision_kind"] = contract.kind.value
    sample.detail["supervision_assignment"] = assignment
    sample.detail["validation_policy"] = contract.validation_policy

    # Shape is checked before rendering so that a record which simply has no assistant turn is
    # reported as such. The canonical validator rejects it too, but reporting it as a render
    # failure would hide the real reason and inflate the unrenderable count.
    try:
        messages = _qwen_messages(conversation)
    except Exception as exc:  # noqa: BLE001 - malformed message shapes are quarantined
        sample.status = STATUS_UNRENDERABLE
        sample.detail["reason"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return sample
    expected_turns = sum(1 for message in messages if message.get("role") == "assistant")
    if expected_turns == 0:
        sample.status = STATUS_NO_ASSISTANT_TURN
        sample.detail["reason"] = "record has no assistant turn"
        return sample
    if not any(message.get("role") == "user" for message in messages):
        sample.status = STATUS_NO_ASSISTANT_TURN
        sample.detail["reason"] = "record has no user turn"
        return sample

    try:
        rendered = renderer.render_sft(conversation)
    except Exception as exc:  # noqa: BLE001 - unrenderable records are quarantined
        sample.status = STATUS_UNRENDERABLE
        sample.detail["reason"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return sample

    tokenizer = renderer._load()
    encoded = tokenizer(rendered.text, add_special_tokens=False, return_offsets_mapping=True)
    tokens: list[int] = list(encoded["input_ids"])
    offsets: list[tuple[int, int]] = [tuple(pair) for pair in encoded["offset_mapping"]]

    try:
        spans, error = assistant_spans_with_offsets(
            tokenizer, messages, rendered.text, tokens, offsets
        )
    except ValueError as exc:
        sample.status = STATUS_UNMASKABLE
        sample.detail["reason"] = str(exc)
        return sample
    if error is not None:
        sample.status = error.split(":", 1)[0]
        sample.detail["reason"] = error
        return sample

    supervised = [index for start, end in spans for index in range(start, min(end, len(tokens)))]
    # A span reaching past the rendered token list means the prefix and full renders disagree;
    # that is a template anomaly, and guessing a truncation for it would be repairing data.
    if any(end > len(tokens) for _, end in spans):
        sample.status = STATUS_UNMASKABLE
        sample.detail["reason"] = "assistant span extends past the rendered sequence"
        return sample

    sample.detail["spans"] = spans
    sample.detail["rendered_tokens"] = len(tokens)
    sample.detail["supervised_tokens"] = len(supervised)
    sample.detail["template_hash"] = rendered.chat_template_hash

    # Under CALL_PREDICTION the terminal assistant call *is* the target, so it must carry loss.
    # Assistant-span masking already covers every assistant turn, which means a call-prediction
    # record needs no special rendering; this asserts that rather than assuming it, so a future
    # masking change cannot silently drop the call from the loss and leave a sample that looks
    # trainable while supervising nothing.
    #
    # `spans` is ordered by assistant turn, and the contract requires the target to be the final
    # assistant message, so a one-span-per-turn count makes `spans[-1]` that turn's span.
    if contract.terminal_target_type == "assistant_tool_call":
        turns = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        if len(spans) != len(turns):
            sample.status = STATUS_UNMASKABLE
            sample.detail["reason"] = (
                f"{contract.kind.value} needs a supervised span per assistant turn, "
                f"found {len(spans)} for {len(turns)}"
            )
            return sample
        target_start, target_end = spans[-1]
        if not any(target_start <= index < target_end for index in supervised):
            sample.status = STATUS_UNMASKABLE
            sample.detail["reason"] = (
                f"{contract.kind.value} terminal call span carries no supervised token"
            )
            return sample
        sample.detail["target_span"] = [target_start, target_end]

    if len(tokens) <= max_seq_length:
        sample.tokens = tokens
        sample.supervised = supervised
        return sample

    if supervised and max(supervised) >= max_seq_length:
        sample.status = STATUS_TARGET_TRUNCATED
        sample.detail["first_supervised"] = min(supervised)
        sample.detail["last_supervised"] = max(supervised)
        return sample

    sample.status = STATUS_CONTEXT_TAIL_TRUNCATED
    sample.detail["full_length"] = len(tokens)
    sample.tokens = tokens[:max_seq_length]
    sample.supervised = supervised
    return sample


def sample_fingerprint(samples: list[SupervisedSample]) -> str:
    """Stable digest of the trainable sample stream, for cache invalidation."""
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            json.dumps(
                [sample.record_id, sample.status, len(sample.tokens), len(sample.supervised)],
                separators=(",", ":"),
            ).encode()
        )
    return digest.hexdigest()


def disposition_counts(samples: list[SupervisedSample]) -> dict[str, int]:
    return dict(sorted(Counter(sample.status for sample in samples).items()))


def supervision_kind_counts(samples: list[SupervisedSample]) -> dict[str, int]:
    """Trainable samples per supervision kind.

    Reported separately so a corpus of call-prediction records is never summarised as though it
    were a corpus of complete trajectories.
    """
    return dict(
        sorted(
            Counter(
                sample.supervision_kind or "UNCLASSIFIED"
                for sample in samples
                if sample.status in TRAINABLE
            ).items()
        )
    )


def supervision_composition(samples: list[SupervisedSample]) -> dict[str, Any]:
    """Per-kind record and target counts, with the mixture as a fraction of trainable records."""
    trainable = [s for s in samples if s.status in TRAINABLE]
    per_kind: dict[str, dict[str, int]] = {}
    for sample in samples:
        kind = sample.supervision_kind or "UNCLASSIFIED"
        entry = per_kind.setdefault(kind, {"records": 0, "trainable": 0, "targets": 0})
        entry["records"] += 1
        if sample.status in TRAINABLE:
            entry["trainable"] += 1
            entry["targets"] += len(sample.supervised)
    total = len(trainable)
    for kind, entry in per_kind.items():
        entry["share_of_trainable"] = round(entry["trainable"] / total, 6) if total else 0.0
        entry["kind"] = kind
    return {
        "trainable_records": total,
        "kinds": dict(sorted(per_kind.items())),
    }


def supervised_token_total(samples: list[SupervisedSample]) -> int:
    return sum(len(sample.supervised) for sample in samples)


def token_total(samples: list[SupervisedSample]) -> int:
    return sum(len(sample.tokens) for sample in samples)


def overflow_report(samples: list[SupervisedSample], max_seq_length: int) -> dict[str, Any]:
    """Evidence for the sequence-length policy: what was kept, cut, and dropped."""
    total = len(samples)
    counts = disposition_counts(samples)
    dropped = sum(count for status, count in counts.items() if status not in TRAINABLE)

    def fraction(value: int) -> float:
        return round(value / total, 6) if total else 0.0

    return {
        "schema_version": 1,
        "max_seq_length": max_seq_length,
        "records_considered": total,
        "dispositions": counts,
        "trainable_records": total - dropped,
        "dropped_records": dropped,
        "dropped_fraction": fraction(dropped),
        "tail_truncated_records": counts.get(STATUS_CONTEXT_TAIL_TRUNCATED, 0),
        "target_truncated_records": counts.get(STATUS_TARGET_TRUNCATED, 0),
        "no_assistant_turn_records": counts.get(STATUS_NO_ASSISTANT_TURN, 0),
        "unrenderable_records": counts.get(STATUS_UNRENDERABLE, 0),
        "unmaskable_records": counts.get(STATUS_UNMASKABLE, 0),
        "supervised_tokens": supervised_token_total(samples),
        "total_tokens": token_total(samples),
        "policy": (
            "keep the whole sequence when it fits; keep the first max_seq_length tokens when "
            "every supervised token survives (trailing non-target context is dropped); drop "
            "the record when any supervised token would be cut, because partially removed "
            "assistant supervision is corrupted supervision. Unrenderable and unmaskable "
            "records are quarantined, never repaired."
        ),
    }
