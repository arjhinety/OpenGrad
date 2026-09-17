"""What the Study 002 prose decision classifier may read: contract ``prose-decision-input-v1``.

This module holds **no classification logic**. It answers two questions and nothing else:

1. :func:`eligibility` -- *is this normalization-v3 record structurally eligible to be classified by the
   prose decision classifier?* Yes, or no with explicit reasons.
2. :func:`build_classifier_input` -- for an eligible record, the typed, versioned object the classifier
   receives, with deterministic serialization.

The contract (docs/research/study-002/32-CLASSIFIER-INPUT-CONTRACT.md) in brief:

* **Visible features:** the user message, the final assistant response, the tool definitions presented
  with the record, and a structural flag for whether a structured tool call exists.
* **Never features:** source dataset name, sampling stratum, P-DET or model reference labels, held-out
  labels, future evaluation outcomes. Record identity and source live in :class:`ClassifierProvenance`,
  which is serialized separately and is not part of the feature hash.
* **Structured calls** go to structural CALL routing (layer A), never to the prose classifier.
* **Post-tool-result responses** are excluded; a later, separately preregistered classifier may handle
  that stage.
* **Multi-turn records** are excluded. No multi-turn conversation is reduced to its final message.

Contract ``prose-decision-input-v2`` (docs/research/study-002/36 §2) is added alongside, not in place of, v1:
:func:`first_reply_eligibility` and :func:`build_first_reply_input` admit the **first assistant reply** of any
record, whatever follows it, with the same four features. Later turns are never read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengrad.data import versions
from opengrad.data.canonical import ToolConversation, stable_json
from opengrad.data.semantic import validate_training_trajectory

CONTRACT_VERSION = versions.CLASSIFIER_INPUT_CONTRACT_VERSION

# ── exclusion reasons, in precedence order (the first that applies is the primary reason) ──────────

EVALUATION_ONLY_OR_HELDOUT = "EVALUATION_ONLY_OR_HELDOUT"
MALFORMED_OR_UNRENDERABLE = "MALFORMED_OR_UNRENDERABLE"
STRUCTURAL_CALL = "STRUCTURAL_CALL"
POST_TOOL_RESULT_RESPONSE = "POST_TOOL_RESULT_RESPONSE"
MULTI_TURN_UNSUPPORTED = "MULTI_TURN_UNSUPPORTED"

EXCLUSION_ORDER = (
    EVALUATION_ONLY_OR_HELDOUT,
    MALFORMED_OR_UNRENDERABLE,
    STRUCTURAL_CALL,
    POST_TOOL_RESULT_RESPONSE,
    MULTI_TURN_UNSUPPORTED,
)

#: Splits that are evaluation material by name (``adapt_when2call`` refuses the same set).
EVALUATION_SPLITS = frozenset(
    {"mcq", "test", "llm_judge", "mcq_test", "llm_judge_test", "preference"}
)

#: The feature fields, in the order the contract lists them. Nothing else is a feature.
FEATURE_FIELDS = ("user_message", "assistant_response", "tools", "structured_call_present")

#: Held-out inputs, pinned. ``sha256`` is over the file bytes (parquet) or LF-normalized text (JSON).
HELDOUT_PROMPT_FILES = (
    (
        ".cache/normalization/raw/when2call_test_mcq.parquet",
        "17cdf1a07dad0bb5bb2a6c5663b7ba547e64c681add4a50151622788595d028f",
    ),
    (
        ".cache/normalization/raw/when2call_test_llm_judge.parquet",
        "2b613123d6643394a1905bc4f5b2b92a25c8acd97806312e19a4474615a9a853",
    ),
)
HELDOUT_PARTITION = (
    "reports/evaluation/behavioral-heldout-v2-partition.json",
    "56c6abc2e24c524b45c382f42927fa02117b2a04c02f04e89e6a40d8b520f901",
)
HELDOUT_QUARANTINES = (
    (
        "reports/evaluation/behavioral-heldout-v2-quarantine.json",
        "705b9ef68185c3ce7a997e9a9f728633b9f042b2304a96635dea34f891855cf4",
    ),
    (
        "reports/evaluation/behavioral-heldout-v2-quarantine--toolpolicy-canonical-v2-final.json",
        "34173706d01823aa44a2871e72ce278ede6f6c93df6f4009c6270fbfb1b0a2cd",
    ),
)


class ContractViolation(ValueError):
    """The input is not a normalization-v3 record, so the contract does not apply to it at all."""


class IneligibleRecord(ValueError):
    """A classifier input was requested for a record the contract excludes."""

    def __init__(self, result: Eligibility) -> None:
        self.result = result
        super().__init__(
            f"record is not eligible for prose classification: {', '.join(result.reasons)}"
        )


def normalize_prompt(text: str) -> str:
    """Case-folded, whitespace-collapsed text: the P-DET-v1 prompt-matching rule."""
    return " ".join(str(text).split()).casefold()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── held-out index ─────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HeldoutIndex:
    """Evaluation material a training record must not match: prompts (normalized) and record ids."""

    prompts: frozenset[str] = frozenset()
    record_ids: frozenset[str] = frozenset()
    inputs: tuple[tuple[str, str], ...] = ()

    @classmethod
    def load(cls, root: Path) -> HeldoutIndex:
        """Build the index from the pinned inputs, refusing any whose bytes have changed."""
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        prompts: set[str] = set()
        record_ids: set[str] = set()
        inputs: list[tuple[str, str]] = []
        for relative, expected in HELDOUT_PROMPT_FILES:
            path = root / relative
            observed = _sha256_bytes(path.read_bytes())
            if observed != expected:
                raise ContractViolation(f"held-out input changed: {relative}")
            prompts.update(
                normalize_prompt(q)
                for q in pq.read_table(path, columns=["question"]).column(0).to_pylist()
            )
            inputs.append((relative, observed))
        for relative, expected in (HELDOUT_PARTITION, *HELDOUT_QUARANTINES):
            data = (root / relative).read_bytes().replace(b"\r\n", b"\n")
            if _sha256_bytes(data) != expected:
                raise ContractViolation(f"held-out input changed: {relative}")
            payload = json.loads(data)
            if "example_ids" in payload:
                for side in sorted(payload["example_ids"]):
                    record_ids.update(str(item) for item in payload["example_ids"][side])
            for item in payload.get("excluded", []):
                record_ids.add(str(item["example_id"]))
            inputs.append((relative, expected))
        return cls(frozenset(prompts), frozenset(record_ids), tuple(inputs))


# ── eligibility ────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reasons: tuple[str, ...]
    details: tuple[str, ...] = ()

    @property
    def primary_reason(self) -> str | None:
        return self.reasons[0] if self.reasons else None


def _require_v3(record: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ContractViolation("record has no metadata object")
    if metadata.get("normalization_version") != versions.NORMALIZATION_VERSION:
        raise ContractViolation(
            f"record is not {versions.NORMALIZATION_VERSION}: {metadata.get('normalization_version')!r}"
        )
    if metadata.get("adapter_version") != versions.ADAPTER_VERSION:
        raise ContractViolation(
            f"record adapter_version {metadata.get('adapter_version')!r} is not authoritative"
        )
    if "behavior" in metadata:
        raise ContractViolation(
            "record carries a behaviour label; the classifier input must be pre-classifier"
        )
    if not isinstance(record.get("messages"), list) or not isinstance(record.get("tools"), list):
        raise ContractViolation("record messages and tools must be lists")
    return metadata


def _body_roles(messages: list[Any]) -> list[str]:
    roles = [
        str(message.get("role")) if isinstance(message, Mapping) else "?" for message in messages
    ]
    return roles[1:] if roles[:1] == ["system"] else roles


def _malformations(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[str]:
    messages = record["messages"]
    problems: list[str] = []
    if metadata.get("parse_status") != "VALID":
        problems.append(f"parse_status:{metadata.get('parse_status')}")
    if not messages:
        return [*problems, "no_messages"]
    if any(not isinstance(message, Mapping) for message in messages):
        return [*problems, "message_not_object"]
    for message in messages:
        if message.get("role") in {"system", "user"} and not isinstance(
            message.get("content"), str
        ):
            problems.append(f"{message.get('role')}_content_not_text")
    if not any(message.get("role") == "user" for message in messages):
        problems.append("no_user_turn")
    final = messages[-1]
    if final.get("role") != "assistant":
        problems.append(f"final_role:{final.get('role')}")
    elif not final.get("tool_calls") and not (
        isinstance(final.get("content"), str) and final["content"].strip()
    ):
        problems.append("final_assistant_empty")
    try:
        conversation = ToolConversation(
            str(record.get("id")),
            str(record.get("source")),
            list(record["tools"]),
            list(messages),
            dict(metadata),
        )
        # The renderer's own precondition (Qwen35_2BRenderer.render_sft calls the same trajectory gate).
        problems.extend(
            f"trajectory:{issue.code}" for issue in validate_training_trajectory(conversation)
        )
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        problems.append(f"unvalidatable:{type(exc).__name__}")
    return problems


def _heldout_hits(
    record: Mapping[str, Any], metadata: Mapping[str, Any], heldout: HeldoutIndex
) -> list[str]:
    hits: list[str] = []
    if metadata.get("eligibility") == "evaluation_only":
        hits.append("eligibility:evaluation_only")
    if str(metadata.get("split")) in EVALUATION_SPLITS:
        hits.append(f"split:{metadata.get('split')}")
    raw_source = metadata.get("source")
    source: Mapping[str, Any] = raw_source if isinstance(raw_source, Mapping) else {}
    for identity in {str(record.get("id")), str(source.get("upstream_id"))}:
        if identity in heldout.record_ids:
            hits.append("heldout_record_id")
    for message in record["messages"]:
        if (
            isinstance(message, Mapping)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and normalize_prompt(message["content"]) in heldout.prompts
        ):
            hits.append("heldout_prompt")
            break
    return hits


def eligibility(record: Mapping[str, Any], heldout: HeldoutIndex) -> Eligibility:
    """Is this record structurally eligible for the prose decision classifier?

    Deterministic and structural: it reads roles, the presence of structured calls and tool results,
    emptiness, trajectory validity and held-out membership. It never reads what the response *says*.
    Every applicable reason is returned, in :data:`EXCLUSION_ORDER`.
    """
    metadata = _require_v3(record)
    messages = record["messages"]
    reasons: list[str] = []
    details: list[str] = []
    hits = _heldout_hits(record, metadata, heldout)
    if hits:
        reasons.append(EVALUATION_ONLY_OR_HELDOUT)
        details.extend(hits)
    malformed = _malformations(record, metadata)
    if malformed:
        reasons.append(MALFORMED_OR_UNRENDERABLE)
        details.extend(malformed)
    objects = [message for message in messages if isinstance(message, Mapping)]
    if any(message.get("role") == "assistant" and message.get("tool_calls") for message in objects):
        reasons.append(STRUCTURAL_CALL)
    if any(message.get("role") == "tool" for message in objects):
        reasons.append(POST_TOOL_RESULT_RESPONSE)
    # Multi-turn: more than one user turn, or a tool-free conversation that is not exactly one user turn
    # followed by one assistant turn. A single user turn followed by a tool loop is already excluded by
    # its tool use and is not also called multi-turn.
    body = _body_roles(messages)
    tool_use = STRUCTURAL_CALL in reasons or POST_TOOL_RESULT_RESPONSE in reasons
    if body.count("user") != 1 or (not tool_use and body != ["user", "assistant"]):
        reasons.append(MULTI_TURN_UNSUPPORTED)
        details.append("shape:" + "-".join(role[:1].upper() for role in body))
    return Eligibility(not reasons, tuple(reasons), tuple(details))


# ── the classifier input object ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassifierFeatures:
    """Everything the classifier may read. Nothing identifies the record or its source."""

    user_message: str
    assistant_response: str
    tools: tuple[Mapping[str, Any], ...]
    structured_call_present: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_message": self.user_message,
            "assistant_response": self.assistant_response,
            "tools": [dict(tool) for tool in self.tools],
            "structured_call_present": self.structured_call_present,
        }


@dataclass(frozen=True)
class ClassifierProvenance:
    """Identity and provenance, kept apart from the features and never hashed with them."""

    record_key: str
    record_id: str
    source_name: str
    dataset_id: str
    raw_record_hash: str
    canonical_hash: str
    normalization_version: str
    adapter_key: str
    adapter_version: str
    schema_normalization_version: str

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ClassifierInput:
    features: ClassifierFeatures
    provenance: ClassifierProvenance
    contract_version: str = field(default=CONTRACT_VERSION)

    def features_payload(self) -> dict[str, Any]:
        return {"contract_version": self.contract_version, **self.features.as_dict()}

    def features_json(self) -> str:
        """Canonical serialization: sorted keys, no insignificant whitespace, UTF-8 text as-is."""
        return stable_json(self.features_payload())

    def features_sha256(self) -> str:
        return _sha256_bytes(self.features_json().encode("utf-8"))

    def provenance_json(self) -> str:
        return stable_json({"contract_version": self.contract_version, **self.provenance.as_dict()})


def build_classifier_input(record: Mapping[str, Any], heldout: HeldoutIndex) -> ClassifierInput:
    """The classifier input for an eligible record; :class:`IneligibleRecord` for any other."""
    result = eligibility(record, heldout)
    if not result.eligible:
        raise IneligibleRecord(result)
    metadata = record["metadata"]
    messages = record["messages"]
    user, assistant = [message for message in messages if message.get("role") != "system"]
    source = metadata["source"]
    features = ClassifierFeatures(
        user_message=user["content"],
        assistant_response=assistant["content"],
        tools=tuple(json.loads(stable_json(tool)) for tool in record["tools"]),
        structured_call_present=bool(assistant.get("tool_calls")),
    )
    provenance = ClassifierProvenance(
        record_key=f"{source['source_name']}:{metadata['raw_record_hash']}",
        record_id=str(record["id"]),
        source_name=str(source["source_name"]),
        dataset_id=str(source["dataset_id"]),
        raw_record_hash=str(metadata["raw_record_hash"]),
        canonical_hash=str(record["canonical_hash"]),
        normalization_version=str(metadata["normalization_version"]),
        adapter_key=str(metadata["adapter_key"]),
        adapter_version=str(metadata["adapter_version"]),
        schema_normalization_version=str(metadata["schema_normalization_version"]),
    )
    return ClassifierInput(features, provenance)


# ── contract v2: the first reply (docs/research/study-002/36 §2) ────────────────────────────────────
#
# The unit is the first assistant reply of a record: the first assistant turn, directly after exactly one
# user turn (a leading system message allowed), with text and no structured call, whatever follows it. A v1
# single exchange is the special case with nothing after it. The features are v1's four fields; nothing after
# the first reply is ever read, and whether the record continues is provenance only, because it would
# identify the source.

CONTRACT_V2_VERSION = versions.CLASSIFIER_INPUT_CONTRACT_V2_VERSION

FIRST_TURN_NOT_ONE_USER_THEN_ASSISTANT = "FIRST_TURN_NOT_ONE_USER_THEN_ASSISTANT"
FIRST_REPLY_IS_STRUCTURAL_CALL = "FIRST_REPLY_IS_STRUCTURAL_CALL"
FIRST_REPLY_EMPTY = "FIRST_REPLY_EMPTY"

FIRST_REPLY_EXCLUSION_ORDER = (
    EVALUATION_ONLY_OR_HELDOUT,
    MALFORMED_OR_UNRENDERABLE,
    FIRST_TURN_NOT_ONE_USER_THEN_ASSISTANT,
    FIRST_REPLY_IS_STRUCTURAL_CALL,
    FIRST_REPLY_EMPTY,
)

#: v1 malformation checks about the record's *final* message, which a first reply does not depend on.
FINAL_MESSAGE_CHECKS = ("final_role:", "final_assistant_empty")

UNIT_SINGLE_EXCHANGE = "single_exchange"
UNIT_HAS_CONTINUATION = "has_continuation"


def _first_reply_body(messages: list[Any]) -> list[Any]:
    leading_system = bool(messages) and isinstance(messages[0], Mapping) and messages[0].get("role") == "system"
    return messages[1:] if leading_system else messages


def first_reply_eligibility(record: Mapping[str, Any], heldout: HeldoutIndex) -> Eligibility:
    """Is this record's first assistant reply eligible under ``prose-decision-input-v2``?

    Structural, like :func:`eligibility`: held-out membership over every turn, v1's malformation checks over the
    whole record except the two about the final message, then the shape of the first exchange. It never reads
    what a response says.
    """
    metadata = _require_v3(record)
    reasons: list[str] = []
    details: list[str] = []
    hits = _heldout_hits(record, metadata, heldout)
    if hits:
        reasons.append(EVALUATION_ONLY_OR_HELDOUT)
        details.extend(hits)
    malformed = [
        problem for problem in _malformations(record, metadata) if not problem.startswith(FINAL_MESSAGE_CHECKS)
    ]
    if malformed:
        reasons.append(MALFORMED_OR_UNRENDERABLE)
        details.extend(malformed)
    body = _first_reply_body(record["messages"])
    head = [message.get("role") if isinstance(message, Mapping) else "?" for message in body[:2]]
    if head != ["user", "assistant"]:
        reasons.append(FIRST_TURN_NOT_ONE_USER_THEN_ASSISTANT)
        details.append("head:" + "-".join(str(role)[:1].upper() for role in head))
    else:
        reply = body[1]
        if reply.get("tool_calls"):
            reasons.append(FIRST_REPLY_IS_STRUCTURAL_CALL)
        elif not (isinstance(reply.get("content"), str) and reply["content"].strip()):
            reasons.append(FIRST_REPLY_EMPTY)
    return Eligibility(not reasons, tuple(reasons), tuple(details))


@dataclass(frozen=True)
class FirstReplyProvenance(ClassifierProvenance):
    """v1's provenance plus whether the record continues after the first reply. Never a feature."""

    unit_kind: str = UNIT_SINGLE_EXCHANGE


def build_first_reply_input(record: Mapping[str, Any], heldout: HeldoutIndex) -> ClassifierInput:
    """The ``prose-decision-input-v2`` input for an eligible first reply; :class:`IneligibleRecord` otherwise."""
    result = first_reply_eligibility(record, heldout)
    if not result.eligible:
        raise IneligibleRecord(result)
    metadata = record["metadata"]
    body = _first_reply_body(record["messages"])
    user, reply = body[0], body[1]
    source = metadata["source"]
    features = ClassifierFeatures(
        user_message=user["content"],
        assistant_response=reply["content"],
        tools=tuple(json.loads(stable_json(tool)) for tool in record["tools"]),
        structured_call_present=False,
    )
    provenance = FirstReplyProvenance(
        record_key=f"{source['source_name']}:{metadata['raw_record_hash']}",
        record_id=str(record["id"]),
        source_name=str(source["source_name"]),
        dataset_id=str(source["dataset_id"]),
        raw_record_hash=str(metadata["raw_record_hash"]),
        canonical_hash=str(record["canonical_hash"]),
        normalization_version=str(metadata["normalization_version"]),
        adapter_key=str(metadata["adapter_key"]),
        adapter_version=str(metadata["adapter_version"]),
        schema_normalization_version=str(metadata["schema_normalization_version"]),
        unit_kind=UNIT_SINGLE_EXCHANGE if len(body) == 2 else UNIT_HAS_CONTINUATION,
    )
    return ClassifierInput(features, provenance, contract_version=CONTRACT_V2_VERSION)
