"""Supervision contracts: what each canonical record actually supervises.

Why this module exists
---------------------
Not every legitimate post-training dataset has the same conversational trajectory shape.
xLAM/APIGen is `query + tools -> assistant tool_call`: the supervised objective is *predict the
next tool invocation*, and the corpus structurally contains no tool-result turn. A corpus that
claims to contain a full execution trajectory has a different objective and does require its
calls to be resolved.

Both are useful. Treating the first as an unresolved trajectory discards it; treating every
unresolved call as valid corrupts the second. So the distinction is made explicit here, as a
canonical data-model concept, rather than inferred from which messages happen to be missing at
training time.

The governing principle:

    Dataset heterogeneity is allowed. Semantic ambiguity is not.

What this module deliberately does not do
-----------------------------------------
* It does not weaken validation. A malformed call, an undeclared tool, invalid arguments, a
  duplicate call id, a mismatched result, or a message appearing before a required result are
  all still rejected, under every contract. The *only* thing a contract may change is whether a
  terminal call requires a future environment response.
* It does not fabricate turns. A `CALL_PREDICTION` record is rendered exactly as its messages
  are; no synthetic `tool_result` is inserted to make it look like a trajectory.
* It is not a source-name conditional. `if source == "xlam"` appears nowhere; the adapter states
  what upstream means, and the canonical record carries that meaning independently of the name.

Adding a contract
-----------------
Add a member to `SupervisionKind`, add its contract to `CONTRACTS`, add the required rules to
`validate_training_trajectory` if the new contract needs different trajectory handling, and
extend the tests. An unknown kind is never silently mapped onto an existing one: it is a
validation failure that quarantines the record.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

SUPERVISION_METADATA_KEY = "supervision"
CONTRACT_VERSION = "supervision_contract_v1"


class SupervisionKind(str, Enum):
    """The trajectory shape a record's supervised target belongs to.

    Deliberately small: the two kinds that the corpus actually contains. The enum and the
    contract registry are the extension point, so a future kind (direct-response-only,
    tool-selection-only, argument-generation, repair, preference-pair, multi-turn policy) is
    added by declaring it here rather than by inventing a string at a call site.
    """

    COMPLETE_TRAJECTORY = "COMPLETE_TRAJECTORY"
    CALL_PREDICTION = "CALL_PREDICTION"


class SupervisionAssignment(str, Enum):
    """How a record's kind was decided. Recorded so the decision is auditable.

    The distinction matters because supervision classification changes how an example is
    interpreted even when its text is byte-identical.
    """

    UPSTREAM_DECLARED = "upstream_declared"
    SOURCE_ADAPTER = "source_adapter"
    DERIVED = "derived"
    LEGACY_DEFAULT = "legacy_default"
    MANUAL = "manual"


@dataclass(frozen=True)
class SupervisionContract:
    """The rules one supervision kind is validated and rendered under.

    `validation_policy` is a version string recorded on every accepted record, so a future
    change to a contract's rules cannot be confused with the records judged under the old ones.
    """

    kind: SupervisionKind
    description: str
    terminal_target_type: str
    tool_result_required_after_terminal_call: bool
    allow_intermediate_calls: bool
    require_intermediate_results: bool
    supervised_message_roles: tuple[str, ...]
    validation_policy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "terminal_target_type": self.terminal_target_type,
            "tool_result_required_after_terminal_call": (
                self.tool_result_required_after_terminal_call
            ),
            "allow_intermediate_calls": self.allow_intermediate_calls,
            "require_intermediate_results": self.require_intermediate_results,
            "supervised_message_roles": list(self.supervised_message_roles),
            "validation_policy": self.validation_policy,
        }


CONTRACTS: dict[SupervisionKind, SupervisionContract] = {
    SupervisionKind.COMPLETE_TRAJECTORY: SupervisionContract(
        kind=SupervisionKind.COMPLETE_TRAJECTORY,
        description=(
            "A full execution trajectory: every tool call is answered by its result and the "
            "conversation ends in a terminal assistant response."
        ),
        terminal_target_type="assistant_final_response",
        tool_result_required_after_terminal_call=True,
        allow_intermediate_calls=True,
        require_intermediate_results=True,
        supervised_message_roles=("assistant",),
        validation_policy="complete_target_v1",
    ),
    SupervisionKind.CALL_PREDICTION: SupervisionContract(
        kind=SupervisionKind.CALL_PREDICTION,
        description=(
            "Next-call prediction: the terminal assistant tool call is itself the supervised "
            "target, so no tool result is required after it. Calls that are not terminal must "
            "still be resolved."
        ),
        terminal_target_type="assistant_tool_call",
        tool_result_required_after_terminal_call=False,
        allow_intermediate_calls=True,
        require_intermediate_results=True,
        supervised_message_roles=("assistant",),
        validation_policy="call_prediction_v1",
    ),
}

# Records predating supervision metadata were all validated as complete trajectories, so that is
# the only safe reading of them. This is a compatibility default for existing artifacts, not a
# fallback for unknown values: an explicitly declared kind that is not in `CONTRACTS` is an
# error, never this.
LEGACY_DEFAULT_KIND = SupervisionKind.COMPLETE_TRAJECTORY


def contract_for_kind(kind: SupervisionKind | str) -> SupervisionContract:
    """Look up a contract, rejecting anything undeclared.

    No coercion and no silent default: a caller that cannot supply a declared kind must fail,
    because guessing one would reinterpret the record.
    """
    if isinstance(kind, str):
        try:
            kind = SupervisionKind(kind)
        except ValueError as exc:
            declared = ", ".join(sorted(member.value for member in SupervisionKind))
            raise ValueError(
                f"unknown supervision kind {kind!r}; declared kinds are: {declared}"
            ) from exc
    try:
        return CONTRACTS[kind]
    except KeyError as exc:  # a kind added to the enum but not to CONTRACTS
        raise ValueError(f"no contract declared for supervision kind {kind.value!r}") from exc


def declared_kinds() -> frozenset[str]:
    return frozenset(kind.value for kind in SupervisionKind)


def supervision_block(
    kind: SupervisionKind,
    *,
    assignment: SupervisionAssignment,
    adapter: str,
    adapter_version: str,
    note: str = "",
) -> dict[str, Any]:
    """Build the canonical `metadata.supervision` block for one record."""
    contract = contract_for_kind(kind)
    return {
        "kind": kind.value,
        "assignment": assignment.value,
        "adapter": adapter,
        "adapter_version": adapter_version,
        "contract_version": CONTRACT_VERSION,
        "validation_policy": contract.validation_policy,
        "note": note,
    }


def validate_supervision_block(metadata: dict[str, Any], *, require: bool = False) -> dict[str, Any]:
    """Validate `metadata.supervision`, returning the block.

    ``require`` is used at the training boundary, where an undeclared kind means the record's
    target was never classified and must not be trained as though it had been. The generic
    canonical validator does not require it, because Canonical-v1 records predate the field.
    """
    block = metadata.get(SUPERVISION_METADATA_KEY)
    if block is None:
        if require:
            raise ValueError(
                "supervision metadata is required at the training boundary; "
                "the record declares no supervision contract"
            )
        return {}
    if not isinstance(block, dict):
        raise TypeError("metadata.supervision must be an object")
    kind = block.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError("metadata.supervision.kind is required")
    # Rejects an undeclared kind with the declared set in the message, rather than coercing.
    contract = contract_for_kind(kind)
    if block.get("contract_version") not in {None, CONTRACT_VERSION}:
        raise ValueError(
            f"unsupported supervision contract version: {block.get('contract_version')!r}"
        )
    if block.get("validation_policy") not in {None, contract.validation_policy}:
        raise ValueError(
            f"supervision validation_policy {block.get('validation_policy')!r} does not match "
            f"the contract for {kind!r} ({contract.validation_policy!r})"
        )
    assignment = block.get("assignment")
    if assignment is not None and assignment not in {
        member.value for member in SupervisionAssignment
    }:
        raise ValueError(f"unknown supervision assignment: {assignment!r}")
    return block


def resolve_contract(metadata: dict[str, Any]) -> tuple[SupervisionContract, str]:
    """Return the contract governing a record, plus how it was determined.

    Absence of an explicit kind is read as `LEGACY_DEFAULT_KIND` and reported as
    ``"legacy_default"`` so a caller can tell inheritance from declaration.
    """
    block = validate_supervision_block(metadata)
    if not block:
        return contract_for_kind(LEGACY_DEFAULT_KIND), SupervisionAssignment.LEGACY_DEFAULT.value
    return contract_for_kind(block["kind"]), str(
        block.get("assignment") or SupervisionAssignment.DERIVED.value
    )
