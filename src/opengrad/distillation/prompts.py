"""Prompt-state dataset extraction from canonical tool trajectories for on-policy distillation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengrad.data.canonical import ToolConversation


@dataclass
class PromptState:
    prompt_state_id: str
    canonical_id: str
    source_dataset: str
    source_revision: str
    turn_index: int
    behavior_category: str
    system_prompt: str
    tools: list[dict[str, Any]]
    conversation_prefix: list[dict[str, Any]]
    target_action_type: str  # "CALL" or "ANSWER"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_state_id": self.prompt_state_id,
            "canonical_id": self.canonical_id,
            "source_dataset": self.source_dataset,
            "source_revision": self.source_revision,
            "turn_index": self.turn_index,
            "behavior_category": self.behavior_category,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "conversation_prefix": self.conversation_prefix,
            "target_action_type": self.target_action_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptState:
        return cls(
            prompt_state_id=str(data["prompt_state_id"]),
            canonical_id=str(data["canonical_id"]),
            source_dataset=str(data["source_dataset"]),
            source_revision=str(data.get("source_revision", "unknown")),
            turn_index=int(data.get("turn_index", 0)),
            behavior_category=str(data.get("behavior_category", "tool_policy")),
            system_prompt=str(data.get("system_prompt", "")),
            tools=list(data.get("tools") or []),
            conversation_prefix=list(data.get("conversation_prefix") or []),
            target_action_type=str(data.get("target_action_type", "CALL")),
            metadata=dict(data.get("metadata") or {}),
        )


def extract_prompt_states(
    conversations: Sequence[ToolConversation | dict[str, Any]],
    output_file: Path | None = None,
    profile: str = "broad",  # "broad" or "residual"
    held_out_ids: set[str] | None = None,
) -> list[PromptState]:
    """Extract valid prompt states immediately preceding an assistant action decision.

    Firewalled from held-out evaluation examples and quarantined rows (Section 21).
    """
    excluded = held_out_ids or set()
    prompt_states: list[PromptState] = []

    for conv in conversations:
        c_id = conv.id if isinstance(conv, ToolConversation) else str(conv.get("id", "c"))
        if c_id in excluded:
            continue

        c_source = (
            conv.source if isinstance(conv, ToolConversation) else conv.get("source", "unknown")
        )
        c_source_id = (
            c_source.get("dataset_id", "source") if isinstance(c_source, dict) else str(c_source)
        )
        c_rev = c_source.get("revision", "pinned") if isinstance(c_source, dict) else "pinned"
        c_tools = conv.tools if isinstance(conv, ToolConversation) else conv.get("tools", [])
        c_messages = (
            conv.messages if isinstance(conv, ToolConversation) else conv.get("messages", [])
        )
        c_meta = conv.metadata if isinstance(conv, ToolConversation) else conv.get("metadata", {})

        # Walk conversation prefix to find assistant turns
        prefix: list[dict[str, Any]] = []
        system_text = ""

        for turn_idx, msg in enumerate(c_messages):
            role = msg.get("role", "")
            if role == "system":
                system_text = str(msg.get("content", ""))
                prefix.append(msg)
                continue

            if role == "assistant":
                # Create prompt state for the assistant's decision at this point
                has_calls = bool(msg.get("tool_calls"))
                target_type = "CALL" if has_calls else "ANSWER"

                # If residual profile, prioritize tool calls or multi-turn decisions
                if profile == "residual" and target_type != "CALL" and turn_idx <= 1:
                    prefix.append(msg)
                    continue

                p_state = PromptState(
                    prompt_state_id=f"{c_id}_turn_{turn_idx}",
                    canonical_id=c_id,
                    source_dataset=c_source_id,
                    source_revision=c_rev,
                    turn_index=turn_idx,
                    behavior_category=c_meta.get("behavior_category", "tool_policy"),
                    system_prompt=system_text,
                    tools=c_tools,
                    conversation_prefix=list(prefix),
                    target_action_type=target_type,
                    metadata={"profile": profile, "turn": turn_idx},
                )
                prompt_states.append(p_state)

            prefix.append(msg)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            for ps in prompt_states:
                f.write(json.dumps(ps.to_dict(), ensure_ascii=False) + "\n")

    return prompt_states
