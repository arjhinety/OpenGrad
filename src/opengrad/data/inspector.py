"""Template and tokenization inspector for assistant loss-mask verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from opengrad.data.canonical import ToolConversation
from opengrad.data.renderers import Qwen35_2BRenderer


@dataclass
class InspectedTemplate:
    raw_conversation: list[dict[str, Any]]
    rendered_text: str
    token_ids: list[int]
    tokens: list[str]
    loss_mask: list[int]  # 1 = computed in loss, 0 = masked out
    statistics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_conversation": self.raw_conversation,
            "rendered_text": self.rendered_text,
            "token_ids": self.token_ids,
            "tokens": self.tokens,
            "loss_mask": self.loss_mask,
            "statistics": self.statistics,
        }

    def render_display(self, max_tokens: int = 60) -> str:
        lines = [
            "# Template & Loss Mask Inspection\n",
            "## Summary Statistics",
            f"- **Total Tokens:** {self.statistics.get('total_tokens', len(self.token_ids))}",
            f"- **Assistant / Trained Tokens:** {self.statistics.get('trained_tokens', 0)} ({self.statistics.get('trained_percentage', 0.0):.1f}%)",
            f"- **Masked Tokens (Prompt/System):** {self.statistics.get('masked_tokens', 0)}",
            "",
            "## Token Loss Mask Table (First tokens)",
            f"{'Index':<6} {'ID':<8} {'Mask':<6} {'Trained?':<10} {'Token String'}",
            "-" * 65,
        ]

        for i, (tid, tok, mask) in enumerate(
            zip(self.token_ids[:max_tokens], self.tokens[:max_tokens], self.loss_mask[:max_tokens])
        ):
            trained_str = "YES (LOSS)" if mask == 1 else "NO (MASK)"
            tok_repr = repr(tok)
            lines.append(f"{i:<6} {tid:<8} {mask:<6} {trained_str:<10} {tok_repr}")

        if len(self.token_ids) > max_tokens:
            lines.append(f"... [{len(self.token_ids) - max_tokens} remaining tokens truncated]")

        return "\n".join(lines)


def inspect_template(
    example: ToolConversation | dict[str, Any],
    model_revision: str = "15852e8c16360a2fea060d615a32b45270f8a8fc",
    enable_thinking: bool = False,
) -> InspectedTemplate:
    """Inspect conversation rendering, token IDs, and assistant-only loss mask."""
    if isinstance(example, dict):
        example_conv = ToolConversation(
            str(example.get("id", "inspect")),
            str(example.get("source", "unknown")),
            list(example.get("tools") or []),
            list(example.get("messages") or []),
            dict(example.get("metadata") or {}),
        )
    else:
        example_conv = example

    renderer = Qwen35_2BRenderer(revision=model_revision, enable_thinking=enable_thinking)
    rendered = renderer.render_sft(example_conv)
    rendered_text = rendered.text

    tokenizer = renderer._load()
    encoded = tokenizer(rendered_text, add_special_tokens=False)
    token_ids: list[int] = list(encoded["input_ids"])
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Compute assistant-only loss mask
    # For Qwen chat template:
    # <|im_start|>assistant ... <|im_end|>
    # Tokens inside assistant turns have loss_mask = 1, others have 0.
    loss_mask: list[int] = [0] * len(token_ids)

    in_assistant = False
    for i, tok in enumerate(tokens):
        if "<|im_start|>assistant" in tok or tok == "assistant":
            in_assistant = True
            loss_mask[i] = 0  # Do not train on assistant header itself
            continue
        if "<|im_end|>" in tok:
            if in_assistant:
                loss_mask[i] = 1  # Predict end of turn
                in_assistant = False
            continue
        if in_assistant:
            loss_mask[i] = 1

    trained_tokens = sum(loss_mask)
    masked_tokens = len(loss_mask) - trained_tokens
    pct = round(trained_tokens / max(1, len(loss_mask)) * 100.0, 2)

    stats = {
        "total_tokens": len(loss_mask),
        "trained_tokens": trained_tokens,
        "masked_tokens": masked_tokens,
        "trained_percentage": pct,
    }

    return InspectedTemplate(
        raw_conversation=example_conv.messages,
        rendered_text=rendered_text,
        token_ids=token_ids,
        tokens=tokens,
        loss_mask=loss_mask,
        statistics=stats,
    )
