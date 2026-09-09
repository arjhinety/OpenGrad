from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opengrad.data.canonical import ToolConversation
from opengrad.data.schema import normalize_tool


@dataclass(frozen=True)
class RenderedTrainingExample:
    example_id: str
    text: str
    model_id: str
    model_revision: str
    renderer: str
    tokenizer_revision: str
    chat_template_hash: str
    enable_thinking: bool


def _qwen_messages(example: ToolConversation) -> list[dict[str, Any]]:
    result = []
    for message in example.messages:
        item = {
            key: value
            for key, value in message.items()
            if key in {"role", "content", "tool_call_id", "name"}
        }
        calls = message.get("tool_calls", [])
        if calls:
            item["tool_calls"] = [
                {"name": c["name"], "arguments": c.get("arguments", {})} for c in calls
            ]
        result.append(item)
    return result


def _qwen_tools(example: ToolConversation) -> list[dict[str, Any]]:
    return [normalize_tool(tool) for tool in example.tools]


class Qwen35_2BRenderer:
    model_id = "qwen3.5-2b"
    hf_repo = "Qwen/Qwen3.5-2B"
    model_revision = "15852e8c16360a2fea060d615a32b45270f8a8fc"
    renderer_version = "qwen3_5_2b_v1"

    def __init__(
        self,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        enable_thinking: bool = False,
    ) -> None:
        self.model_revision = revision or self.model_revision
        self.cache_dir = cache_dir
        self.enable_thinking = enable_thinking
        self._tokenizer: Any = None

    def _load(self) -> Any:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "rendering requires the optional training dependency: transformers"
                ) from exc
            kwargs: dict[str, Any] = {"revision": self.model_revision, "trust_remote_code": False}
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir
            self._tokenizer = AutoTokenizer.from_pretrained(self.hf_repo, **kwargs)
        return self._tokenizer

    def render_sft(self, example: ToolConversation) -> RenderedTrainingExample:
        example.validate_training_semantics()
        tokenizer = self._load()
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": False,
            "enable_thinking": self.enable_thinking,
        }
        if example.tools:
            kwargs["tools"] = _qwen_tools(example)
        text = tokenizer.apply_chat_template(_qwen_messages(example), **kwargs)
        template = str(getattr(tokenizer, "chat_template", ""))
        template_hash = hashlib.sha256(template.encode()).hexdigest()
        return RenderedTrainingExample(
            example.id,
            str(text),
            self.hf_repo,
            self.model_revision,
            self.renderer_version,
            self.model_revision,
            template_hash,
            self.enable_thinking,
        )

    def render_evaluation(self, example: Any) -> RenderedTrainingExample:
        """Render an evaluation example with the same frozen Qwen contract.

        Evaluation examples intentionally do not reuse an SFT assistant answer.
        Only the question and declared tool catalogue are sent to the model.
        """
        if hasattr(example, "validate"):
            example.validate()
        tokenizer = self._load()
        messages = [{"role": "user", "content": str(example.question)}]
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": self.enable_thinking,
        }
        if example.tools:
            kwargs["tools"] = [normalize_tool(tool) for tool in example.tools]
        text = tokenizer.apply_chat_template(messages, **kwargs)
        template = str(getattr(tokenizer, "chat_template", ""))
        return RenderedTrainingExample(
            example.example_id,
            str(text),
            self.hf_repo,
            self.model_revision,
            self.renderer_version,
            self.model_revision,
            hashlib.sha256(template.encode()).hexdigest(),
            self.enable_thinking,
        )

    def token_lengths(self, example: ToolConversation) -> dict[str, int]:
        rendered = self.render_sft(example)
        tokenizer = self._load()
        return {
            "total": len(tokenizer(rendered.text, add_special_tokens=False)["input_ids"]),
            "system_tool_catalogue": len(
                tokenizer.apply_chat_template(
                    _qwen_messages(example),
                    tools=_qwen_tools(example),
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )
            if example.tools
            else 0,
        }

    def text_token_length(self, text: str) -> int:
        """Count prompt tokens without applying truncation."""
        tokenizer = self._load()
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def renderer_for(model: str) -> Qwen35_2BRenderer:
    if model in {"qwen3.5-2b", "Qwen/Qwen3.5-2B"}:
        return Qwen35_2BRenderer()
    raise ValueError(f"no exact renderer registered for {model}")


def render_materialized_corpus(
    input_dir: Path, output_path: Path, *, model: str = "Qwen/Qwen3.5-2B"
) -> dict[str, Any]:
    """Persist tokenizer-backed JSONL with canonical parent IDs and lineage."""
    from opengrad.data.materialize import iter_materialized_rows
    from opengrad.data.real_analysis import _restore_row

    renderer = renderer_for(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    digest = hashlib.sha256()
    records = 0
    failures: dict[str, int] = {}
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for raw in iter_materialized_rows(input_dir):
            row = _restore_row(raw)
            try:
                example = ToolConversation(
                    row["id"], row["source"], row["tools"], row["messages"], row["metadata"]
                )
                rendered = renderer.render_sft(example)
                item = {
                    "canonical_id": example.id,
                    "canonical_hash": row.get("canonical_hash"),
                    "text": rendered.text,
                    "text_sha256": hashlib.sha256(rendered.text.encode()).hexdigest(),
                    "model": rendered.model_id,
                    "model_revision": rendered.model_revision,
                    "tokenizer_revision": rendered.tokenizer_revision,
                    "template_hash": rendered.chat_template_hash,
                    "renderer": rendered.renderer,
                    "enable_thinking": rendered.enable_thinking,
                }
                line = (
                    json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                handle.write(line)
                digest.update(line.encode("utf-8"))
                records += 1
            except Exception as exc:  # noqa: BLE001 - quarantine renderer-specific failures
                key = f"{type(exc).__name__}: {exc}"
                failures[key] = failures.get(key, 0) + 1
    temporary.replace(output_path)
    return {
        "schema_version": 1,
        "status": "MEASURED_WITH_RENDER_FAILURES" if failures else "FULL_RENDERED",
        "model": renderer.hf_repo,
        "model_revision": renderer.model_revision,
        "renderer": renderer.renderer_version,
        "template_hash": hashlib.sha256(str(renderer._load().chat_template).encode()).hexdigest(),
        "canonical_parent": str(input_dir),
        "rendered_records": records,
        "render_failures": failures,
        "render_checksum": digest.hexdigest(),
        "output": str(output_path),
    }
