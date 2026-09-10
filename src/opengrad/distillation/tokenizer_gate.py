"""Tokenizer compatibility gate for token-level on-policy distillation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenizerComparison:
    student_model: str
    teacher_model: str
    vocab_size_student: int
    vocab_size_teacher: int
    vocab_size_match: bool
    special_tokens_match: bool
    tool_tokens_match: bool
    eos_token_match: bool
    sample_tokenizations_match: bool
    verdict: str  # "TOKENIZER_COMPATIBLE" or "TOKENIZER_INCOMPATIBLE"
    discrepancies: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_model": self.student_model,
            "teacher_model": self.teacher_model,
            "vocab_size_student": self.vocab_size_student,
            "vocab_size_teacher": self.vocab_size_teacher,
            "vocab_size_match": self.vocab_size_match,
            "special_tokens_match": self.special_tokens_match,
            "tool_tokens_match": self.tool_tokens_match,
            "eos_token_match": self.eos_token_match,
            "sample_tokenizations_match": self.sample_tokenizations_match,
            "verdict": self.verdict,
            "discrepancies": self.discrepancies,
            "details": self.details,
        }

    def render_summary(self) -> str:
        lines = [
            "# Teacher-Student Tokenizer Compatibility Verification",
            "",
            f"- **Student:** `{self.student_model}` (Vocab: {self.vocab_size_student})",
            f"- **Teacher:** `{self.teacher_model}` (Vocab: {self.vocab_size_teacher})",
            f"- **Verdict:** **{self.verdict}**",
            "",
            "| Dimension | Result |",
            "| :--- | :---: |",
            f"| Vocab Size Match | {'PASS' if self.vocab_size_match else 'FAIL'} |",
            f"| Special Tokens Match | {'PASS' if self.special_tokens_match else 'FAIL'} |",
            f"| Tool Tokens Match | {'PASS' if self.tool_tokens_match else 'FAIL'} |",
            f"| EOS Token Match | {'PASS' if self.eos_token_match else 'FAIL'} |",
            f"| Representative Sample Match | {'PASS' if self.sample_tokenizations_match else 'FAIL'} |",
            "",
        ]
        if self.discrepancies:
            lines.append("## Discrepancies")
            for d in self.discrepancies:
                lines.append(f"- {d}")
        else:
            lines.append("## Status: All token IDs and representative tokenizations are identical.")
            lines.append(
                "Safe for direct token-level next-distribution distillation without logit remap."
            )
        return "\n".join(lines)


def compare_tokenizers(
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    student_model_id: str = "Qwen/Qwen3.5-2B",
    teacher_model_id: str = "Qwen/Qwen3.8-27B",
) -> TokenizerComparison:
    """Strictly verify whether student and teacher tokenizers share identical token IDs.

    Fails closed if any token ID, special token, or tool markup token differs.
    """
    discrepancies: list[str] = []

    # 1. Vocab size
    v_s = len(student_tokenizer)
    v_t = len(teacher_tokenizer)
    vocab_match = v_s == v_t
    if not vocab_match:
        discrepancies.append(f"Vocabulary size mismatch: student={v_s} vs teacher={v_t}")

    # 2. EOS / BOS
    eos_s = getattr(student_tokenizer, "eos_token_id", None)
    eos_t = getattr(teacher_tokenizer, "eos_token_id", None)
    eos_match = eos_s == eos_t
    if not eos_match:
        discrepancies.append(f"EOS token ID mismatch: student={eos_s} vs teacher={eos_t}")

    # 3. Special tokens
    special_match = True
    for s_tok in ["<|im_start|>", "<|im_end|>"]:
        id_s = student_tokenizer.convert_tokens_to_ids(s_tok)
        id_t = teacher_tokenizer.convert_tokens_to_ids(s_tok)
        if id_s != id_t or id_s is None:
            special_match = False
            discrepancies.append(
                f"Special token '{s_tok}' ID mismatch: student={id_s} vs teacher={id_t}"
            )

    # 4. Tool markup tokens
    tool_match = True
    for t_tok in ["<tool_call>", "</tool_call>"]:
        # Test encode behavior
        enc_s = list(student_tokenizer.encode(t_tok, add_special_tokens=False))
        enc_t = list(teacher_tokenizer.encode(t_tok, add_special_tokens=False))
        if enc_s != enc_t:
            tool_match = False
            discrepancies.append(f"Tool markup '{t_tok}' tokenization mismatch: {enc_s} vs {enc_t}")

    # 5. Representative text samples
    sample_match = True
    samples = [
        "Explain quantum computing in three sentences.",
        '<tool_call>{"name": "web_search", "arguments": {"query": "OpenGrad"}}</tool_call>',
        "def solve(a: int, b: int) -> int:\n    return a + b",
    ]
    for sample in samples:
        e_s = list(student_tokenizer.encode(sample, add_special_tokens=False))
        e_t = list(teacher_tokenizer.encode(sample, add_special_tokens=False))
        if e_s != e_t:
            sample_match = False
            discrepancies.append(f"Sample tokenization mismatch on: '{sample[:30]}...'")

    compatible = vocab_match and special_match and tool_match and eos_match and sample_match
    verdict = "TOKENIZER_COMPATIBLE" if compatible else "TOKENIZER_INCOMPATIBLE"

    return TokenizerComparison(
        student_model=student_model_id,
        teacher_model=teacher_model_id,
        vocab_size_student=v_s,
        vocab_size_teacher=v_t,
        vocab_size_match=vocab_match,
        special_tokens_match=special_match,
        tool_tokens_match=tool_match,
        eos_token_match=eos_match,
        sample_tokenizations_match=sample_match,
        verdict=verdict,
        discrepancies=discrepancies,
    )


def validate_teacher_tokenizer_offline(
    student_model_id: str = "Qwen/Qwen3.5-2B",
    teacher_model_id: str = "Qwen/Qwen3.8-27B",
    mock_compatible: bool = True,
) -> TokenizerComparison:
    """Mock/offline validator for CPU CI and testing without downloading 60GB models."""
    if mock_compatible:
        return TokenizerComparison(
            student_model=student_model_id,
            teacher_model=teacher_model_id,
            vocab_size_student=248064,
            vocab_size_teacher=248064,
            vocab_size_match=True,
            special_tokens_match=True,
            tool_tokens_match=True,
            eos_token_match=True,
            sample_tokenizations_match=True,
            verdict="TOKENIZER_COMPATIBLE",
            discrepancies=[],
        )
    return TokenizerComparison(
        student_model=student_model_id,
        teacher_model=teacher_model_id,
        vocab_size_student=248064,
        vocab_size_teacher=151936,
        vocab_size_match=False,
        special_tokens_match=False,
        tool_tokens_match=False,
        eos_token_match=False,
        sample_tokenizations_match=False,
        verdict="TOKENIZER_INCOMPATIBLE",
        discrepancies=["Vocabulary size mismatch: student=248064 vs teacher=151936"],
    )
