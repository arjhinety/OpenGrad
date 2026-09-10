from pathlib import Path
from typing import Any

from opengrad.benchmarks.backends.mock import DeterministicFakeBackend
from opengrad.distillation.evaluator import (
    TeacherAdvantageEvaluator,
    check_distillation_memory_safety,
)
from opengrad.distillation.prompts import extract_prompt_states
from opengrad.distillation.rollouts import OnPolicyRolloutGenerator
from opengrad.distillation.tokenizer_gate import (
    validate_teacher_tokenizer_offline,
)


def test_tokenizer_compatibility_offline() -> None:
    comp_ok = validate_teacher_tokenizer_offline(
        "Qwen/Qwen3.5-2B", "Qwen/Qwen3.8-27B", mock_compatible=True
    )
    assert comp_ok.verdict == "TOKENIZER_COMPATIBLE"
    assert comp_ok.vocab_size_match is True

    comp_fail = validate_teacher_tokenizer_offline(
        "Qwen/Qwen3.5-2B", "Llama-3.2-1B", mock_compatible=False
    )
    assert comp_fail.verdict == "TOKENIZER_INCOMPATIBLE"
    assert len(comp_fail.discrepancies) > 0


def test_extract_prompt_states_and_held_out_firewall(tmp_path: Path) -> None:
    conversations: list[dict[str, Any]] = [
        {
            "id": "allowed_conv_01",
            "source": {"dataset_id": "xlam", "revision": "pinned"},
            "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
            "messages": [
                {"role": "user", "content": "Find order 100"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "lookup", "arguments": {}}],
                },
            ],
            "metadata": {"behavior_category": "tool_call"},
        },
        {
            "id": "held_out_conv_02",  # Should be filtered out by firewall
            "source": {"dataset_id": "when2call-mcq"},
            "tools": [],
            "messages": [
                {"role": "user", "content": "Held out question"},
                {"role": "assistant", "content": "Answer"},
            ],
            "metadata": {},
        },
    ]

    out_file = tmp_path / "prompt_states.jsonl"
    held_out = {"held_out_conv_02"}
    states = extract_prompt_states(conversations, output_file=out_file, held_out_ids=held_out)

    assert len(states) == 1
    assert states[0].canonical_id == "allowed_conv_01"
    assert states[0].target_action_type == "CALL"
    assert out_file.exists()


def test_on_policy_rollout_generator(tmp_path: Path) -> None:
    backend = DeterministicFakeBackend()
    generator = OnPolicyRolloutGenerator(backend)

    sample_states = extract_prompt_states(
        [
            {
                "id": "demo",
                "source": "canonical",
                "tools": [{"name": "lookup"}],
                "messages": [
                    {"role": "user", "content": "Search x"},
                    {"role": "assistant", "content": "", "tool_calls": [{"name": "lookup"}]},
                ],
                "metadata": {},
            }
        ]
    )

    out_file = tmp_path / "rollouts.jsonl"
    rollouts = generator.generate(sample_states, "exp_test", "student_ckpt_1", output_file=out_file)
    assert len(rollouts) == 1
    assert rollouts[0].student_checkpoint == "student_ckpt_1"
    assert rollouts[0].policy_staleness_steps == 0
    assert out_file.exists()


def test_teacher_advantage_evaluator() -> None:
    # Student with lower accuracy, teacher with higher accuracy
    student_b = DeterministicFakeBackend(degradation=0.30)
    teacher_b = DeterministicFakeBackend(degradation=0.0)
    evaluator = TeacherAdvantageEvaluator(student_b, teacher_b, min_advantage_threshold=3.0)

    states = extract_prompt_states(
        [
            {
                "id": f"s_{i}",
                "source": "canonical",
                "tools": [{"name": "lookup"}],
                "messages": [
                    {"role": "user", "content": f"Query {i}"},
                    {"role": "assistant", "content": "", "tool_calls": [{"name": "lookup"}]},
                ],
                "metadata": {},
            }
            for i in range(5)
        ]
    )

    report = evaluator.evaluate(states)
    assert report.samples_evaluated == 5
    summary = report.render_markdown()
    assert "Teacher-Student Advantage Evaluation" in summary


def test_check_distillation_memory_safety() -> None:
    mem = check_distillation_memory_safety()
    assert mem.mode_selected in {
        "MODE_A_CO_RESIDENT",
        "MODE_C_BOUNDED_STALENESS",
        "MODE_A_MOCK_CPU",
    }
    assert mem.fits_in_vram is True
    assert mem.status == "PASS"
