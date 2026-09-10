from pathlib import Path

from opengrad.training.distillation import OnPolicyDistillationTrainerBackend
from opengrad.training.dpo import DPOTrainerBackend
from opengrad.training.sft import SFTTrainerBackend
from opengrad.training.teacher import CachedTeacherProvider, MockTeacherProvider


def test_sft_trainer_backend_mock(tmp_path: Path) -> None:
    backend = SFTTrainerBackend()
    res = backend.train(
        "exp_sft_test",
        {"learning_rate": 2e-5, "max_steps": 5, "micro_batch_size": 2},
        output_dir=tmp_path,
        dry_run=True,
    )
    assert res.algorithm == "sft"
    assert res.total_steps == 5
    assert len(res.checkpoints_created) == 1
    assert Path(res.final_checkpoint_path).exists()


def test_dpo_trainer_backend_mock(tmp_path: Path) -> None:
    backend = DPOTrainerBackend()
    res = backend.train(
        "exp_dpo_test",
        {"learning_rate": 5e-6, "beta": 0.1, "max_steps": 5},
        output_dir=tmp_path,
        dry_run=True,
    )
    assert res.algorithm == "dpo"
    assert res.total_steps == 5
    assert "beta" in res.algorithm_diagnostics


def test_on_policy_distillation_trainer_mock(tmp_path: Path) -> None:
    backend = OnPolicyDistillationTrainerBackend()
    res = backend.train(
        "exp_distill_test",
        {"iterations": 2, "prompts_per_iteration": 3},
        output_dir=tmp_path,
        dry_run=True,
    )
    assert res.algorithm == "on_policy_distillation"
    assert res.total_steps == 10
    assert (tmp_path / "rollouts" / "rollout_history.jsonl").exists()


def test_cached_teacher_provider(tmp_path: Path) -> None:
    mock = MockTeacherProvider()
    cached = CachedTeacherProvider(mock, cache_dir=tmp_path)

    res1 = cached.generate_feedback("prompt_1", "student_output_1")
    assert res1.cached is False

    res2 = cached.generate_feedback("prompt_1", "student_output_1")
    assert res2.cached is True
    assert res2.score == res1.score
