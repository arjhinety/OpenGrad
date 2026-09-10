"""Distillation package."""

from opengrad.distillation.evaluator import (
    DistillationSmokeResult,
    TeacherAdvantageEvaluator,
    TeacherAdvantageReport,
    check_distillation_memory_safety,
)
from opengrad.distillation.prompts import (
    PromptState,
    extract_prompt_states,
)
from opengrad.distillation.rollouts import (
    DistillationRollout,
    OnPolicyRolloutGenerator,
)
from opengrad.distillation.tokenizer_gate import (
    TokenizerComparison,
    compare_tokenizers,
    validate_teacher_tokenizer_offline,
)

__all__ = [
    "DistillationRollout",
    "DistillationSmokeResult",
    "OnPolicyRolloutGenerator",
    "PromptState",
    "TeacherAdvantageEvaluator",
    "TeacherAdvantageReport",
    "TokenizerComparison",
    "check_distillation_memory_safety",
    "compare_tokenizers",
    "extract_prompt_states",
    "validate_teacher_tokenizer_offline",
]
