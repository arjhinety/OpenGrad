"""Training backends package supporting SFT, DPO, and On-Policy Distillation."""

from opengrad.training.distillation import (
    MockRolloutProvider,
    OnPolicyDistillationTrainerBackend,
    RolloutProvider,
    RolloutRecord,
)
from opengrad.training.dpo import DPOTrainerBackend
from opengrad.training.protocol import (
    TrainerBackend,
    TrainingMetadata,
    TrainingRunResult,
)
from opengrad.training.sft import SFTTrainerBackend
from opengrad.training.teacher import (
    CachedTeacherProvider,
    MockTeacherProvider,
    TeacherProvider,
    TeacherResponse,
)

__all__ = [
    "CachedTeacherProvider",
    "DPOTrainerBackend",
    "MockRolloutProvider",
    "MockTeacherProvider",
    "OnPolicyDistillationTrainerBackend",
    "RolloutProvider",
    "RolloutRecord",
    "SFTTrainerBackend",
    "TeacherProvider",
    "TeacherResponse",
    "TrainerBackend",
    "TrainingMetadata",
    "TrainingRunResult",
]
