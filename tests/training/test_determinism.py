"""Seeding and kernel determinism (`opengrad.training.determinism`, Study 002 docs 05 and 14).

torch is not in the dev environment, so a recording stand-in checks the exact calls each mode makes.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import pytest

from opengrad.training.determinism import (
    CUBLAS_WORKSPACE_CONFIG,
    DECLARED_DETERMINISTIC,
    NON_DETERMINISTIC_KERNEL,
    UNDECLARED,
    apply_determinism,
    resolve_determinism,
    seed_everything,
)
from opengrad.training.dpo_runner import resolve_dpo_settings
from opengrad.training.sft_runner import TrainingConfigError, resolve_settings


class FakeTorch:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.cuda = SimpleNamespace(manual_seed_all=lambda s: self.calls.append(("cuda_seed", s)))
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True))

    def manual_seed(self, seed: int) -> None:
        self.calls.append(("seed", seed))

    def use_deterministic_algorithms(self, flag: bool) -> None:
        self.calls.append(("deterministic_algorithms", flag))


@pytest.mark.parametrize(
    ("block", "mode"),
    [
        ({}, UNDECLARED),
        ({"seed": 0}, UNDECLARED),
        ({"determinism": "DECLARED_DETERMINISTIC"}, DECLARED_DETERMINISTIC),
        ({"determinism": "NON_DETERMINISTIC_KERNEL"}, NON_DETERMINISTIC_KERNEL),
    ],
)
def test_resolve_determinism(block: dict, mode: str) -> None:
    assert resolve_determinism({"reproducibility": block}) == mode


def test_an_unknown_mode_is_an_error_not_a_default() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        resolve_determinism({"reproducibility": {"determinism": "deterministic"}})


def test_declared_deterministic_sets_every_flag_and_the_cublas_workspace() -> None:
    torch, env = FakeTorch(), {}
    record = apply_determinism(DECLARED_DETERMINISTIC, torch, env)
    assert torch.calls == [("deterministic_algorithms", True)]
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert CUBLAS_WORKSPACE_CONFIG == ":4096:8"
    assert env == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    assert record["cublas_workspace_config_set"] is True


def test_an_existing_cublas_setting_is_kept_and_recorded() -> None:
    env = {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}
    record = apply_determinism(DECLARED_DETERMINISTIC, FakeTorch(), env)
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert record["cublas_workspace_config_set"] is False
    assert record["cublas_workspace_config"] == ":16:8"


@pytest.mark.parametrize("mode", [NON_DETERMINISTIC_KERNEL, UNDECLARED])
def test_other_modes_change_no_flag(mode: str) -> None:
    torch, env = FakeTorch(), {}
    assert apply_determinism(mode, torch, env) == {"mode": mode}
    assert torch.calls == [] and env == {}
    assert torch.backends.cudnn.benchmark is True


def test_undeclared_seeds_exactly_what_study_001_seeded() -> None:
    torch = FakeTorch()
    state = random.getstate()
    assert seed_everything(7, UNDECLARED, torch) == ["torch", "torch.cuda"]
    assert torch.calls == [("seed", 7), ("cuda_seed", 7)]
    assert random.getstate() == state


def test_a_declared_mode_also_seeds_python_random() -> None:
    torch = FakeTorch()
    seeded = seed_everything(3, NON_DETERMINISTIC_KERNEL, torch)
    assert seeded[0] == "python.random" and seeded[-2:] == ["torch", "torch.cuda"]
    first = random.random()
    seed_everything(3, NON_DETERMINISTIC_KERNEL, FakeTorch())
    assert random.random() == first


def test_an_unknown_mode_is_refused_by_both_functions() -> None:
    with pytest.raises(ValueError):
        seed_everything(1, "sometimes", FakeTorch())
    with pytest.raises(ValueError):
        apply_determinism("sometimes", FakeTorch(), {})


def test_the_mode_is_in_both_recorded_settings() -> None:
    experiment = {"reproducibility": {"seed": 1, "determinism": NON_DETERMINISTIC_KERNEL}}
    sft_trainer = {"tuning_method": "full", "micro_batch_tokens": 4096}
    assert (
        resolve_settings(experiment, sft_trainer).to_dict()["determinism"]
        == NON_DETERMINISTIC_KERNEL
    )
    dpo = resolve_dpo_settings(experiment, {"reference": "initial_policy"})
    assert dpo.to_dict()["determinism"] == NON_DETERMINISTIC_KERNEL
    bad = {"reproducibility": {"determinism": "yes"}}
    with pytest.raises(TrainingConfigError):
        resolve_settings(bad, sft_trainer)
