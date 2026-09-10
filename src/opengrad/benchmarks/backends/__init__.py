"""Inference backends package."""

from opengrad.benchmarks.backends.mock import DeterministicFakeBackend
from opengrad.benchmarks.backends.protocol import GenerationResult, InferenceBackend
from opengrad.benchmarks.backends.transformers_backend import TransformersInferenceBackend

__all__ = [
    "DeterministicFakeBackend",
    "GenerationResult",
    "InferenceBackend",
    "TransformersInferenceBackend",
]
