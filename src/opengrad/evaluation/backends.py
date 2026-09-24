"""Inference backends for the frozen baseline evaluation: the deterministic fake, Transformers and vLLM,
the pinned revisions they check, and `build_backend`. Split out of `runner.py` on 2026-09-24 with no change
in behaviour; `runner` re-exports these names.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Protocol

from opengrad.data.canonical import CanonicalEvaluationExample

PINNED_MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


PINNED_TEMPLATE_HASH = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"


PINNED_EVALUATOR_REVISION = "2d97c7d5a8de0b16a2e58e4376e231fe06ab16dc"


CANONICAL_MODEL_ID = "Qwen/Qwen3.5-2B"


BASELINE_EXPERIMENT_ID = "tool_calling/qwen35_2b/baseline"


# Engines a real run may declare. The engine is part of the measurement: two engines
# produce different tokens (different kernels, different samplers, different reduction
# order), so a baseline is only comparable to another run that names the same engine.
SUPPORTED_ENGINES = ("vllm", "transformers")


DEFAULT_ENGINE = "vllm"


class InferenceBackend(Protocol):
    """Stable generation interface shared by fake and model-backed runners.

    ``engine_metadata()`` and ``generate_batch()`` are optional but strongly preferred:
    the engine that produced a baseline is part of the measurement, and batched generation
    is what makes an A100 worth using. A backend without them is still valid and falls back
    to one-prompt-at-a-time generation.
    """

    name: str

    def generate(
        self,
        prompt: str,
        *,
        example: CanonicalEvaluationExample,
        generation_config: dict[str, Any],
    ) -> str: ...


def _decision(value: str) -> str:
    return {
        "tool_call": "CALL",
        "call": "CALL",
        "must_call": "CALL",
        "direct": "ANSWER",
        "can_answer": "ANSWER",
        "answer": "ANSWER",
        "request_for_info": "CLARIFY",
        "clarify": "CLARIFY",
        "cannot_answer": "UNSUPPORTED",
        "unsupported": "UNSUPPORTED",
    }.get(value.casefold(), "UNKNOWN")


class FakeDeterministicBackend:
    """CPU backend that emits native-looking output for every routing class."""

    name = "deterministic-mock"

    def __init__(
        self,
        *,
        adversarial: bool = False,
        scenarios: dict[str, str] | None = None,
    ) -> None:
        self.adversarial = adversarial
        self.scenarios = scenarios or {}
        self.last_truncated = False

    def generate(
        self,
        prompt: str,
        *,
        example: CanonicalEvaluationExample,
        generation_config: dict[str, Any],
    ) -> str:
        self.last_truncated = False
        explicit = self.scenarios.get(example.example_id)
        if explicit is not None:
            if explicit == "cutoff":
                self.last_truncated = True
                return '<tool_call>{"name":"cutoff","arguments":'
            if explicit == "observation":
                return (
                    '<tool_call>{"name":"lookup","arguments":{}}</tool_call>\nObservation: value x'
                )
            if explicit == "multiple":
                return '<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{}}</tool_call>'
            if explicit == "malformed":
                return '<tool_call>{"name":"broken","arguments":</tool_call>'
            return explicit
        expected = _decision(example.expected_decision)
        if self.adversarial:
            index = int(hashlib.sha256(example.example_id.encode()).hexdigest()[:2], 16) % 7
            if index == 0:
                return '<tool_call>{"name":"broken","arguments":</tool_call>'
            if index == 1:
                return (
                    '<tool_call>{"name":"a","arguments":{}}</tool_call>'
                    '<tool_call>{"name":"b","arguments":{}}</tool_call>'
                )
            if index == 2:
                return "Could you clarify which account you mean?"
            if index == 3:
                return "I cannot help with that unsupported request."
            if index == 4:
                return (
                    '<tool_call>{"name":"lookup","arguments":{}}</tool_call>\nObservation: value x'
                )
            if index == 5:
                self.last_truncated = True
                return '<tool_call>{"name":"cutoff","arguments":'
        if expected == "CALL":
            name = str(example.tools[0].get("name", "tool")) if example.tools else "tool"
            return f"<tool_call>{json.dumps({'name': name, 'arguments': {}}, sort_keys=True)}</tool_call>"
        if expected == "CLARIFY":
            return "Could you clarify the missing information?"
        if expected == "UNSUPPORTED":
            return "I cannot help with that unsupported request."
        return "Here is the requested information."


class TransformersInferenceBackend:
    """Lazy Transformers backend; importing/loading it is never needed for dry-run."""

    name = "transformers"

    def __init__(self, *, model_id: str, revision: str, cache_dir: str | None = None) -> None:
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self._tokenizer: Any = None
        self._model: Any = None
        self.last_truncated = False

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            AutoModelForCausalLM = transformers.AutoModelForCausalLM
            AutoTokenizer = transformers.AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("GPU inference requires the evaluation extra") from exc
        kwargs: dict[str, Any] = {"revision": self.revision, "trust_remote_code": False}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto", **kwargs
        )
        self._model.eval()
        return self._tokenizer, self._model

    def generate(
        self,
        prompt: str,
        *,
        example: CanonicalEvaluationExample,
        generation_config: dict[str, Any],
    ) -> str:
        tokenizer, model = self._load()
        torch = importlib.import_module("torch")

        encoded = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        kwargs = {
            key: value
            for key, value in generation_config.items()
            if key in {"temperature", "top_p", "max_new_tokens", "do_sample"}
        }
        with torch.inference_mode():
            output = model.generate(**encoded, **kwargs)
        generated = output[0, encoded["input_ids"].shape[1] :]
        self.last_truncated = len(generated) >= int(generation_config.get("max_new_tokens", 0))
        return str(tokenizer.decode(generated, skip_special_tokens=False))

    def engine_metadata(self) -> dict[str, Any]:
        version = None
        try:
            version = importlib.import_module("transformers").__version__
        except ImportError:
            pass
        return {
            "name": self.name,
            "version": version,
            "model_id": self.model_id,
            "revision": self.revision,
            "batching": "serial",
        }


class VLLMInferenceBackend:
    """vLLM engine: continuous batching, and the declared engine of record for B0.

    Prompts are rendered by OpenGrad's pinned Qwen renderer and handed here as raw text, so
    this backend never re-applies a chat template. That is deliberate: vLLM ships its own
    tokenizer/template stack, and letting it render the prompt would put the frozen prompt
    contract in the hands of a component the frozen config does not name. The prompt that
    was validated is the prompt that is generated from.

    Throughput comes from ``generate_batch``: vLLM schedules the submitted prompts
    continuously, so the runner's submission chunk bounds memory and progress granularity
    rather than the engine's utilisation.
    """

    name = "vllm"

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int = 256,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
        seed: int = 0,
        cache_dir: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        self.seed = seed
        self.cache_dir = cache_dir
        self._llm: Any = None
        self.last_truncated = False
        self.last_truncated_flags: list[bool] = []

    def _load(self) -> Any:
        if self._llm is not None:
            return self._llm
        try:
            vllm = importlib.import_module("vllm")
        except ImportError as exc:
            raise RuntimeError(
                "the vllm engine requires the gpu-vllm extra: pip install '.[gpu-vllm]'"
            ) from exc
        self._expose_venv_bin_for_jit()
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "revision": self.revision,
            "trust_remote_code": False,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "tensor_parallel_size": self.tensor_parallel_size,
            "enforce_eager": self.enforce_eager,
            "seed": self.seed,
            "disable_log_stats": True,
        }
        if self.cache_dir:
            kwargs["download_dir"] = self.cache_dir
        self._llm = vllm.LLM(**kwargs)
        return self._llm

    @staticmethod
    def _expose_venv_bin_for_jit() -> None:
        """Put this interpreter's bin directory on PATH.

        vLLM's flashinfer kernels JIT-compile at engine startup and shell out to a `ninja`
        executable. Invoked as ``.venv/bin/opengrad`` the venv's bin directory is not on
        PATH, so that build dies with ``FileNotFoundError: 'ninja'`` and surfaces only as
        the unhelpful "Engine core initialization failed". The venv ships ninja as its own
        console script, so making its bin directory visible fixes it without touching the
        host environment.
        """
        bin_dir = str(Path(sys.executable).parent)
        entries = os.environ.get("PATH", "").split(os.pathsep)
        if bin_dir and bin_dir not in entries:
            os.environ["PATH"] = os.pathsep.join([bin_dir, *entries])

    def engine_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "name": self.name,
            "model_id": self.model_id,
            "revision": self.revision,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "tensor_parallel_size": self.tensor_parallel_size,
            "enforce_eager": self.enforce_eager,
            "seed": self.seed,
            "batching": "continuous",
        }
        try:
            metadata["version"] = importlib.import_module("vllm").__version__
        except ImportError:
            metadata["version"] = None
        return metadata

    def _sampling_params(self, generation_config: dict[str, Any]) -> Any:
        vllm = importlib.import_module("vllm")
        # temperature 0.0 is greedy, which is what the frozen do_sample=False asks for.
        return vllm.SamplingParams(
            temperature=float(generation_config.get("temperature", 0.0)),
            top_p=float(generation_config.get("top_p", 1.0)),
            max_tokens=int(generation_config["max_new_tokens"]),
            n=1,
            seed=self.seed,
        )

    def generate_batch(
        self,
        prompts: list[str],
        *,
        examples: list[CanonicalEvaluationExample],
        generation_config: dict[str, Any],
    ) -> list[str]:
        llm = self._load()
        params = self._sampling_params(generation_config)
        try:
            outputs = llm.generate(prompts, params, use_tqdm=False)
        except Exception:  # noqa: BLE001 - identify which example the engine refused
            # One prompt the engine cannot schedule must not take the run down without
            # saying which: retry singly so the failure names its example.
            return self._generate_individually(llm, prompts, examples, params)
        raws: list[str] = []
        flags: list[bool] = []
        for output in outputs:
            completion = output.outputs[0]
            raws.append(str(completion.text))
            flags.append(str(getattr(completion, "finish_reason", "")) == "length")
        self.last_truncated_flags = flags
        self.last_truncated = any(flags)
        return raws

    def _generate_individually(
        self, llm: Any, prompts: list[str], examples: list[CanonicalEvaluationExample], params: Any
    ) -> list[str]:
        raws: list[str] = []
        flags: list[bool] = []
        for prompt, example in zip(prompts, examples):
            try:
                output = llm.generate([prompt], params, use_tqdm=False)[0]
            except Exception as exc:
                raise RuntimeError(
                    f"vllm could not generate for example {example.example_id}: {exc}"
                ) from exc
            completion = output.outputs[0]
            raws.append(str(completion.text))
            flags.append(str(getattr(completion, "finish_reason", "")) == "length")
        self.last_truncated_flags = flags
        self.last_truncated = any(flags)
        return raws

    def generate(
        self,
        prompt: str,
        *,
        example: CanonicalEvaluationExample,
        generation_config: dict[str, Any],
    ) -> str:
        return self.generate_batch(
            [prompt], examples=[example], generation_config=generation_config
        )[0]


def build_backend(name: str, config: dict[str, Any]) -> InferenceBackend:
    """Construct the declared engine for a real run."""
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    model_id = str(config.get("model_id", CANONICAL_MODEL_ID))
    revision = str(config.get("model_revision", PINNED_MODEL_REVISION))
    common = {
        "model_id": model_id,
        "revision": revision,
    }
    if name == "vllm":
        context_length = int(runtime.get("context_length", 4096))
        # vLLM refuses a request whose prompt plus max_tokens exceeds max_model_len, so the
        # window must cover the largest in-contract prompt *and* its completion. Prompts
        # beyond context_length are bucketed as overflow by the frozen policy, not dropped.
        max_new_tokens = int((config.get("generation") or {}).get("max_new_tokens", 512))
        return VLLMInferenceBackend(
            **common,
            dtype=str(runtime.get("precision", "bfloat16")),
            max_model_len=int(runtime.get("max_model_len") or context_length + max_new_tokens),
            gpu_memory_utilization=float(runtime.get("gpu_memory_utilization", 0.90)),
            max_num_seqs=int(runtime.get("max_num_seqs", 256)),
            tensor_parallel_size=int(runtime.get("tensor_parallel_size", 1)),
            enforce_eager=bool(runtime.get("enforce_eager", False)),
            seed=int(config.get("seed", 0)),
        )
    if name == "transformers":
        return TransformersInferenceBackend(**common)
    raise ValueError(f"unsupported engine: {name} (supported: {', '.join(SUPPORTED_ENGINES)})")
