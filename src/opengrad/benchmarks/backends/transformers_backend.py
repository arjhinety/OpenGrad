"""Lazy Hugging Face Transformers backend for model inference."""

from __future__ import annotations

import importlib
import time
from typing import Any

from opengrad.benchmarks.backends.protocol import GenerationResult


class TransformersInferenceBackend:
    """Lazy Transformers backend; importing/loading it is never needed for dry-runs."""

    name = "transformers"

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: str | None = None,
        precision: str = "bfloat16",
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.precision = precision
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            AutoModelForCausalLM = transformers.AutoModelForCausalLM
            AutoTokenizer = transformers.AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers backend requires the 'gpu-evaluation' or 'training' extra"
            ) from exc

        kwargs: dict[str, Any] = {"revision": self.revision, "trust_remote_code": False}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        dtype = torch.bfloat16 if self.precision == "bfloat16" else torch.float16
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map="auto", **kwargs
        )
        self._model.eval()
        return self._tokenizer, self._model

    def generate(
        self,
        prompt: str,
        *,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        tokenizer, model = self._load()
        torch = importlib.import_module("torch")

        config = generation_config or {}
        encoded = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[1])

        kwargs: dict[str, Any] = {
            "max_new_tokens": int(config.get("max_output_tokens", config.get("max_new_tokens", 512))),
            "do_sample": bool(config.get("do_sample", False)),
        }
        if config.get("temperature") is not None and float(config["temperature"]) > 0:
            kwargs["temperature"] = float(config["temperature"])
            kwargs["do_sample"] = True
        if config.get("top_p") is not None:
            kwargs["top_p"] = float(config["top_p"])

        start_time = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**encoded, **kwargs)
        elapsed = time.monotonic() - start_time

        generated_ids = output[0, prompt_tokens:]
        completion_tokens = len(generated_ids)
        decoded_text = str(tokenizer.decode(generated_ids, skip_special_tokens=False))

        return GenerationResult(
            text=decoded_text,
            token_ids=generated_ids.tolist(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency=round(elapsed, 6),
            ttft=round(elapsed / max(1, completion_tokens), 6),
            backend_metadata={
                "backend": self.name,
                "model_id": self.model_id,
                "revision": self.revision,
                "precision": self.precision,
            },
        )
