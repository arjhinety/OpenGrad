#!/usr/bin/env python3
"""H200 + vLLM general-capability run across the OpenGrad checkpoint ladder.

This answers one question: the OpenWeights sentinel showed the promoted checkpoint declining tasks
it should be able to do, and the frozen tool-policy partition contains no ANSWER examples, so
nothing in the primary evaluation could have detected it. Is that lost capability, or learned
refusal -- and at which training stage does it first appear?

Design, all of it driven by where the cost actually is:

* **Cost is dominated by container start and model load, not by tokens.** Loading a 2B checkpoint
  costs ~95 GPU-seconds; generating all of IFEval costs a few. So one container per CHECKPOINT
  runs every benchmark, rather than one container per benchmark. Twelve model loads would have
  cost roughly three times what four do, and bought nothing.
* **Persist after every job, never at the end.** A worker that dies during MMLU-Pro must not take
  IFEval and GSM8K down with it. Each job commits to the volume as it finishes.
* **The checkpoint is verified before the engine starts.** Loading the wrong weights and finding
  out afterwards means paying for the GPU twice.
* **Greedy everywhere.** temperature 0, top_k 1, top_p 1, fixed seed, prefix caching off. Prefix
  caching would make MMLU-Pro results depend on submission order, since all items in a category
  share a 5-shot prefix.

Nothing here fabricates a score. A stage that cannot be loaded records a status and returns; a
benchmark whose request file is absent is skipped explicitly rather than silently.

Usage:
    modal run scripts/modal/h200_capability.py --stage M1_DPO_CURRENT --jobs ifeval,gsm8k,sentinel
    modal run scripts/modal/h200_capability.py --stage BASE --jobs mmlu_pro --mmlu-limit 280
"""

# NOTE: deliberately no `from __future__ import annotations`. Modal inspects the annotation on
# `modal.parameter()` fields to pick a serializer, and with PEP 563 in force it receives the string
# "str" instead of the type and fails with "'str' object has no attribute '__name__'". Python 3.12
# evaluates `int | None` natively, so the future import buys nothing here anyway.

import json
import os
import time
from pathlib import Path

import modal

GPU = "H200"
GPU_HOURLY_USD = 4.54  # confirmed from `modal billing rates`, not assumed

VOL = "/vol"
MAX_MODEL_LEN = 5760
SEED = 0

# This module is re-imported INSIDE the container as /root/h200_capability.py, where there is no
# grandparent directory and parents[2] raises IndexError at import time -- before any GPU is
# allocated, but also before anything useful runs. ROOT is only needed locally, to resolve the
# files added to the image, so the container branch can be any valid path.
_PARENTS = Path(__file__).resolve().parents
ROOT = _PARENTS[2] if len(_PARENTS) >= 3 else Path(".")

# Mirrors results/benchmarks/checkpoint_ladder.json. Duplicated here rather than read from it so
# the container has no dependency on a file that could drift; the runner asserts the two agree
# before spending GPU time.
LADDER = {
    "BASE": {
        "repo": "Qwen/Qwen3.5-2B",
        "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "subfolder": "",
        "expected_weight_sha256": None,   # multi-file naming; identity asserted via config sha
        "expected_config_sha256": None,
        "architecture_note": "Qwen3_5ForConditionalGeneration (multimodal). Run text-only.",
    },
    "M0_SFT": {
        "repo": "arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CanonicalV2-Final",
        "revision": "acffaf6eb068e556f9a127f36c9478201afb49fa",
        "subfolder": "",
        "expected_weight_sha256": None,
        "expected_config_sha256": None,
        "architecture_note": "Qwen3_5ForCausalLM",
    },
    "M1_DPO_CURRENT": {
        "repo": "arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2",
        "revision": "f33d20308982f37deb459076f489e794d5521ee3",
        "subfolder": "dpo-checkpoint-30",
        # The one stage with a hard pin: this is the promoted checkpoint and every earlier result
        # in the study is anchored to these digests.
        "expected_weight_sha256": "903f9b115d418d1988c0d5e1dd4bda7e04809a54a6bc7e34431982fbee2262f6",
        "expected_config_sha256": "88bf86c270d616198909ed1eefef8d8c21ac1fa13f62e947f20f8e1ebd02c211",
        "architecture_note": "Qwen3_5ForCausalLM",
    },
    "M1_DPO_HISTORICAL": {
        "repo": "arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO",
        "revision": "3a1250c7b8e86758e2605ee67092e24fc8afa5aa",
        "subfolder": "checkpoint-300",
        "expected_weight_sha256": None,
        "expected_config_sha256": None,
        "architecture_note": "Qwen3_5ForCausalLM; DPO applied directly to base, no SFT parent -- supplementary only",
    },
}

app = modal.App("opengrad-h200-capability")
volume = modal.Volume.from_name("opengrad-quant", create_if_missing=False)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .env({
        "FLASHINFER_WORKSPACE_BASE": "/vol/h200/flashinfer",
        "TORCHINDUCTOR_CACHE_DIR": "/vol/h200/inductor",
        "VLLM_CACHE_ROOT": "/vol/h200/vllm",
        "CUDA_HOME": "/usr/local/cuda",
        "HF_HOME": "/vol/hf",
    })
    # Same engine as the frozen BF16 reference, so these generations sit in the same measurement
    # frame as everything already recorded. 0.11.0 rejects Qwen3_5ForCausalLM outright.
    .pip_install("vllm==0.29.0", "transformers==5.14.1", "huggingface_hub")
    .add_local_file(str(ROOT / "results/benchmarks/datasets/ifeval_v1.jsonl"), "/data/ifeval.jsonl")
    .add_local_file(str(ROOT / "results/benchmarks/datasets/gsm8k_v1.jsonl"), "/data/gsm8k.jsonl")
    .add_local_file(str(ROOT / "results/benchmarks/datasets/mmlu_pro_v1.jsonl"), "/data/mmlu_pro.jsonl")
    .add_local_file(str(ROOT / "results/benchmarks/openweights_parity_cases_v1.json"),
                    "/data/sentinel.json")
)


def _sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pkg_version(name: str) -> str:
    from importlib.metadata import version

    try:
        return version(name)
    except Exception:  # noqa: BLE001 - provenance records "unknown" rather than aborting the run
        return "unknown"


@app.cls(
    image=image,
    gpu=GPU,
    volumes={VOL: volume},
    timeout=60 * 120,
    cpu=16.0,
    memory=131072,
    scaledown_window=60 * 2,
)
class CapabilityEngine:
    stage: str = modal.parameter()

    @modal.enter()
    def start(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from vllm import LLM

        self.container_started_at = time.time()
        spec = LADDER[self.stage]
        self.spec = spec

        local_root = f"{VOL}/ladder/{self.stage}"
        os.makedirs(local_root, exist_ok=True)
        patterns = [f"{spec['subfolder']}/*"] if spec["subfolder"] else None
        download_started = time.time()
        snapshot_download(
            spec["repo"], revision=spec["revision"], local_dir=local_root,
            allow_patterns=patterns,
            # training_state.pt is an optimizer snapshot, hundreds of MB, never used for inference.
            ignore_patterns=["*.pt", "*.html", "evaluation/*", "README.md"],
        )
        self.download_seconds = round(time.time() - download_started, 3)
        model_dir = os.path.join(local_root, spec["subfolder"]) if spec["subfolder"] else local_root

        weight_files = sorted(
            f for f in os.listdir(model_dir) if ".safetensors" in f and not f.endswith(".json")
        )
        if not weight_files:
            raise RuntimeError(f"no safetensors under {model_dir}: {os.listdir(model_dir)}")
        self.weight_sha = _sha256(os.path.join(model_dir, weight_files[0])) if len(weight_files) == 1 else None
        self.config_sha = _sha256(os.path.join(model_dir, "config.json"))

        # Hard identity check where a digest is pinned. A silently different checkpoint would
        # invalidate every comparison against the frozen results.
        if spec["expected_weight_sha256"] and self.weight_sha != spec["expected_weight_sha256"]:
            raise RuntimeError(f"{self.stage} weight sha mismatch: {self.weight_sha}")
        if spec["expected_config_sha256"] and self.config_sha != spec["expected_config_sha256"]:
            raise RuntimeError(f"{self.stage} config sha mismatch: {self.config_sha}")

        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        volume.commit()

        load_started = time.time()
        self.llm = LLM(
            model=model_dir,
            dtype="bfloat16",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.90,
            max_num_seqs=256,
            enable_prefix_caching=False,  # would make MMLU-Pro depend on submission order
            seed=SEED,
            trust_remote_code=True,
        )
        self.load_seconds = round(time.time() - load_started, 3)

        props = torch.cuda.get_device_properties(0)
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as handle:
            cfg = json.load(handle)
        template_path = os.path.join(model_dir, "chat_template.jinja")
        self.environment = {
            "stage": self.stage,
            "model_repo": spec["repo"],
            "model_revision": spec["revision"],
            "subfolder": spec["subfolder"],
            "architectures": cfg.get("architectures"),
            "weights_sha256": self.weight_sha,
            "weight_files": weight_files,
            "config_sha256": self.config_sha,
            "chat_template_sha256": _sha256(template_path) if os.path.exists(template_path) else None,
            "tokenizer_json_sha256": _sha256(os.path.join(model_dir, "tokenizer.json")),
            "add_bos_token": bool(getattr(self.tokenizer, "add_bos_token", False)),
            "eos_token": self.tokenizer.eos_token,
            "eos_token_id": self.tokenizer.eos_token_id,
            "gpu_name": props.name,
            "gpu_total_memory_gib": round(props.total_memory / 1024**3, 2),
            "gpu_capability": f"{props.major}.{props.minor}",
            "gpu_hourly_usd": GPU_HOURLY_USD,
            "cuda": str(torch.version.cuda),
            "torch": str(torch.__version__),
            "vllm": _pkg_version("vllm"),
            "transformers": _pkg_version("transformers"),
            "dtype": "bfloat16",
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": 256,
            "enable_prefix_caching": False,
            "seed": SEED,
            "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": SEED},
            "model_load_seconds": self.load_seconds,
            "download_seconds": self.download_seconds,
        }
        print(json.dumps(self.environment, indent=1), flush=True)

    # -- primitives ----------------------------------------------------------------------------

    def _sampling(self, max_tokens: int):
        from vllm import SamplingParams

        return SamplingParams(temperature=0.0, top_p=1.0, top_k=1, seed=SEED,
                              max_tokens=max_tokens, n=1)

    def _render(self, messages: list[dict], tools: list | None = None) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )

    def _persist_jsonl(self, name: str, rows: list[dict]) -> str:
        out_dir = f"{VOL}/h200/capability/{self.stage}"
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/{name}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n" for r in rows)
        volume.commit()
        return path

    def _persist_json(self, name: str, payload: dict) -> str:
        out_dir = f"{VOL}/h200/capability/{self.stage}"
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/{name}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        volume.commit()
        return path

    def _run_requests(self, requests: list[dict], label: str) -> dict:
        """Generate for a list of frozen requests and persist the generations before returning.

        Request identity is preserved by construction: vLLM returns outputs in submission order,
        and that is reconciled against the submitted id list before anything is written.
        """
        prompts = [self._render(r["messages"], r.get("tools")) for r in requests]
        max_tokens = max(r["max_tokens"] for r in requests)
        prompt_tokens_each = [
            len(self.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts
        ]
        over = [
            r["example_id"] for r, n in zip(requests, prompt_tokens_each)
            if n + r["max_tokens"] > MAX_MODEL_LEN
        ]
        if over:
            # Truncation would silently change what the benchmark asked. Fail instead.
            raise RuntimeError(f"{label}: {len(over)} prompts exceed the context budget, e.g. {over[:3]}")

        started = time.time()
        outputs = self.llm.generate(prompts, self._sampling(max_tokens))
        elapsed = time.time() - started

        if len(outputs) != len(requests):
            raise RuntimeError(f"{label}: submitted {len(requests)}, got {len(outputs)}")

        meta = {**self.environment, "benchmark": label, "max_tokens": max_tokens}
        rows = []
        for req, out, ptok in zip(requests, outputs, prompt_tokens_each):
            gen = out.outputs[0]
            rows.append({
                "example_id": req["example_id"],
                "benchmark": req.get("benchmark", label),
                "output": gen.text,
                "finish_reason": gen.finish_reason,
                "output_tokens": len(gen.token_ids),
                "prompt_tokens": ptok,
                "prompt_sha256": __import__("hashlib").sha256(
                    self._render(req["messages"], req.get("tools")).encode("utf-8")).hexdigest(),
                "generation_metadata": meta,
            })

        # Round-trip check: every submitted id present exactly once, nothing invented.
        got = [r["example_id"] for r in rows]
        if sorted(got) != sorted(r["example_id"] for r in requests):
            raise RuntimeError(f"{label}: example_id round-trip failed")

        path = self._persist_jsonl(f"generations_{label}", rows)
        prompt_tokens = sum(r["prompt_tokens"] for r in rows)
        output_tokens = sum(r["output_tokens"] for r in rows)
        truncated = sum(1 for r in rows if r["finish_reason"] == "length")
        stats = {
            "benchmark": label,
            "requests": len(requests),
            "elapsed_seconds": round(elapsed, 3),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens_per_second": round((prompt_tokens + output_tokens) / elapsed, 2) if elapsed else None,
            "truncated_by_length": truncated,
            "estimated_gpu_cost_usd": round(elapsed / 3600 * GPU_HOURLY_USD, 4),
            "generations_path": path,
        }
        print(json.dumps(stats, indent=1), flush=True)
        return stats

    # -- jobs ----------------------------------------------------------------------------------

    @modal.method()
    def run(self, jobs: list[str], mmlu_limit: int | None = None,
            gsm8k_arms: list[str] | None = None) -> dict:
        results = {}
        for job in jobs:
            try:
                if job == "sentinel":
                    results[job] = self._sentinel()
                else:
                    requests = self._load_requests(job, mmlu_limit, gsm8k_arms)
                    results[job] = self._run_requests(requests, job)
                results[job]["status"] = "COMPLETE"
            except Exception as exc:  # noqa: BLE001 - a failed job must not stop the others
                # A failed job is recorded, not hidden, and the remaining jobs still run.
                results[job] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
                print(f"JOB {job} FAILED: {type(exc).__name__}: {exc}", flush=True)
            self._persist_json("run_summary", {
                "stage": self.stage, "environment": self.environment, "jobs": results,
            })
        payload = {
            "stage": self.stage,
            "environment": self.environment,
            "jobs": results,
            "container_seconds": round(time.time() - self.container_started_at, 2),
        }
        payload["estimated_container_cost_usd"] = round(
            payload["container_seconds"] / 3600 * GPU_HOURLY_USD, 4)
        self._persist_json("run_summary", payload)
        return json.loads(json.dumps(payload, default=str))

    def _load_requests(self, job: str, mmlu_limit: int | None,
                       gsm8k_arms: list[str] | None) -> list[dict]:
        path = {"ifeval": "/data/ifeval.jsonl", "gsm8k": "/data/gsm8k.jsonl",
                "mmlu_pro": "/data/mmlu_pro.jsonl"}[job]
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if job == "gsm8k" and gsm8k_arms:
            rows = [r for r in rows if r["arm"] in gsm8k_arms]
        if job == "mmlu_pro" and mmlu_limit:
            # A pilot must be representative, not the first N of one category: take a
            # deterministic stratified slice, evenly spread across every category.
            by_cat: dict[str, list] = {}
            for r in rows:
                by_cat.setdefault(r["category"], []).append(r)
            per_cat = max(1, mmlu_limit // len(by_cat))
            picked = []
            for cat in sorted(by_cat):
                items = sorted(by_cat[cat], key=lambda r: r["example_id"])
                step = max(1, len(items) // per_cat)
                picked.extend(items[::step][:per_cat])
            rows = sorted(picked, key=lambda r: r["example_id"])
        return rows

    def _sentinel(self) -> dict:
        """The unmodified 7-case OpenWeights ParitySuite, replayed for this stage.

        Multi-turn cases feed the model's OWN first reply back into history verbatim, exactly as
        OpenWeights does on device. The cases are not edited -- this is a regression smoke test,
        not a capability benchmark, and its value depends entirely on staying fixed.
        """
        with open("/data/sentinel.json", encoding="utf-8") as handle:
            spec = json.load(handle)
        tool = spec["weather_tool"]
        max_tokens = spec["params"]["max_tokens"]
        started = time.time()
        rows = []

        for case in spec["cases"]:
            tools = [tool] if case.get("tools") else None
            history: list[dict] = []
            raw = ""
            for turn in case["turns"]:
                role = turn["role"]
                if role == "assistant_raw_placeholder":
                    history.append({"role": "assistant", "content": raw})
                    continue
                if role == "tool_result":
                    history.append({"role": "tool", "name": turn["name"], "content": turn["content"]})
                    continue
                if role == "user_none":
                    pass  # regenerate with no new user message
                else:
                    history.append({"role": role, "content": turn["content"]})
                prompt = self._render(history, tools)
                out = self.llm.generate([prompt], self._sampling(max_tokens))[0]
                raw = out.outputs[0].text
            rows.append({
                "example_id": case["id"],
                "benchmark": "sentinel",
                "id": case["id"],
                "output": raw,
                "raw": raw,
                "turns": len(case["turns"]),
                "generation_metadata": {**self.environment, "benchmark": "sentinel"},
            })

        path = self._persist_jsonl("generations_sentinel", rows)
        return {
            "benchmark": "sentinel",
            "requests": len(rows),
            "elapsed_seconds": round(time.time() - started, 3),
            "generations_path": path,
            "estimated_gpu_cost_usd": round((time.time() - started) / 3600 * GPU_HOURLY_USD, 4),
        }


@app.local_entrypoint()
def main(stage: str, jobs: str = "ifeval,gsm8k,sentinel", mmlu_limit: int = 0,
         gsm8k_arms: str = ""):
    if stage not in LADDER:
        raise SystemExit(f"unknown stage {stage!r}; known: {sorted(LADDER)}")
    job_list = [j.strip() for j in jobs.split(",") if j.strip()]
    arms = [a.strip() for a in gsm8k_arms.split(",") if a.strip()] or None
    engine = CapabilityEngine(stage=stage)
    result = engine.run.remote(job_list, mmlu_limit or None, arms)
    print(json.dumps(result, indent=2, default=str))
