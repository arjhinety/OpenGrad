#!/usr/bin/env python3
"""H200 + vLLM evaluation of the frozen M1-v2 checkpoint, under a hard credit budget.

Design decisions that exist because credits are the binding constraint:

* **One model load.** A single `@app.cls` holds the vLLM engine for the life of the container;
  every phase reuses it. Loading a 2B model per benchmark would dominate the bill.
* **The checkpoint is already on the `opengrad-quant` volume** from the quantization phase, so no
  paid GPU time is spent re-downloading ~3.5GB from HuggingFace.
* **Offline batch `generate()` rather than an HTTP server.** Every workload here is a fixed prompt
  set, and vLLM's scheduler already applies continuous batching across a submitted list. An HTTP
  layer would add client tuning and overhead without changing GPU occupancy for this shape of work.
  Multi-turn cases are the exception and are driven sequentially inside one call.
* **Mode B is exclusive.** The performance phase runs alone; a capability workload sharing the GPU
  would silently invalidate every latency number.

Nothing here fabricates a score. Benchmarks whose datasets do not exist in this repo are not run
and are recorded as blocked elsewhere.

Usage:
    modal run scripts/modal/h200_eval.py --phase validate
    modal run scripts/modal/h200_eval.py --phase sweep
    modal run scripts/modal/h200_eval.py --phase frozen
    modal run scripts/modal/h200_eval.py --phase parity
    modal run scripts/modal/h200_eval.py --phase perf
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

GPU = "H200"
GPU_HOURLY_USD = 4.54  # confirmed from `modal billing rates`, not assumed

MODEL_REPO = "arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2"
MODEL_REVISION = "f33d20308982f37deb459076f489e794d5521ee3"
CHECKPOINT = "dpo-checkpoint-30"
EXPECTED_WEIGHT_SHA256 = "903f9b115d418d1988c0d5e1dd4bda7e04809a54a6bc7e34431982fbee2262f6"
EXPECTED_CONFIG_SHA256 = "88bf86c270d616198909ed1eefef8d8c21ac1fa13f62e947f20f8e1ebd02c211"

VOL = "/vol"
MODEL_DIR = f"{VOL}/source/m1-v2/{CHECKPOINT}"

# The frozen measurement contract, identical to the quantization phase so results are comparable.
MAX_MODEL_LEN = 5760
MAX_NEW_TOKENS = 512
SEED = 0

app = modal.App("opengrad-h200-eval")
volume = modal.Volume.from_name("opengrad-quant", create_if_missing=False)

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path(".")

# A CUDA *devel* base, not debian_slim: FlashInfer JIT-compiles its sampling and Gated DeltaNet
# prefill kernels at engine start and needs nvcc. Without a toolkit the engine dies with
# "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist" after the model has
# already loaded — i.e. after paying for the GPU. This base is also already cached in this
# workspace from the GGUF phase, so it costs no extra pull.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .env({
        # Compile once, reuse across phases: the JIT cache lives on the volume.
        "FLASHINFER_WORKSPACE_BASE": "/vol/h200/flashinfer",
        "TORCHINDUCTOR_CACHE_DIR": "/vol/h200/inductor",
        "VLLM_CACHE_ROOT": "/vol/h200/vllm",
        "CUDA_HOME": "/usr/local/cuda",
    })
    # vLLM 0.29.0 is not an arbitrary pin: it is the exact engine that produced the frozen
    # BF16 reference (results/quantization/m1_v2_reference.json -> runtime.recorded_engine),
    # so regenerated per-example predictions are comparable to it. It is also the first
    # version that supports Qwen3_5ForCausalLM at all — 0.11.0 rejects the architecture.
    .pip_install("vllm==0.29.0", "transformers==5.14.1", "huggingface_hub")
    .add_local_file(
        str(ROOT / "results/quantization/frozen_prompts_v1.jsonl"),
        "/data/frozen_prompts_v1.jsonl",
    )
    .add_local_file(
        str(ROOT / "results/benchmarks/openweights_parity_cases_v1.json"),
        "/data/openweights_parity_cases_v1.json",
    )
)


def _sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.cls(
    image=image,
    gpu=GPU,
    volumes={VOL: volume},
    timeout=60 * 90,
    cpu=16.0,
    memory=131072,
    scaledown_window=60 * 5,
)
class Engine:
    """One persistent vLLM engine on one H200. Every phase is a method on this class."""

    @modal.enter()
    def start(self):
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        self.started_at = time.time()

        # Integrity before inference. A silently different checkpoint would invalidate every
        # comparison against the frozen quantization results.
        weights = f"{MODEL_DIR}/model.safetensors"
        if not os.path.exists(weights):
            raise RuntimeError(f"checkpoint missing from volume: {MODEL_DIR}")
        self.weight_sha = _sha256(weights)
        self.config_sha = _sha256(f"{MODEL_DIR}/config.json")
        if self.weight_sha != EXPECTED_WEIGHT_SHA256:
            raise RuntimeError(f"weights sha mismatch: {self.weight_sha}")
        if self.config_sha != EXPECTED_CONFIG_SHA256:
            raise RuntimeError(f"config sha mismatch: {self.config_sha}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        load_started = time.time()
        self.llm = LLM(
            model=MODEL_DIR,
            dtype="bfloat16",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.90,
            max_num_seqs=256,
            enable_prefix_caching=False,  # prefix reuse would make a result depend on order
            seed=SEED,
            trust_remote_code=True,
        )
        self.load_seconds = round(time.time() - load_started, 3)

        props = torch.cuda.get_device_properties(0)
        self.environment = {
            "gpu_name": props.name,
            "gpu_total_memory_gib": round(props.total_memory / 1024**3, 2),
            "gpu_capability": f"{props.major}.{props.minor}",
            "modal_gpu_identifier": GPU,
            "gpu_hourly_usd": GPU_HOURLY_USD,
            "cuda": str(torch.version.cuda),
            "torch": str(torch.__version__),
            "driver": _driver_version(),
            "vllm": _pkg_version("vllm"),
            "transformers": _pkg_version("transformers"),
            "model_dir": MODEL_DIR,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "checkpoint": CHECKPOINT,
            "weights_sha256": self.weight_sha,
            "config_sha256": self.config_sha,
            "dtype": "bfloat16",
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": 0.90,
            "max_num_seqs": 256,
            "enable_prefix_caching": False,
            "seed": SEED,
            "model_load_seconds": self.load_seconds,
            "chat_template_sha256": _sha256(f"{MODEL_DIR}/chat_template.jinja")
            if os.path.exists(f"{MODEL_DIR}/chat_template.jinja") else None,
            "add_bos_token": bool(getattr(self.tokenizer, "add_bos_token", False)),
            "eos_token": self.tokenizer.eos_token,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        print(json.dumps(self.environment, indent=1), flush=True)


    def _persist(self, name: str, payload: dict) -> dict:
        """Write the result on the volume before returning it.

        The return trip can fail for reasons that have nothing to do with the computation — a
        library type that the local CLI cannot unpickle already destroyed one completed run. The
        authoritative copy is the one written here.
        """
        out_dir = f"{VOL}/h200/results"
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/{name}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        volume.commit()
        payload["persisted_to"] = path
        # json round-trip strips any residual library types before the value crosses the wire.
        return json.loads(json.dumps(payload, default=str))

    # -- generation primitives -----------------------------------------------------------------

    def _sampling(self, max_tokens: int = MAX_NEW_TOKENS):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=SEED,
            max_tokens=max_tokens, n=1,
        )

    def _generate(self, prompts: list[str], max_tokens: int = MAX_NEW_TOKENS) -> tuple[list, dict]:
        started = time.time()
        outputs = self.llm.generate(prompts, self._sampling(max_tokens))
        elapsed = time.time() - started
        prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        stats = {
            "requests": len(prompts),
            "elapsed_seconds": round(elapsed, 3),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "requests_per_second": round(len(prompts) / elapsed, 3) if elapsed else None,
            "output_tokens_per_second": round(output_tokens / elapsed, 2) if elapsed else None,
            "total_tokens_per_second": round((prompt_tokens + output_tokens) / elapsed, 2)
            if elapsed else None,
            "estimated_gpu_cost_usd": round(elapsed / 3600 * GPU_HOURLY_USD, 4),
        }
        return outputs, stats

    def _render(self, messages: list[dict], tools: list | None = None) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )

    # -- Phase 0: validation -------------------------------------------------------------------

    @modal.method()
    def smoke(self) -> dict:
        """CALL / ANSWER / CLARIFY / UNSUPPORTED plus special-token integrity."""
        weather = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            },
        }
        cases = [
            ("CALL", [{"role": "user", "content": "What is the weather in Manila? Use the tool."}], [weather]),
            ("ANSWER", [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}], None),
            ("CLARIFY", [{"role": "user", "content": "Book it for me."}], [weather]),
            ("UNSUPPORTED", [{"role": "user", "content": "Launch a rocket to Mars right now."}], [weather]),
        ]
        prompts = [self._render(m, t) for _, m, t in cases]

        # Special-token integrity: the rendered prompt must start with exactly one <|im_start|>
        im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        integrity = []
        for (label, _, _), prompt in zip(cases, prompts):
            ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            integrity.append({
                "case": label,
                "first_token": ids[0],
                "first_is_im_start": ids[0] == im_start,
                "leading_duplicate_im_start": ids[:2] == [im_start, im_start],
                "prompt_tokens": len(ids),
            })

        outputs, stats = self._generate(prompts)
        results = [
            {
                "case": label,
                "prompt": prompt,
                "raw": out.outputs[0].text,
                "output_tokens": len(out.outputs[0].token_ids),
                "finish_reason": out.outputs[0].finish_reason,
            }
            for (label, _, _), prompt, out in zip(cases, prompts, outputs)
        ]
        return self._persist("smoke", {
            "phase": "validate",
            "environment": self.environment,
            "special_token_integrity": integrity,
            "double_bos_detected": any(i["leading_duplicate_im_start"] for i in integrity),
            "results": results,
            "stats": stats,
        })

    # -- Phase 0.5: saturation sweep -----------------------------------------------------------

    @modal.method()
    def sweep(self, sizes: list[int]) -> dict:
        """Throughput vs submitted concurrency, using real frozen prompts.

        Kept deliberately short: this is a tuning measurement, and every second of it is paid for
        out of the same budget as the evidence.
        """
        rows = _load_frozen("confirmatory")
        records = []
        for size in sizes:
            batch = [r["prompt"] for r in rows[:size]]
            _, stats = self._generate(batch, max_tokens=128)
            stats["concurrency"] = size
            records.append(stats)
            print(f"  concurrency {size}: {stats['total_tokens_per_second']} tok/s "
                  f"({stats['elapsed_seconds']}s)", flush=True)
        best = max(records, key=lambda r: r["total_tokens_per_second"] or 0)
        plateau = next(
            (r for r in records
             if (r["total_tokens_per_second"] or 0) >= 0.97 * (best["total_tokens_per_second"] or 1)),
            best,
        )
        return self._persist("sweep", {
            "phase": "sweep",
            "environment": self.environment,
            "records": records,
            "max_throughput": best,
            "selected_concurrency": plateau["concurrency"],
            "selection_rule": "lowest concurrency reaching >=97% of peak total tokens/sec",
        })

    # -- Phase 1: frozen confirmatory set (regenerates the missing per-example vLLM predictions) --

    @modal.method()
    def frozen(self, partition: str = "confirmatory") -> dict:
        rows = _load_frozen(partition)
        prompts = [r["prompt"] for r in rows]
        outputs, stats = self._generate(prompts)
        generations = [
            {
                "example_id": r["example_id"],
                "raw": out.outputs[0].text,
                "truncated": out.outputs[0].finish_reason == "length",
                "prompt_tokens": len(out.prompt_token_ids),
                "output_tokens": len(out.outputs[0].token_ids),
            }
            for r, out in zip(rows, outputs)
        ]
        submitted = {r["example_id"] for r in rows}
        returned = {g["example_id"] for g in generations}
        if submitted != returned:
            raise RuntimeError(f"reconciliation failed: {len(submitted)} vs {len(returned)}")

        out_dir = f"{VOL}/h200/generations"
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/vllm-bf16.{partition}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.writelines(json.dumps(g, ensure_ascii=False, sort_keys=True) + "\n" for g in generations)
        volume.commit()
        return self._persist("frozen", {
            "phase": "frozen",
            "partition": partition,
            "environment": self.environment,
            "submitted": len(submitted),
            "returned": len(returned),
            "output": path,
            "stats": stats,
        })

    # -- Phase 2: OpenWeights ParitySuite -------------------------------------------------------

    @modal.method()
    def parity(self) -> dict:
        """The seven OpenWeights cases, multi-turn ones driven sequentially."""
        spec = json.loads(Path("/data/openweights_parity_cases_v1.json").read_text(encoding="utf-8"))
        tool = spec["weather_tool"]
        max_tokens = spec["params"]["max_tokens"]
        results = []

        for case in spec["cases"]:
            turns = case["turns"]
            tools = [tool] if case.get("tools") else None
            history: list[dict] = []
            raw = ""
            per_turn = []
            for turn in turns:
                if turn["role"] == "assistant_raw_placeholder":
                    history.append({"role": "assistant", "content": raw})
                    continue
                if turn["role"] == "tool_result":
                    history.append({
                        "role": "tool", "name": turn["name"], "content": turn["content"],
                    })
                    # OpenWeights regenerates immediately after the tool result, with no further
                    # user message, so the generation is driven here rather than by a user turn.
                    prompt = self._render(history, tools)
                    outs, stats = self._generate([prompt], max_tokens=max_tokens)
                    raw = outs[0].outputs[0].text
                    per_turn.append({
                        "prompt_tokens": len(outs[0].prompt_token_ids),
                        "output_tokens": len(outs[0].outputs[0].token_ids),
                        "elapsed_seconds": stats["elapsed_seconds"],
                    })
                    continue
                if turn["role"] != "user":
                    continue  # e.g. the tool-result case has no trailing user message
                history.append({"role": "user", "content": turn["content"]})
                prompt = self._render(history, tools)
                outs, stats = self._generate([prompt], max_tokens=max_tokens)
                raw = outs[0].outputs[0].text
                per_turn.append({
                    "prompt_tokens": len(outs[0].prompt_token_ids),
                    "output_tokens": len(outs[0].outputs[0].token_ids),
                    "elapsed_seconds": stats["elapsed_seconds"],
                })
            results.append({
                "id": case["id"],
                "raw": raw,
                "turns": per_turn,
            })
        return self._persist("parity", {
            "phase": "parity",
            "environment": self.environment,
            "suite": spec["suite"],
            "params": spec["params"],
            "results": results,
        })


    @modal.method()
    def capability_suite(self, sweep_sizes: list[int], partition: str = "confirmatory") -> dict:
        """Sweep + frozen partition + OpenWeights parity in ONE container.

        Cold start is ~94s of model load plus FlashInfer JIT. Paying that three times to run three
        phases would cost more than the phases themselves, so they share a single engine. Mode B
        (performance) deliberately stays separate: it needs exclusive GPU access to be valid.
        """
        out = {"phase": "capability_suite"}
        out["sweep"] = self.sweep.local(sweep_sizes)
        out["frozen"] = self.frozen.local(partition)
        out["parity"] = self.parity.local()
        return self._persist("capability_suite", out)

    # -- Phase 5: performance, exclusive ---------------------------------------------------------

    @modal.method()
    def perf(self, concurrencies: list[int]) -> dict:
        """Mode B. Nothing else may touch the GPU while this runs."""
        rows = _load_frozen("confirmatory")
        rows_sorted = sorted(rows, key=lambda r: r["input_tokens"])
        short = rows_sorted[0]["prompt"]
        median = rows_sorted[len(rows_sorted) // 2]["prompt"]
        long = rows_sorted[-1]["prompt"]

        records = []
        for label, prompt in (("short", short), ("median", median), ("long", long)):
            # Single-stream TTFT proxy and decode rate.
            outs, stats = self._generate([prompt], max_tokens=128)
            records.append({
                "shape": label,
                "concurrency": 1,
                "prompt_tokens": len(outs[0].prompt_token_ids),
                "output_tokens": len(outs[0].outputs[0].token_ids),
                **stats,
            })
        for c in concurrencies:
            batch = [r["prompt"] for r in rows_sorted[:c]]
            _, stats = self._generate(batch, max_tokens=128)
            stats["shape"] = "mixed"
            stats["concurrency"] = c
            records.append(stats)

        import torch

        return self._persist("perf", {
            "phase": "perf",
            "environment": self.environment,
            "exclusive": True,
            "cold_start_model_load_seconds": self.load_seconds,
            "peak_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_memory_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "records": records,
        })


def _driver_version() -> str | None:
    try:
        import subprocess

        return subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance records None when nvidia-smi is unavailable
        return None


def _pkg_version(name: str) -> str | None:
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:  # noqa: BLE001 - provenance records None rather than aborting the run
        return None


def _load_frozen(partition: str) -> list[dict]:
    rows = []
    with open("/data/frozen_prompts_v1.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["partition"] == partition:
                rows.append(row)
    rows.sort(key=lambda r: r["example_id"])
    return rows


@app.local_entrypoint()
def main(phase: str = "validate", partition: str = "confirmatory", sizes: str = "32,64,128,256"):
    out = ROOT / "results/benchmarks/h200"
    out.mkdir(parents=True, exist_ok=True)
    engine = Engine()

    started = time.time()
    if phase == "validate":
        result = engine.smoke.remote()
    elif phase == "sweep":
        result = engine.sweep.remote([int(s) for s in sizes.split(",")])
    elif phase == "frozen":
        result = engine.frozen.remote(partition)
    elif phase == "parity":
        result = engine.parity.remote()
    elif phase == "capability":
        result = engine.capability_suite.remote([int(s) for s in sizes.split(",")], partition)
    elif phase == "perf":
        result = engine.perf.remote([int(s) for s in sizes.split(",")])
    else:
        raise SystemExit(f"unknown phase: {phase}")

    wall = time.time() - started
    result["local_wall_seconds"] = round(wall, 2)
    result["estimated_billed_gpu_usd_upper_bound"] = round(wall / 3600 * GPU_HOURLY_USD, 4)

    path = out / f"{phase}_{partition if phase == 'frozen' else 'run'}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print(f"wall {wall:.0f}s  upper-bound GPU cost ${result['estimated_billed_gpu_usd_upper_bound']}")
