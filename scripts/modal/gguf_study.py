#!/usr/bin/env python3
"""GGUF/llama.cpp deployment branch of the M1-v2 quantization study, on a Modal A100.

Everything the study needs from a GPU lives here: build llama.cpp at a pinned tag, convert the
promoted BF16 checkpoint, prove runtime parity, compute an importance matrix from training-side
text, walk the quantization ladder, and generate on the frozen confirmatory prompts. Scoring and
the preservation gate deliberately stay on the host, in OpenGrad's own evaluator, so a runtime can
never be the thing that decides whether it passed.

Two contract details are easy to get wrong and are enforced rather than assumed:

* The prompts are rendered once by OpenGrad's pinned Qwen renderer and sent as raw text to
  ``/completion``. llama.cpp never re-applies a chat template here. Its built-in tool-call parser
  does not recognise Qwen3.5's XML emission anyway (see opengrad.formatting.parser), so the raw
  text comes home and OpenGrad's parser reads it.
* ``llama-imatrix`` defaults ``parse_special`` to false, which would tokenize ``<|im_start|>`` as
  literal characters. ``--parse-special`` is passed explicitly.

Usage (from the repository root):

    modal run scripts/modal/gguf_study.py --stage prepare
    modal run scripts/modal/gguf_study.py --stage parity
    modal run scripts/modal/gguf_study.py --stage imatrix
    modal run scripts/modal/gguf_study.py --stage ladder
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------------------------
# Pins. Every one of these ends up in the artifact provenance.
# ---------------------------------------------------------------------------------------------

LLAMA_CPP_TAG = "b10919"
CUDA_VERSION = "12.8.1"
CUDA_ARCH = "80"  # A100 = sm_80; narrowing this cuts the build from ~40min to ~12min.

MODEL_REPO = "arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2"
MODEL_REVISION = "f33d20308982f37deb459076f489e794d5521ee3"
# The repo publishes four DPO checkpoints (30/60/90/120) as subdirectories. Only checkpoint 30 was
# promoted, so the download is restricted to it: pulling the repo root would fetch ~15GB of
# checkpoints this study must not measure and hand the converter an ambiguous directory.
CHECKPOINT = "dpo-checkpoint-30"
# Verified on the host from .workspace/quantization/source/m1-v2/dpo-checkpoint-30/model.safetensors
EXPECTED_WEIGHT_SHA256 = "903f9b115d418d1988c0d5e1dd4bda7e04809a54a6bc7e34431982fbee2262f6"
EXPECTED_WEIGHT_BYTES = 3763692048
# config.json is pinned too, not just the weights. The conversion path reads hyperparameters from
# it, so an edited config produces a different model from identical weights — and this study had
# exactly that bug, where the script itself rewrote the file on the cached volume.
EXPECTED_CONFIG_SHA256 = "88bf86c270d616198909ed1eefef8d8c21ac1fa13f62e947f20f8e1ebd02c211"

# The frozen measurement, restated from configs/evaluation/tool_calling/qwen35_2b_baseline.yaml.
MAX_NEW_TOKENS = 512
SLOT_CONTEXT = 5760
# Ordered by increasing bits per weight. The study mandates Q4_K_M/Q5_K_M/Q6_K/Q8_0 "at minimum";
# the rungs below Q4 are what actually answer RQ3 ("the lowest-bit GGUF that preserves >=99%"),
# because a ladder whose smallest rung passes cannot tell you where the floor is. Most of the
# sub-Q4 formats are expected to fail the gate — that is the finding, not a wasted run.
#
# IQ4_XS and IQ3_M are included because they are imatrix-dependent and this study computed a real
# importance matrix; IQ4_XS in particular is usually smaller than Q4_K_M at comparable quality, so
# omitting it would bias "smallest passing format wins" toward the wrong answer.
LADDER = (
    "Q2_K",
    "Q3_K_M",
    "IQ3_M",
    "IQ4_XS",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
)

VOLUME = "opengrad-quant"


def _repo_root() -> Path:
    """Repository root locally; a harmless stand-in inside the container.

    This module is imported in both places. Locally it resolves to the repo so the image can mount
    the study inputs; in the container the file lands at /root/<name>.py, where parents[2] does not
    exist and an unguarded index crash-loops every function before it runs.
    """
    here = Path(__file__).resolve()
    return here.parents[2] if len(here.parents) >= 3 else here.parent


ROOT = _repo_root()

LLAMA = "/opt/llama.cpp"
BIN = f"{LLAMA}/build/bin"
VOL = "/vol"

app = modal.App("opengrad-gguf-study")
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands(
        f"git clone --depth 1 --branch {LLAMA_CPP_TAG} "
        f"https://github.com/ggml-org/llama.cpp {LLAMA}",
        # The build container has no GPU, so there is no real libcuda.so and the CUDA *driver* API
        # symbols (cuMemCreate, cuGetErrorString, ...) do not resolve at link time.
        #
        # Adding -L<stubs> is not enough, which is how the previous two attempts failed: CMake's
        # FindCUDAToolkit searches lib64 for `cuda_driver`, does not find it, so never populates
        # CUDA::cuda_driver and never emits -lcuda at all. A library search path cannot help with a
        # library that was never requested. Putting the stub where the toolkit already looks makes
        # it resolve normally.
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so",
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1",
        f"cmake -S {LLAMA} -B {LLAMA}/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON "
        f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH} -DLLAMA_CURL=ON "
        "-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF",
        f"cmake --build {LLAMA}/build --config Release -j $(nproc) "
        "--target llama-server llama-quantize llama-imatrix llama-bench llama-tokenize",
        # Remove the stub from the final image. At run time the real driver is injected by the
        # GPU runtime; a leftover stub on the loader path would shadow it and every CUDA call
        # would fail with "not supported" at the worst possible moment.
        "rm -f /usr/local/cuda/lib64/libcuda.so /usr/local/cuda/lib64/libcuda.so.1",
        f"ls -la {BIN}",
    )
    .pip_install(
        "torch==2.10.0",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "numpy",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "transformers==5.14.1",
        "huggingface_hub",
        "requests",
        "pyyaml",
    )
    # The requantization guard runs from OpenGrad's own source rather than a copy of its logic, so
    # the rule the tests cover is literally the rule the quantizer enforces.
    .add_local_dir(ROOT / "src/opengrad", "/opt/opengrad/src/opengrad")
    .add_local_file(
        ROOT / "results/quantization/frozen_prompts_v1.jsonl",
        "/data/frozen_prompts_v1.jsonl",
    )
    .add_local_file(
        ROOT / "manifests/quantization/m1_v2_imatrix_calibration_v2.txt",
        "/data/imatrix_calibration.txt",
    )
)


# ---------------------------------------------------------------------------------------------
# Helpers that run inside the container
# ---------------------------------------------------------------------------------------------


def _run(command: list[str], **kwargs) -> str:
    print("+", " ".join(str(part) for part in command), flush=True)
    result = subprocess.run(command, text=True, capture_output=True, **kwargs)
    if result.returncode != 0:
        print(result.stdout[-8000:], flush=True)
        print(result.stderr[-8000:], flush=True)
        raise RuntimeError(f"command failed ({result.returncode}): {command[0]}")
    return result.stdout


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gguf_metadata(path: str, needles: tuple[str, ...] = ()) -> dict:
    """Read key/value metadata back out of a written GGUF.

    Used to verify that a conversion flag had the effect it advertises. `--no-mtp` is supposed to
    leave `block_count` at `num_hidden_layers` and emit no `nextn` key; trusting that without
    looking would reintroduce exactly the class of error this study keeps finding, where a
    plausible command is assumed to have done what its name says.
    """
    sys.path.insert(0, f"{LLAMA}/gguf-py")
    from gguf import GGUFReader  # noqa: PLC0415  (only importable inside the container)

    reader = GGUFReader(path, "r")
    out: dict = {"keys_matched": {}}
    for key, field in reader.fields.items():
        lowered = key.lower()
        if not needles or any(needle.lower() in lowered for needle in needles):
            try:
                out["keys_matched"][key] = field.contents()
            except Exception as exc:  # metadata shape varies by type; the key's presence is data
                out["keys_matched"][key] = f"<unreadable: {type(exc).__name__}: {exc}>"
    out["tensor_count"] = len(reader.tensors)
    out["nextn_key_present"] = any("nextn" in k.lower() for k in reader.fields)
    return out


def _load_prompts(partition: str = "confirmatory") -> list[dict]:
    rows = []
    with open("/data/frozen_prompts_v1.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["partition"] == partition:
                rows.append(row)
    rows.sort(key=lambda row: row["example_id"])
    return rows


class Server:
    """llama-server lifecycle, so every stage talks to the engine the same way."""

    def __init__(self, model: str, *, n_parallel: int, gpu_layers: int = 999, port: int = 8080):
        self.model = model
        self.n_parallel = n_parallel
        self.gpu_layers = gpu_layers
        self.port = port
        self.process: subprocess.Popen | None = None
        self.log_path = f"/tmp/llama-server-{port}.log"
        self._log = None

    def __enter__(self):
        command = [
            f"{BIN}/llama-server",
            "-m", self.model,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-ngl", str(self.gpu_layers),
            # Per-slot context must cover the longest frozen prompt (5235 tokens) plus the
            # 512-token completion budget; llama-server divides --ctx-size across slots.
            "--ctx-size", str(SLOT_CONTEXT * self.n_parallel),
            "--parallel", str(self.n_parallel),
            "--no-webui",
            "--seed", "0",
        ]
        print("+", " ".join(command), flush=True)
        # Log to a FILE, not a pipe. llama-server writes a few lines per request, and an unread
        # subprocess.PIPE holds only ~64KB before the OS blocks the writer — which deadlocks the
        # server partway through a long run, with no error and no exit. That is exactly what
        # happened on the first 1277-prompt attempt: the server answered several hundred requests,
        # filled the pipe, and stopped responding forever.
        self._log = open(self.log_path, "w+", encoding="utf-8", errors="replace")
        self.process = subprocess.Popen(
            command, stdout=self._log, stderr=subprocess.STDOUT, text=True
        )
        self._await_ready()
        return self

    def read_log(self, tail: int = 8000) -> str:
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as handle:
                return handle.read()[-tail:]
        except OSError:
            return ""

    def _await_ready(self, timeout: int = 900) -> None:
        import requests

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited early:\n{self.read_log()}")
            try:
                response = requests.get(f"http://127.0.0.1:{self.port}/health", timeout=5)
                if response.status_code == 200:
                    print(f"llama-server ready after {int(timeout - (deadline - time.time()))}s")
                    return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError(f"llama-server did not become ready:\n{self.read_log()}")

    def __exit__(self, *exc) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self._log is not None:
            self._log.close()


# ---------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------


def _reconcile_mtp_declaration(source: str) -> dict:
    """Decide how the converter must be told to treat the absent multi-token-prediction head.

    The promoted checkpoint declares `mtp_num_hidden_layers: 1` but contains no `mtp.*` tensors:
    both trainers load through `AutoModelForCausalLM`, which instantiates the text-only causal LM
    and never builds the multi-token-prediction head, so it was never trained and never saved. The
    config field was inherited from the multimodal base repo.

    That inconsistency is not cosmetic. `Qwen3_5TextModel` inherits `_QwenMtpMixin` (via
    `_LinearAttentionVReorderBase` -> `Qwen3NextModel`), whose `__init__` does:

        self.block_count = self.hparams["num_hidden_layers"]
        if not self.no_mtp:
            n_mtp = self.hparams.get("mtp_num_hidden_layers", 0)
            if n_mtp == 0:
                assert self.opt_num_mtp_layers != 0
            self.block_count += n_mtp

    So converting as-is extends `block_count` to 25 and emits `nextn_predict_layers: 1` for a block
    whose weights do not exist.

    This function only *diagnoses*. An earlier revision deleted the two MTP keys from the
    checkpoint's `config.json`, which was wrong twice over: it mutated an artifact this study
    declares immutable, and it traded one failure for another — with the key gone, `n_mtp` is 0,
    `opt_num_mtp_layers` is only ever populated from `mtp.*` tensors found during indexing, and the
    assertion above fires. The remedy is the converter's own supported switch, `--no-mtp`
    (`--no-nextn`), which skips that branch entirely and leaves `block_count` at 24. The caller
    applies it; the checkpoint is left byte-identical.

    Grafting the base repo's MTP tensors is the other conceivable route and is deliberately not
    taken: a head trained against the base backbone, bolted onto one that has since moved through
    2400 SFT and 30 DPO steps, is a *different model*. That belongs in a separate, measured
    experiment, not silently inside the artifact under test.
    """
    import json as _json

    from safetensors import safe_open

    with open(f"{source}/config.json", encoding="utf-8") as handle:
        config = _json.load(handle)

    with safe_open(f"{source}/model.safetensors", framework="pt") as weights:
        mtp_tensors = sorted(key for key in weights.keys() if key.lower().startswith("mtp."))

    declared = config.get("mtp_num_hidden_layers")
    record = {
        "declared_mtp_num_hidden_layers": declared,
        "mtp_tensors_present": len(mtp_tensors),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "config_mutated": False,
    }
    if declared and not mtp_tensors:
        record["action"] = "convert_with_--no-mtp"
        record["converter_flag"] = "--no-mtp"
        record["expected_block_count"] = config.get("num_hidden_layers")
        record["reason"] = (
            "config declares an MTP head the checkpoint does not contain; --no-mtp makes the "
            "converter skip the nextn block instead of writing one with no weights"
        )
    elif mtp_tensors:
        record["action"] = "convert_with_mtp"
        record["converter_flag"] = None
        record["tensors"] = mtp_tensors[:20]
    else:
        record["action"] = "no_mtp_declared_and_none_present"
        record["converter_flag"] = "--no-mtp"
    print(f"MTP reconciliation: {record}", flush=True)
    return record


@app.function(image=image, volumes={VOL: volume}, timeout=60 * 60, cpu=8.0, memory=32768)
def prepare() -> dict:
    """Fetch the promoted checkpoint, verify its bytes, and convert it to BF16 GGUF.

    BF16 rather than F16 on purpose: the source weights are BF16, so an F16 hop would introduce a
    rounding step that has nothing to do with the quantization being studied and would pollute the
    parity measurement it is meant to establish.
    """
    from huggingface_hub import snapshot_download

    root = f"{VOL}/source/m1-v2"
    source = f"{root}/{CHECKPOINT}"
    os.makedirs(f"{VOL}/gguf", exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=root,
        allow_patterns=[f"{CHECKPOINT}/*"],
    )

    weights = f"{source}/model.safetensors"
    observed_bytes = os.path.getsize(weights)
    observed_sha = _sha256(weights)
    if observed_sha != EXPECTED_WEIGHT_SHA256 or observed_bytes != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError(
            "downloaded weights are not the promoted checkpoint:\n"
            f"  sha256 {observed_sha} (expected {EXPECTED_WEIGHT_SHA256})\n"
            f"  bytes  {observed_bytes} (expected {EXPECTED_WEIGHT_BYTES})"
        )

    # The checkpoint is declared immutable, so that is enforced rather than trusted. An earlier
    # revision of this script rewrote config.json in place on this very volume, which is exactly
    # the failure this check exists to catch — a cached, silently-edited source would otherwise
    # convert cleanly and produce an artifact that no longer descends from the promoted model.
    config_path = f"{source}/config.json"
    if _sha256(config_path) != EXPECTED_CONFIG_SHA256:
        print("config.json does not match the promoted checkpoint; re-fetching", flush=True)
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            filename=f"{CHECKPOINT}/config.json",
            local_dir=root,
            force_download=True,
        )
        restored = _sha256(config_path)
        if restored != EXPECTED_CONFIG_SHA256:
            raise RuntimeError(
                f"config.json still differs after re-fetch: {restored} "
                f"(expected {EXPECTED_CONFIG_SHA256})"
            )

    mtp = _reconcile_mtp_declaration(source)

    target = f"{VOL}/gguf/m1-v2-bf16.gguf"
    command = [
        "python", f"{LLAMA}/convert_hf_to_gguf.py",
        source,
        "--outfile", target,
        "--outtype", "bf16",
    ]
    if mtp.get("converter_flag"):
        command.append(mtp["converter_flag"])
    _run(command)
    # Commit before verifying, not after. The conversion is the expensive step; a bug in the
    # verification below must not be able to throw away a GGUF that was written successfully,
    # which is exactly what happened when this helper raised a NameError on its first run.
    volume.commit()

    # The whole point of --no-mtp is that block_count stays at num_hidden_layers and no nextn key
    # is written. Read it back off the artifact instead of assuming the flag did what it says.
    block_count = _gguf_metadata(target, ("qwen35.block_count", "nextn"))

    llama_commit = _run(["git", "-C", LLAMA, "rev-parse", "HEAD"]).strip()
    return {
        "stage": "prepare",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "checkpoint": CHECKPOINT,
        "source_dir": source,
        "source_weight_sha256": observed_sha,
        "source_weight_bytes": observed_bytes,
        "gguf": target,
        "gguf_sha256": _sha256(target),
        "gguf_bytes": os.path.getsize(target),
        "outtype": "bf16",
        "llama_cpp_tag": LLAMA_CPP_TAG,
        "llama_cpp_commit": llama_commit,
        "mtp_reconciliation": mtp,
        "converter_command": " ".join(command),
        "gguf_metadata": block_count,
        "source_config_sha256": EXPECTED_CONFIG_SHA256,
    }


# Tokenization needs no GPU, but the binary does: llama.cpp is built with the CUDA backend, so
# llama-server links libcuda.so.1 and will not even load without it. The build-time stub cannot be
# left in /usr/local/cuda/lib64 to satisfy that, because this image puts that directory ahead of
# the driver on LD_LIBRARY_PATH — the stub would shadow the real driver on every GPU stage. So the
# binary-running stages get a GPU, and the stub stays deleted.
@app.function(
    image=image, volumes={VOL: volume}, gpu="A100-80GB", timeout=60 * 60, cpu=8.0, memory=32768
)
def tokenizer_parity() -> dict:
    """Compare llama.cpp's tokenization of every frozen prompt against the HF tokenizer.

    This is the cheapest place to catch the parity break that would otherwise be misread as
    quantization damage: a BOS token llama.cpp inserts and Transformers does not. The pinned
    tokenizer declares add_bos_token=False and bos_token=None, so a leading token that is not in
    the HF sequence is a hard failure, not a rounding difference.
    """
    import requests
    from transformers import AutoTokenizer

    prompts = _load_prompts("confirmatory")
    tokenizer = AutoTokenizer.from_pretrained(f"{VOL}/source/m1-v2/{CHECKPOINT}")

    # The sample list is capped for readability; the counters are not, so the reported totals are
    # the real totals rather than the size of the sample.
    sample: list[dict] = []
    mismatch_count = 0
    bos_insertions = 0
    checked = 0
    with Server(f"{VOL}/gguf/m1-v2-bf16.gguf", n_parallel=1, gpu_layers=0) as server:
        session = requests.Session()
        for row in prompts:
            expected = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
            response = session.post(
                f"http://127.0.0.1:{server.port}/tokenize",
                json={"content": row["prompt"]},
                timeout=120,
            )
            response.raise_for_status()
            observed = response.json()["tokens"]
            checked += 1
            if observed == expected:
                continue
            mismatch_count += 1
            leading_extra = observed[1:] == expected
            if leading_extra:
                bos_insertions += 1
            if len(sample) < 10:
                sample.append(
                    {
                        "example_id": row["example_id"],
                        "expected_len": len(expected),
                        "observed_len": len(observed),
                        "expected_head": expected[:8],
                        "observed_head": observed[:8],
                        "leading_extra_token": observed[0] if leading_extra else None,
                    }
                )

    result = {
        "stage": "tokenizer_parity",
        "checked": checked,
        "exact_matches": checked - mismatch_count,
        "mismatches": mismatch_count,
        "mismatch_sample": sample,
        "bos_insertions": bos_insertions,
        "passed": mismatch_count == 0,
    }
    os.makedirs(f"{VOL}/results", exist_ok=True)
    with open(f"{VOL}/results/tokenizer_parity.json", "w") as handle:
        json.dump(result, handle, indent=2)
    volume.commit()
    return result


@app.function(
    image=image, volumes={VOL: volume}, gpu="A100-80GB", timeout=60 * 90, cpu=8.0, memory=65536
)
def imatrix() -> dict:
    """Compute the importance matrix from the frozen training-side calibration corpus."""
    os.makedirs(f"{VOL}/imatrix", exist_ok=True)
    target = f"{VOL}/imatrix/m1-v2.imatrix"
    output = _run(
        [
            f"{BIN}/llama-imatrix",
            "-m", f"{VOL}/gguf/m1-v2-bf16.gguf",
            "-f", "/data/imatrix_calibration.txt",
            "-o", target,
            "--parse-special",
            "-ngl", "999",
            "-c", "512",
            "-b", "512",
        ]
    )
    volume.commit()
    return {
        "stage": "imatrix",
        "imatrix": target,
        "imatrix_sha256": _sha256(target),
        "imatrix_bytes": os.path.getsize(target),
        "calibration_sha256": _sha256("/data/imatrix_calibration.txt"),
        "parse_special": True,
        "tail": output[-2000:],
    }


# Same reason as tokenizer_parity: llama-quantize is CPU work, but the binary will not load
# without the CUDA driver library it was linked against.
@app.function(
    image=image, volumes={VOL: volume}, gpu="A100-80GB", timeout=60 * 90, cpu=16.0, memory=65536
)
def quantize(quant_type: str) -> dict:
    """Quantize from the BF16 GGUF directly. Never from an already-quantized source."""
    import sys

    sys.path.insert(0, "/opt/opengrad/src")
    from opengrad.promotion.artifacts import assert_quantizable_source

    if quant_type not in LADDER:
        raise ValueError(f"unsupported quantization type: {quant_type}")
    source = f"{VOL}/gguf/m1-v2-bf16.gguf"
    target = f"{VOL}/gguf/m1-v2-{quant_type}.gguf"
    # Cheap to get wrong and invisible afterwards: Q4_K_M produced from Q8_0 looks identical in a
    # directory listing but carries two roundings instead of one.
    assert_quantizable_source(os.path.basename(source), source_type="bf16")
    imatrix_path = f"{VOL}/imatrix/m1-v2.imatrix"
    if not os.path.exists(imatrix_path):
        raise RuntimeError("importance matrix is missing; run the imatrix stage first")

    command = [
        f"{BIN}/llama-quantize",
        "--imatrix", imatrix_path,
        source,
        target,
        quant_type,
        "16",
    ]
    started = time.time()
    output = _run(command)
    volume.commit()
    return {
        "stage": "quantize",
        "quant_type": quant_type,
        "source_gguf": source,
        "source_sha256": _sha256(source),
        "artifact": target,
        "artifact_sha256": _sha256(target),
        "artifact_bytes": os.path.getsize(target),
        "imatrix_sha256": _sha256(imatrix_path),
        "command": " ".join(command),
        "elapsed_seconds": round(time.time() - started, 3),
        "tail": output[-2000:],
    }


@app.function(
    image=image, volumes={VOL: volume}, gpu="A100-80GB", timeout=60 * 180, cpu=16.0, memory=65536
)
def generate(artifact: str, partition: str = "confirmatory", n_parallel: int = 8) -> dict:
    """Generate on every frozen prompt and bring the raw text home unscored.

    Concurrency is bounded by ``n_parallel`` slots, and every submitted example_id is reconciled
    against the returned set before anything is written: a runtime that quietly answers fewer
    prompts than it was given would otherwise report a better score for having answered less.
    """
    import concurrent.futures

    import requests

    model = f"{VOL}/gguf/{artifact}"
    if not os.path.exists(model):
        raise RuntimeError(f"missing artifact: {model}")
    prompts = _load_prompts(partition)

    results: dict[str, dict] = {}
    started = time.time()
    with Server(model, n_parallel=n_parallel) as server:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=n_parallel * 2, pool_maxsize=n_parallel * 2
        )
        session.mount("http://", adapter)

        def one(row: dict) -> tuple[str, dict]:
            payload = {
                "prompt": row["prompt"],
                "n_predict": MAX_NEW_TOKENS,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "seed": 0,
                # Prefix reuse across requests would make a result depend on submission order.
                "cache_prompt": False,
            }
            try:
                response = session.post(
                    f"http://127.0.0.1:{server.port}/completion", json=payload, timeout=1800
                )
                response.raise_for_status()
                body = response.json()
                return row["example_id"], {
                    "example_id": row["example_id"],
                    "raw": body.get("content", ""),
                    "truncated": bool(body.get("stopped_limit", False)),
                    "timings": body.get("timings", {}),
                    "tokens_predicted": body.get("tokens_predicted"),
                    "tokens_evaluated": body.get("tokens_evaluated"),
                }
            except Exception as exc:  # surfaced, never silently dropped
                return row["example_id"], {
                    "example_id": row["example_id"],
                    "raw": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as pool:
            for index, (example_id, payload) in enumerate(pool.map(one, prompts), start=1):
                results[example_id] = payload
                if index % 200 == 0:
                    print(f"  {index}/{len(prompts)} ({time.time() - started:.0f}s)", flush=True)

    submitted = {row["example_id"] for row in prompts}
    if set(results) != submitted:
        raise RuntimeError(
            f"generation set does not reconcile: submitted {len(submitted)}, "
            f"returned {len(results)}, missing {sorted(submitted - set(results))[:10]}"
        )

    os.makedirs(f"{VOL}/generations", exist_ok=True)
    stem = artifact.replace(".gguf", "")
    destination = f"{VOL}/generations/{stem}.{partition}.jsonl"
    with open(destination, "w", encoding="utf-8", newline="\n") as handle:
        for example_id in sorted(results):
            handle.write(json.dumps(results[example_id], ensure_ascii=False, sort_keys=True) + "\n")
    volume.commit()

    errors = [row for row in results.values() if row.get("error")]
    return {
        "stage": "generate",
        "artifact": artifact,
        "partition": partition,
        "submitted": len(submitted),
        "returned": len(results),
        "errors": len(errors),
        "error_sample": errors[:5],
        "output": destination,
        "elapsed_seconds": round(time.time() - started, 3),
    }


@app.function(
    image=image, volumes={VOL: volume}, gpu="A100-80GB", timeout=60 * 60, cpu=8.0, memory=32768
)
def bench(artifact: str) -> dict:
    """Prefill and decode throughput for one artifact, on identical hardware and settings."""
    model = f"{VOL}/gguf/{artifact}"
    output = _run(
        [
            f"{BIN}/llama-bench",
            "-m", model,
            "-p", "512,2048",
            "-n", "128",
            "-ngl", "999",
            "-r", "3",
            "-o", "json",
        ]
    )
    payload = json.loads(output[output.index("[") :])
    return {
        "stage": "bench",
        "artifact": artifact,
        "artifact_bytes": os.path.getsize(model),
        "rows": payload,
    }


@app.function(
    image=image, volumes={VOL: volume}, gpu="A100-80GB", timeout=60 * 90, cpu=8.0, memory=65536
)
def tokenizer_divergence_probe(example_ids: list[str], artifact: str = "m1-v2-bf16.gguf") -> dict:
    """Decide whether the tokenizer divergence changes the evaluated decision.

    The obvious experiment — compare llama.cpp's answer to the vLLM reference — cannot answer this,
    because it changes the engine and the tokenization at the same time. If the decisions differ,
    that design cannot say which one did it.

    So the engine is held constant and only the tokenization is varied: the same llama.cpp server,
    the same BF16 GGUF, the same sampler, asked twice — once with the prompt as text, which
    llama.cpp tokenizes with its own `qwen35` regex, and once with the token ids produced by the
    pinned HF tokenizer, submitted directly. llama-server accepts an array of ids in place of a
    string, so the second call runs the model on exactly the token sequence vLLM would have seen.
    Any difference between those two answers is attributable to tokenization alone.

    A third answer is collected from HF transformers on the same weights, which gives the
    reference-side decision. It is reported as its own column and never conflated with the two
    above: the frozen reference metrics were produced under vLLM 0.29.0, and this is transformers,
    so it is evidence about the checkpoint's behaviour, not a re-run of the reference.
    """
    import requests
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = f"{VOL}/source/m1-v2/{CHECKPOINT}"
    gguf = f"{VOL}/gguf/{artifact}"
    wanted = set(example_ids)
    rows = [row for row in _load_prompts("confirmatory") if row["example_id"] in wanted]
    if len(rows) != len(wanted):
        raise RuntimeError(
            f"probe asked for {len(wanted)} ids, resolved {len(rows)}: "
            f"missing {sorted(wanted - {r['example_id'] for r in rows})}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.add_bos_token:
        raise RuntimeError("pinned tokenizer unexpectedly has add_bos_token=True")

    # Resolved from the tokenizer, never hardcoded: a previous phase of this study was bitten by
    # assuming which id `<|im_start|>` carries, and the whole point here is to assert rather than
    # assume special-token handling.
    special = {
        name: tokenizer.convert_tokens_to_ids(name)
        for name in ("<|im_start|>", "<|im_end|>")
    }
    special["eos_token"] = tokenizer.eos_token
    special["eos_token_id"] = tokenizer.eos_token_id
    special["bos_token"] = tokenizer.bos_token
    special["bos_token_id"] = tokenizer.bos_token_id
    special["add_bos_token"] = bool(getattr(tokenizer, "add_bos_token", False))
    special["add_eos_token"] = bool(getattr(tokenizer, "add_eos_token", False))

    provenance = {
        "gguf_path": gguf,
        "gguf_sha256": _sha256(gguf),
        "gguf_bytes": os.path.getsize(gguf),
        "llama_cpp_tag": LLAMA_CPP_TAG,
        "llama_cpp_commit": _run(["git", "-C", LLAMA, "rev-parse", "HEAD"]).strip(),
        "checkpoint": CHECKPOINT,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "source_file_sha256": {
            name: _sha256(f"{model_path}/{name}")
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                         "chat_template.jinja")
            if os.path.exists(f"{model_path}/{name}")
        },
        "transformers": __import__("transformers").__version__,
        "special_tokens": special,
        "generation": {
            "n_predict": MAX_NEW_TOKENS, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
            "seed": 0, "cache_prompt": False,
        },
    }

    def census(ids: list[int]) -> dict:
        """Count the structural tokens, so an inserted or dropped one cannot pass unnoticed."""
        return {
            "im_start": sum(1 for t in ids if t == special["<|im_start|>"]),
            "im_end": sum(1 for t in ids if t == special["<|im_end|>"]),
            "eos": sum(1 for t in ids if t == special["eos_token_id"]),
            "first": ids[0] if ids else None,
            "last": ids[-1] if ids else None,
        }

    def align(a: list[int], b: list[int]) -> dict:
        """Where the two token streams part company, and whether they ever come back together."""
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if first is None and len(a) == len(b):
            return {"identical": True, "first_divergence": None, "reconverged_suffix": len(a)}
        # Longest common suffix is the honest measure of reconvergence: the tails realign once the
        # divergent span is past, and its length says how localized the damage is.
        suffix = 0
        while suffix < min(len(a), len(b)) and a[-1 - suffix] == b[-1 - suffix]:
            suffix += 1
        return {
            "identical": False,
            "first_divergence": first if first is not None else min(len(a), len(b)),
            "common_prefix": first if first is not None else min(len(a), len(b)),
            "reconverged_suffix": suffix,
            "divergent_span_hf": len(a) - (first or 0) - suffix,
            "divergent_span_llamacpp": len(b) - (first or 0) - suffix,
        }

    out: dict[str, dict] = {}
    for row in rows:
        ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        out[row["example_id"]] = {
            "example_id": row["example_id"],
            "expected_decision": row["expected_decision"],
            "source": row["source"],
            "prompt_text": row["prompt"],
            "prompt_sha256": row["prompt_sha256"],
            "prompt_chars": len(row["prompt"]),
            "hf_token_count": len(ids),
            "hf_token_ids": ids,
            "hf_special_census": census(ids),
        }

    # --- llama.cpp, two tokenizations, everything else identical -----------------------------
    with Server(gguf, n_parallel=1) as server:
        session = requests.Session()
        base = f"http://127.0.0.1:{server.port}"

        def tokenize(text: str) -> list[int]:
            response = session.post(f"{base}/tokenize", json={"content": text}, timeout=300)
            response.raise_for_status()
            return response.json()["tokens"]

        def detokenize(ids: list[int]) -> str:
            response = session.post(f"{base}/detokenize", json={"tokens": ids}, timeout=300)
            response.raise_for_status()
            return response.json()["content"]

        def ask(prompt_field) -> dict:
            payload = {
                "prompt": prompt_field,
                "n_predict": MAX_NEW_TOKENS,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "seed": 0,
                "cache_prompt": False,
            }
            response = session.post(f"{base}/completion", json=payload, timeout=1800)
            response.raise_for_status()
            body = response.json()
            return {
                "raw": body.get("content", ""),
                "truncated": bool(body.get("stopped_limit", False)),
                "tokens_evaluated": body.get("tokens_evaluated"),
                "tokens_predicted": body.get("tokens_predicted"),
                "stop_type": body.get("stop_type"),
            }

        # ---- Prove the direct-token path is literal before trusting any result from it ---------
        # If llama-server prepends a BOS, applies a template, or normalizes the ids, then the
        # "HF token ids" arm is not running HF's tokenization and the whole probe is void. This is
        # the same failure mode as the earlier <|im_start|> BOS collision, so it is tested with a
        # control whose expected token count is known exactly.
        control_ids = out[rows[0]["example_id"]]["hf_token_ids"]
        control = session.post(
            f"{base}/completion",
            json={"prompt": control_ids, "n_predict": 1, "temperature": 0.0, "top_k": 1,
                  "seed": 0, "cache_prompt": False},
            timeout=600,
        )
        control.raise_for_status()
        evaluated = control.json().get("tokens_evaluated")
        direct_token_path = {
            "submitted_token_count": len(control_ids),
            "tokens_evaluated": evaluated,
            "literal": evaluated == len(control_ids),
            "detokenize_roundtrip_matches_prompt":
                detokenize(control_ids) == rows[0]["prompt"],
            "note": (
                "tokens_evaluated must equal the number of ids submitted; a larger value means "
                "the server injected a token (BOS is the usual culprit) and the arm would not be "
                "evaluating HF's tokenization"
            ),
        }
        if not direct_token_path["literal"]:
            raise RuntimeError(
                "llama-server did not evaluate the supplied token ids literally: submitted "
                f"{len(control_ids)}, evaluated {evaluated}. The tokenizer-isolation experiment "
                "is invalid under this condition and must not be interpreted."
            )
        provenance["direct_token_path_validation"] = direct_token_path

        for row in rows:
            record = out[row["example_id"]]
            llamacpp_ids = tokenize(row["prompt"])
            record["llamacpp_token_ids"] = llamacpp_ids
            record["llamacpp_token_count"] = len(llamacpp_ids)
            record["llamacpp_special_census"] = census(llamacpp_ids)
            record["token_alignment"] = align(record["hf_token_ids"], llamacpp_ids)
            record["token_count_delta"] = len(llamacpp_ids) - record["hf_token_count"]
            record["special_token_census_matches"] = (
                record["hf_special_census"]["im_start"]
                == record["llamacpp_special_census"]["im_start"]
                and record["hf_special_census"]["im_end"]
                == record["llamacpp_special_census"]["im_end"]
                and record["hf_special_census"]["first"]
                == record["llamacpp_special_census"]["first"]
            )
            record["llamacpp_own_tokenization"] = ask(row["prompt"])
            record["llamacpp_hf_token_ids"] = ask(record["hf_token_ids"])
            # The text arm must have evaluated llama.cpp's own count, and the id arm HF's count.
            record["arm_token_counts_confirmed"] = (
                record["llamacpp_own_tokenization"]["tokens_evaluated"] == len(llamacpp_ids)
                and record["llamacpp_hf_token_ids"]["tokens_evaluated"]
                == record["hf_token_count"]
            )

    # --- HF transformers on the same weights (SECONDARY cross-check) ---------------------------
    # This arm is not the experiment. The two llama.cpp arms above are, and they have already run
    # by this point. So a failure here is recorded and the primary result is still returned — an
    # optional cross-check must never destroy the measurement it was meant to corroborate.
    #
    # `.to("cuda")` rather than `device_map="cuda"`: device_map routes through accelerate, which is
    # not installed in this image, and installing it would rebuild the image for a secondary arm.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).eval().to("cuda")
        for row in rows:
            record = out[row["example_id"]]
            ids = torch.tensor([record["hf_token_ids"]], device="cuda")
            with torch.no_grad():
                generated = model.generate(
                    ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(generated[0][ids.shape[1]:], skip_special_tokens=True)
            record["hf_transformers"] = {
                "raw": completion,
                "truncated": generated.shape[1] - ids.shape[1] >= MAX_NEW_TOKENS,
                "tokens_predicted": int(generated.shape[1] - ids.shape[1]),
            }
        provenance["hf_cross_check"] = {"ran": True}
    except Exception as exc:
        provenance["hf_cross_check"] = {
            "ran": False,
            "error": f"{type(exc).__name__}: {exc}",
            "impact": (
                "secondary cross-check only; the tokenizer-isolation result from the two "
                "llama.cpp arms is unaffected"
            ),
        }
        print(f"HF cross-check failed (non-fatal): {type(exc).__name__}: {exc}", flush=True)

    return {
        "stage": "tokenizer_divergence_probe",
        "artifact": artifact,
        "checkpoint": CHECKPOINT,
        "provenance": provenance,
        "examples": [out[key] for key in sorted(out)],
        "design": (
            "llamacpp_own_tokenization vs llamacpp_hf_token_ids isolates tokenization with engine, "
            "weights and sampler held constant. hf_transformers is a pinned-weight cross-check "
            "under a different engine and is reported separately; it is NOT a reconstruction of "
            "the frozen vLLM 0.29.0 reference, whose per-example predictions were not preserved."
        ),
    }


@app.function(image=image, volumes={VOL: volume}, timeout=60 * 20, cpu=4.0, memory=16384)
def inspect_tokenizer(artifact: str = "m1-v2-bf16.gguf") -> dict:
    """Read the tokenizer identity the converter wrote into the GGUF.

    The parity run found that all six mismatching prompts are Thai and that llama.cpp emits fewer
    tokens for every one of them. Fewer tokens means coarser splitting, which points at the
    pre-tokenizer regex rather than at the merge table: llama.cpp selects a regex by matching a
    hash of the HF tokenizer, and falls back to a generic `default` split when it recognises none.
    A `default` fallback tokenizes Latin text identically and diverges on scripts where the regex
    is what does the work — which is exactly the observed pattern. This reads the key that decides
    it instead of inferring it.
    """
    model = f"{VOL}/gguf/{artifact}"
    meta = _gguf_metadata(model, ("tokenizer.ggml.pre", "tokenizer.ggml.model", "general.name"))
    return {"stage": "inspect_tokenizer", "artifact": artifact, **meta}


@app.local_entrypoint()
def main(stage: str = "prepare", artifact: str = "", quant_type: str = "", partition: str = "confirmatory"):
    out = ROOT / "results/quantization/gguf"
    out.mkdir(parents=True, exist_ok=True)

    def save(name: str, payload: dict) -> None:
        path = out / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")

    if stage == "prepare":
        save("prepare", prepare.remote())
    elif stage == "parity":
        result = tokenizer_parity.remote()
        save("tokenizer_parity", result)
        print(json.dumps(result, indent=2)[:2000])
    elif stage == "divergence-probe":
        parity = json.loads(
            (out / "tokenizer_parity.json").read_text(encoding="utf-8")
        )
        ids = sorted(m["example_id"] for m in parity["mismatch_sample"])
        if not ids:
            raise SystemExit("no tokenizer mismatches recorded; nothing to probe")
        print(f"probing {len(ids)} mismatching example(s)")
        save("tokenizer_divergence_probe", tokenizer_divergence_probe.remote(ids))
    elif stage == "inspect-tokenizer":
        result = inspect_tokenizer.remote(artifact or "m1-v2-bf16.gguf")
        save("inspect_tokenizer", result)
        print(json.dumps(result, indent=2)[:3000])
    elif stage == "imatrix":
        save("imatrix", imatrix.remote())
    elif stage == "quantize":
        save(f"quantize_{quant_type}", quantize.remote(quant_type))
    elif stage == "generate":
        save(f"generate_{artifact.replace('.gguf', '')}_{partition}", generate.remote(artifact, partition))
    elif stage == "bench":
        save(f"bench_{artifact.replace('.gguf', '')}", bench.remote(artifact))
    elif stage == "ladder":
        for quant in LADDER:
            save(f"quantize_{quant}", quantize.remote(quant))
    else:
        raise SystemExit(f"unknown stage: {stage}")
