#!/usr/bin/env python3
"""ExecuTorch export branch of the M1-v2 quantization study: CPU, Snapdragon, MediaTek.

This produces artifacts; it does not judge them. Behavioural evaluation of the `.pte` files runs
elsewhere (OpenWeights), so nothing here may claim a preservation verdict. Each target reports the
export outcome and its provenance, and the gate is applied later against real generations.

What is known upstream before running anything, and is therefore tested rather than assumed:

* CPU/XNNPACK is the only target with a Qwen3.5 model definition. `examples/models/qwen3_5`
  exists and its `2b_config.json` matches the promoted checkpoint exactly. The bring-up is
  fp32 + static shape, and its README states `q8da4w` for Qwen3.5 is deferred. The README can lag
  the code, so the quantized export is attempted and whatever actually happens is recorded.
* Qualcomm has the plumbing (`backend.qnn`, `pt2e_quantize: qnn_16a4w`) but no Gated DeltaNet
  anywhere under `backends/qualcomm`. The stock export is attempted first so the failure is
  evidence rather than prediction.
* MediaTek has no `export_llm` integration at all and its SDK is registration-gated.

Two deviations from the stock CPU config are required by the frozen measurement and are applied
explicitly rather than silently:

* `max_seq_length`/`max_context_length` 2048 -> 5760. The frozen confirmatory set reaches 5235
  prompt tokens and the completion budget is 512; the stock 2048 would truncate 1277 prompts into
  a different measurement.
* The stock metadata declares `get_bos_id: 248045`, which is `<|im_start|>`. OpenGrad's rendered
  prompts already begin with that token and the pinned tokenizer sets `add_bos_token=False`, so a
  runner that prepends BOS doubles it. Recorded in the handoff rather than patched here, because
  the `.pte` metadata is what the downstream runner reads.

Usage:
    modal run scripts/modal/executorch_export.py --target cpu
    modal run scripts/modal/executorch_export.py --target cpu --quantize 8da4w
    modal run scripts/modal/executorch_export.py --target qnn
    modal run scripts/modal/executorch_export.py --target mediatek-probe
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

EXECUTORCH_COMMIT = "df6147afadf106a0ef3a74f65d80b5c589e63e44"
MODEL_REPO = "arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2"
MODEL_REVISION = "f33d20308982f37deb459076f489e794d5521ee3"
# The repo publishes four DPO checkpoints (30/60/90/120) as subdirectories; only 30 was promoted.
# Restricting the download keeps ~15GB of unmeasured checkpoints out and prevents handing the
# weight converter a directory with four candidates in it.
CHECKPOINT = "dpo-checkpoint-30"
EXPECTED_WEIGHT_SHA256 = "903f9b115d418d1988c0d5e1dd4bda7e04809a54a6bc7e34431982fbee2262f6"

# The frozen measurement window: longest confirmatory prompt 5235 tokens + 512 completion budget.
MAX_CONTEXT = 5760

QNN_SDK_URL = (
    "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
    "Qualcomm_AI_Runtime_Community/All/2.37.0.250724/v2.37.0.250724.zip"
)
QNN_VERSION = "2.37.0.250724"
QNN_SOC = "SM8650"

# MediaTek's NeuroPilot Express portal. Probed, never assumed reachable.
MTK_SDK_URL = "https://neuropilot.mediatek.com/resources/public/npexpress/en/docs/npexpress"

VOL = "/vol"
ET = "/opt/executorch"


def _repo_root() -> Path:
    """Repository root locally; a harmless stand-in inside the container.

    This module is imported in both places. In the container the file lands at /root/<name>.py,
    where parents[2] does not exist, and an unguarded index crash-loops every function.
    """
    here = Path(__file__).resolve()
    return here.parents[2] if len(here.parents) >= 3 else here.parent


ROOT = _repo_root()

app = modal.App("opengrad-executorch-export")
volume = modal.Volume.from_name("opengrad-quant", create_if_missing=True)

base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "wget", "unzip", "build-essential", "cmake", "ninja-build", "zip")
    .run_commands(
        f"git clone https://github.com/pytorch/executorch {ET}",
        f"cd {ET} && git checkout {EXECUTORCH_COMMIT} && git submodule sync && "
        "git submodule update --init --recursive",
    )
    .run_commands(
        f"cd {ET} && ./install_executorch.sh",
        gpu=None,
    )
    .pip_install("huggingface_hub", "safetensors", "transformers==5.14.1")
)

# The SDK probe only needs curl and pip, so it must not pay for the ExecuTorch source build.
probe_image = modal.Image.debian_slim(python_version="3.12").apt_install("curl")

qnn_image = base.apt_install(
    # QNN's x86 libraries are built against LLVM's libc++, not libstdc++. Without it
    # libQnnHtp.so fails to dlopen and the export aborts at backend init — before the
    # partitioner ever sees the model, which makes it look like a model rejection.
    "libc++1",
    "libc++abi1",
).run_commands(
    f"mkdir -p /opt/qnn && cd /opt/qnn && curl -fL --retry 3 -o qnn.zip '{QNN_SDK_URL}' && "
    "unzip -q qnn.zip && rm qnn.zip && ls /opt/qnn",
)


def _run(
    command: list[str] | str,
    *,
    cwd: str | None = None,
    check: bool = True,
    env: dict | None = None,
) -> dict:
    """Run a build step, capturing the outcome instead of only the happy path.

    Failures are data in this study: an export that refuses the hybrid layers is the answer to
    RQ4, so the stderr is returned rather than raised away.
    """
    shell = isinstance(command, str)
    printable = command if shell else " ".join(str(part) for part in command)
    print("+", printable, flush=True)
    started = time.time()
    merged = {**os.environ, **(env or {})}
    result = subprocess.run(
        command, cwd=cwd, shell=shell, text=True, capture_output=True, env=merged, check=False
    )
    # Capture both ends. ExecuTorch dumps the entire lowered graph on a partitioner failure, which
    # is tens of thousands of characters and pushes the actual exception out of a tail-only window.
    # The head holds the error; the tail holds the traceback.
    payload = {
        "command": printable,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout_tail": result.stdout[-12000:],
        "stderr_head": result.stderr[:20000],
        "stderr_tail": result.stderr[-20000:],
        "stderr_bytes": len(result.stderr),
        # Lines that look like an exception, wherever they are in the stream.
        "stderr_exceptions": [
            line.strip()[:500]
            for line in result.stderr.splitlines()
            if re.match(
                r"^\s*(?:\w+\.)*\w*(?:Error|Exception|NotImplementedError|AssertionError)\b[: ]",
                line,
            )
        ][-40:],
    }
    if result.returncode != 0:
        print(result.stdout[-4000:], flush=True)
        print(result.stderr[-4000:], flush=True)
        if check:
            raise RuntimeError(f"command failed ({result.returncode}): {printable}")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict:
    import importlib
    from importlib import metadata

    out = {"executorch_commit": EXECUTORCH_COMMIT}
    for name in ("torch", "torchao", "executorch"):
        try:
            out[name] = str(importlib.import_module(name).__version__)
        except Exception:  # noqa: BLE001 - fall back to the installed distribution's metadata
            # executorch built from source exposes no __version__; the installed distribution
            # still records one, and an unpinnable runtime version is not acceptable provenance.
            try:
                out[name] = metadata.version(name)
            except Exception as exc:  # noqa: BLE001 - recorded as unavailable, never guessed
                out[name] = f"unavailable: {type(exc).__name__}"
    return out


def _ensure_checkpoint() -> str:
    """Download the promoted checkpoint and convert it to ExecuTorch's meta layout.

    The converter normalizes the `model.language_model.*` prefix this checkpoint uses and derives
    `output.weight` from the tied embedding, so all 320 tensors map without a shim.
    """
    from huggingface_hub import snapshot_download

    root = f"{VOL}/source/m1-v2"
    source = f"{root}/{CHECKPOINT}"
    if not os.path.exists(f"{source}/model.safetensors"):
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            local_dir=root,
            allow_patterns=[f"{CHECKPOINT}/*"],
        )
    observed = _sha256(f"{source}/model.safetensors")
    if observed != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(
            f"checkpoint sha256 {observed} is not the promoted {EXPECTED_WEIGHT_SHA256}"
        )

    converted = f"{VOL}/executorch/m1-v2-meta.pth"
    os.makedirs(f"{VOL}/executorch", exist_ok=True)
    if not os.path.exists(converted):
        _run(
            [
                "python", "-m", "executorch.examples.models.qwen3_5.convert_weights",
                source, converted,
            ],
            cwd=ET,
        )
        volume.commit()
    return converted


@app.function(image=base, volumes={VOL: volume}, timeout=60 * 180, cpu=16.0, memory=131072)
def export_cpu(quantize: str = "") -> dict:
    """Export the XNNPACK CPU target, optionally attempting a quantized mode."""
    checkpoint = _ensure_checkpoint()
    label = f"qwen3_5_2b_{quantize or 'fp32'}"
    output = f"{VOL}/executorch/{label}.pte"

    # `++` rather than `+`: Hydra's `+` refuses a key the config already defines, and the stock
    # qwen3_5 yaml already sets export.max_seq_length/max_context_length to 2048. `++` means
    # "add or override", so it is correct whether or not the key is present.
    command = [
        "python", "-m", "extension.llm.export.export_llm",
        "--config", "examples/models/qwen3_5/config/qwen3_5_xnnpack_fp32.yaml",
        '++base.model_class=qwen3_5_2b',
        '++base.params=examples/models/qwen3_5/config/2b_config.json',
        f"++base.checkpoint={checkpoint}",
        f"++export.max_seq_length={MAX_CONTEXT}",
        f"++export.max_context_length={MAX_CONTEXT}",
        f"++export.output_name={output}",
    ]
    if quantize:
        command += [f"++quantization.qmode={quantize}", "++quantization.group_size=32"]

    attempt = _run(command, cwd=ET, check=False)
    volume.commit()

    result = {
        "stage": "export_cpu",
        "target": "xnnpack-cpu",
        "quantization": quantize or "fp32",
        "group_size": 32 if quantize else None,
        "max_context_length": MAX_CONTEXT,
        "checkpoint_sha256": EXPECTED_WEIGHT_SHA256,
        "versions": _versions(),
        "attempt": attempt,
        "exported": os.path.exists(output),
    }
    if result["exported"]:
        result["artifact"] = output
        result["artifact_bytes"] = os.path.getsize(output)
        result["artifact_sha256"] = _sha256(output)
        result["status"] = "EXPORTED_PENDING_EVALUATION"
    else:
        result["status"] = "REJECTED_EXPORT"
    return result


SITE = "/usr/local/lib/python3.12/site-packages/executorch"
QNN_LIFT_PASS = f"{SITE}/backends/qualcomm/_passes/lift_constant_scalar_operands.py"
EXPORT_LLAMA_LIB = f"{SITE}/examples/models/llama/export_llama_lib.py"

_HYBRID_ORIGINAL = """        atten = builder_exported_to_edge.model.layers[0].attention
"""

_HYBRID_PATCHED = '''        # OpenGrad patch: this derives a KV-cache shape for TagQuantIO, but assumes layer 0
        # is a KV-cache attention module. Qwen3.5 is hybrid -- layer_types[0] is
        # "linear_attention", a Gated DeltaNet block carrying recurrent state and no KV cache --
        # so layers[0].attention has no max_context_len/n_kv_heads/head_dim and the export dies
        # with AttributeError. Take the first attention that actually has a cache instead.
        atten = next(
            (
                layer.attention
                for layer in builder_exported_to_edge.model.layers
                if hasattr(layer.attention, "max_context_len")
            ),
            builder_exported_to_edge.model.layers[0].attention,
        )
'''

_LIFT_ORIGINAL = """                n.op != "call_function"
                or isinstance(n.target, (BuiltinMethodType, BuiltinFunctionType))
                or n.target in SKIP_LIFT_OPS
"""

_LIFT_PATCHED = """                n.op != "call_function"
                or isinstance(n.target, (BuiltinMethodType, BuiltinFunctionType))
                or n.target in SKIP_LIFT_OPS
                # OpenGrad patch: higher-order ops (auto_functionalized_v2, wrapping the
                # in-place state mutation in Qwen3.5's Gated DeltaNet) carry no `_schema`,
                # and _create_tensor_args dereferences it unconditionally. A HOP has no
                # liftable constant scalar operands anyway, so skipping is correct, not a
                # workaround. Without this the QNN export dies with
                # AttributeError: 'AutoFunctionalizedV2' object has no attribute '_schema'
                or not hasattr(n.target, "_schema")
"""


def _apply_patch(path_str: str, original: str, patched: str, marker: str, bug: str) -> dict:
    """Apply one surgical upstream fix, refusing to force a match if the source has moved."""
    path = Path(path_str)
    if not path.is_file():
        return {"file": path_str, "applied": False, "reason": "file not found"}
    source = path.read_text(encoding="utf-8")
    before = hashlib.sha256(source.encode()).hexdigest()
    if marker in source:
        return {"file": path_str, "applied": False, "reason": "already patched", "sha256": before}
    if original not in source:
        return {
            "file": path_str,
            "applied": False,
            "reason": "upstream source no longer matches; re-derive the patch rather than forcing",
            "sha256": before,
        }
    path.write_text(source.replace(original, patched, 1), encoding="utf-8")
    return {
        "file": path_str,
        "applied": True,
        "sha256_before": before,
        "sha256_after": hashlib.sha256(path.read_bytes()).hexdigest(),
        "upstream_bug": bug,
    }


def _patch_executorch_for_hybrid() -> list[dict]:
    """Fix upstream assumptions that a model's attention layers are homogeneous.

    Both of these are genuine defects, not fudges to force a pass, and both are triggered by the
    same underlying fact: Qwen3.5 is a hybrid whose layers are not all KV-cache attention. They are
    recorded in the artifact provenance with before/after hashes so nobody later mistakes a patched
    toolchain for a stock one.

    That this list keeps growing is itself a finding. If the QNN path needs many more such fixes,
    the honest conclusion is that it assumes homogeneous attention throughout, and the result
    should be reported that way rather than as "Qwen3.5 works on QNN".
    """
    return [
        _apply_patch(
            QNN_LIFT_PASS,
            _LIFT_ORIGINAL,
            _LIFT_PATCHED,
            'not hasattr(n.target, "_schema")',
            "LiftConstantScalarOperands._create_tensor_args dereferences node.target._schema for "
            "every call_function node; HigherOrderOperator targets such as auto_functionalized_v2 "
            "(introduced by DeltaNet's in-place state mutation) do not have one.",
        ),
        _apply_patch(
            EXPORT_LLAMA_LIB,
            _HYBRID_ORIGINAL,
            _HYBRID_PATCHED,
            "if hasattr(layer.attention, \"max_context_len\")",
            "_to_edge_and_lower_llama derives the QNN TagQuantIO cache shape from "
            "model.layers[0].attention, assuming layer 0 has a KV cache. Qwen3.5's layer 0 is a "
            "Gated DeltaNet block with recurrent state and no cache.",
        ),
    ]


@app.function(image=qnn_image, volumes={VOL: volume}, timeout=60 * 180, cpu=16.0, memory=131072)
def export_qnn(pt2e: str = "qnn_16a4w") -> dict:
    """Attempt the Snapdragon/QNN export through the same entrypoint the CPU target uses."""
    qnn_root = next(
        (str(path) for path in Path("/opt/qnn").glob("qairt/*")),
        None,
    ) or next((str(path) for path in Path("/opt/qnn").iterdir() if path.is_dir()), "/opt/qnn")
    os.environ["QNN_SDK_ROOT"] = qnn_root
    lift_patch = _patch_executorch_for_hybrid()
    print(f"ExecuTorch hybrid patches: {json.dumps(lift_patch, indent=2)}", flush=True)

    checkpoint = _ensure_checkpoint()
    output = f"{VOL}/executorch/qwen3_5_2b_{pt2e or 'fp32'}_{QNN_SOC}.pte"
    command = [
        "python", "-m", "extension.llm.export.export_llm",
        "--config", "examples/models/qwen3_5/config/qwen3_5_xnnpack_fp32.yaml",
        '++base.model_class=qwen3_5_2b',
        '++base.params=examples/models/qwen3_5/config/2b_config.json',
        f"++base.checkpoint={checkpoint}",
        f"++export.max_seq_length={MAX_CONTEXT}",
        f"++export.max_context_length={MAX_CONTEXT}",
        f"++export.output_name={output}",
        "++backend.xnnpack.enabled=False",
        "++backend.qnn.enabled=True",
        f"++backend.qnn.soc_model={QNN_SOC}",
    ]
    # `pt2e=""` runs QNN lowering with no PT2E quantization. That is the experiment that separates
    # two very different conclusions: LiftConstantScalarOperands lives in the quantizer's
    # transform_for_annotation pipeline, so if the unquantized export succeeds the blocker is QNN
    # *quantization* of a state-mutating model, not QNN support for the architecture at all.
    if pt2e:
        command.append(f"++quantization.pt2e_quantize={pt2e}")
    # Hydra swallows the inner exception by default, leaving only "a pass failed". The precise
    # blocker is the whole point of attempting this export, so ask for the full trace.
    attempt = _run(command, cwd=ET, check=False, env={"HYDRA_FULL_ERROR": "1"})
    volume.commit()

    result = {
        "stage": "export_qnn",
        "target": "qualcomm-htp",
        "soc_model": QNN_SOC,
        "qnn_sdk_version": QNN_VERSION,
        "qnn_sdk_root": qnn_root,
        "toolchain_patch": lift_patch,
        "pt2e_quantize": pt2e,
        "max_context_length": MAX_CONTEXT,
        "versions": _versions(),
        "attempt": attempt,
        "exported": os.path.exists(output),
    }
    if result["exported"]:
        result["artifact"] = output
        result["artifact_bytes"] = os.path.getsize(output)
        result["artifact_sha256"] = _sha256(output)
        result["status"] = "EXPORTED_PENDING_EVALUATION"
    else:
        result["status"] = "REJECTED_EXPORT"
    return result


@app.function(image=base, volumes={VOL: volume}, timeout=60 * 60, cpu=8.0, memory=65536)
def audit_pte(artifact: str) -> dict:
    """Count what actually got quantized in an exported .pte.

    Phase 3B of the study forbids assuming the quantizer did what it was asked: an 8da4w export can
    complete while silently leaving most of the model in fp32, and the file size alone cannot tell
    the difference between "weights are int4" and "weights are fp32 but the buffers shrank".

    This reads the operator table out of the program and counts quantized vs unquantized linear
    ops. The expected denominator is derived from the architecture rather than guessed:

        18 linear_attention layers x (5 projections + 3 MLP) = 144
         6 full_attention   layers x (4 projections + 3 MLP) =  42
        ------------------------------------------------------------
                                       186 linear modules, + tied embedding/output
    """
    import importlib
    from collections import Counter

    path = Path(f"{VOL}/executorch/{artifact}")
    if not path.is_file():
        raise RuntimeError(f"missing artifact: {path}")

    errors: list[str] = []
    raw = None
    method = None

    # The serializer's entry point and its return type have both moved between versions: the
    # current build returns a `PTEFile` wrapper rather than the `Program` itself. Pinning either
    # the spelling or the shape produces a silent empty audit, which would read as "nothing was
    # quantized". Try the known spellings, then locate the program structurally.
    for module_name, attr in (
        ("executorch.exir._serialize._program", "deserialize_pte_binary"),
        ("executorch.exir._serialize", "deserialize_pte_binary"),
    ):
        try:
            deserialize = getattr(importlib.import_module(module_name), attr)
            raw = deserialize(path.read_bytes())
            method = f"{module_name}.{attr}"
            break
        except Exception as exc:  # noqa: BLE001 - each failed deserializer is recorded as an attempt
            errors.append(f"{module_name}.{attr}: {type(exc).__name__}: {exc}")

    if raw is None:
        return {
            "stage": "audit_pte",
            "artifact": artifact,
            "bytes": path.stat().st_size,
            "operators": {},
            "error": "could not deserialize the program",
            "attempts": errors,
        }

    # Walk one level down looking for the object that owns `execution_plan`, rather than assuming
    # the deserializer handed back the Program itself.
    program = None
    if hasattr(raw, "execution_plan"):
        program = raw
        container = "<returned directly>"
    else:
        for name in dir(raw):
            if name.startswith("__"):
                continue
            try:
                candidate = getattr(raw, name)
            except Exception:  # noqa: BLE001, S112 - a raising attribute cannot own execution_plan
                continue
            if hasattr(candidate, "execution_plan"):
                program, container = candidate, name
                break

    if program is None:
        return {
            "stage": "audit_pte",
            "artifact": artifact,
            "bytes": path.stat().st_size,
            "operators": {},
            "error": f"deserialized to {type(raw).__name__} with no reachable execution_plan",
            "returned_type": f"{type(raw).__module__}.{type(raw).__name__}",
            "returned_attributes": [n for n in dir(raw) if not n.startswith("__")],
            "attempts": errors,
        }

    operators: Counter = Counter()
    # Constant-tensor dtype histogram. This is the check that actually settles the question:
    # operator names can say "quantized" while the stored weights are still float. ExecuTorch's
    # ScalarType enum is stable at the low end (0 byte, 1 char/int8, 3 int32, 6 float32).
    scalar_types = {
        0: "uint8",
        1: "int8",
        2: "int16",
        3: "int32",
        4: "int64",
        5: "float16",
        6: "float32",
        7: "float64",
        11: "bool",
        15: "bfloat16",
    }
    const_tensors: Counter = Counter()
    const_bytes: Counter = Counter()
    const_elements: Counter = Counter()
    itemsize = {
        "uint8": 1, "int8": 1, "int16": 2, "int32": 4, "int64": 8,
        "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "bool": 1,
    }

    for plan in program.execution_plan:
        for operator in plan.operators:
            operators[f"{operator.name}.{operator.overload}".rstrip(".")] += 1
        for value in getattr(plan, "values", []) or []:
            tensor = getattr(value, "val", None)
            if tensor is None or type(tensor).__name__ != "Tensor":
                continue
            # Only constants carry weights; activations have no backing buffer.
            has_data = bool(getattr(tensor, "constant_buffer_idx", 0)) or (
                getattr(tensor, "data_buffer_idx", 0) or 0
            ) > 0
            if not has_data:
                continue
            name = scalar_types.get(getattr(tensor, "scalar_type", None), f"scalar_type_{getattr(tensor, 'scalar_type', None)}")
            elements = 1
            for dim in getattr(tensor, "sizes", []) or []:
                elements *= int(dim)
            const_tensors[name] += 1
            const_elements[name] += elements
            const_bytes[name] += elements * itemsize.get(name, 0)

    # The operator table is near-empty for a fully delegated program: XNNPACK swallows the whole
    # transformer into one opaque blob, so counting top-level ops would report "nothing quantized"
    # for a model that is entirely quantized. The byte budget is what closes the question — the
    # quantized weights are inside the blob, so its size is the evidence.
    delegates: Counter = Counter()
    delegate_bytes: Counter = Counter()
    delegate_detail = []
    blob_sizes: dict[int, int] = {}
    for index, blob in enumerate(getattr(program, "backend_delegate_data", []) or []):
        blob_sizes[index] = len(getattr(blob, "data", b"") or b"")
    segment_sizes = [int(getattr(s, "size", 0) or 0) for s in getattr(program, "segments", []) or []]

    for plan_index, plan in enumerate(program.execution_plan):
        for delegate in getattr(plan, "delegates", []) or []:
            backend = getattr(delegate, "id", "<unknown>")
            delegates[backend] += 1
            processed = getattr(delegate, "processed", None)
            location = getattr(getattr(processed, "location", None), "name", None) or str(
                getattr(processed, "location", "?")
            )
            data_index = getattr(processed, "index", None)
            size = blob_sizes.get(data_index, 0)
            if not size and data_index is not None and data_index < len(segment_sizes):
                size = segment_sizes[data_index]
            delegate_bytes[backend] += size
            delegate_detail.append(
                {
                    "plan": plan_index,
                    "method": getattr(plan, "name", None),
                    "backend": backend,
                    "location": location,
                    "index": data_index,
                    "bytes": size,
                }
            )

    # XNNPACK does not carry weights inside its delegate blobs; it references them out of the PTE's
    # named-data store. That store, not the blobs, is where a quantized transformer's bytes live, so
    # without it the byte budget is short by exactly the part under investigation.
    def _measure_store(store) -> dict:
        if store is None:
            return {"present": False}
        def _blob_len(item) -> int:
            for attr in ("buffer", "storage", "data"):
                inner = getattr(item, attr, None)
                if isinstance(inner, (bytes, bytearray, memoryview)):
                    return len(inner)
            if isinstance(item, (bytes, bytearray, memoryview)):
                return len(item)
            return 0

        buffers = list(getattr(store, "buffers", []) or [])
        buffer_sizes = [_blob_len(item) for item in buffers]

        entries = []
        errors_here = []
        # `pte_data` maps the tensor's fully-qualified name to the buffer holding its bytes. That
        # mapping is the per-layer audit: it names every weight the backend stores and how large it
        # is, so a layer left in float shows up as a 4-bytes-per-element entry instead of ~0.56.
        for source in ("pte_data", "external_data"):
            table = getattr(store, source, None)
            if not table:
                continue
            try:
                items = table.items() if hasattr(table, "items") else enumerate(table)
                for key, value in items:
                    index = getattr(value, "buffer_index", None)
                    if index is None:
                        index = getattr(value, "index", None)
                    entries.append(
                        {
                            "store": source,
                            "key": key,
                            "buffer_index": index,
                            "bytes": buffer_sizes[index]
                            if isinstance(index, int) and index < len(buffer_sizes)
                            else None,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - store shapes vary; the failure is recorded
                errors_here.append(f"{source}: {type(exc).__name__}: {exc}")

        return {
            "present": True,
            "type": f"{type(store).__module__}.{type(store).__name__}",
            "attributes": [n for n in dir(store) if not n.startswith("_")],
            "buffer_count": len(buffer_sizes),
            "total_bytes": sum(buffer_sizes),
            "entry_count": len(entries),
            "entries": sorted(entries, key=lambda e: -(e["bytes"] or 0)),
            "errors": errors_here,
        }

    named_data = _measure_store(getattr(raw, "named_data", None))
    mutable_data_buffers = getattr(raw, "mutable_data", None)
    try:
        mutable_bytes = sum(
            len(getattr(b, "storage", b"") or b"") for b in (mutable_data_buffers or [])
        )
    except Exception:  # noqa: BLE001 - an unmeasurable buffer is reported as None, not zero
        mutable_bytes = None

    quantized = {name: count for name, count in operators.items() if "quant" in name.lower()}
    linear_like = {
        name: count
        for name, count in operators.items()
        if any(token in name.lower() for token in ("linear", "addmm", "mm", "convolution"))
    }
    return {
        "stage": "audit_pte",
        "artifact": artifact,
        "bytes": path.stat().st_size,
        "deserializer": method,
        "program_container": container,
        "returned_type": f"{type(raw).__module__}.{type(raw).__name__}",
        "execution_plans": len(program.execution_plan),
        "distinct_operators": len(operators),
        "total_operator_instances": sum(operators.values()),
        "operators": dict(sorted(operators.items(), key=lambda item: -item[1])),
        "quantized_operators": dict(sorted(quantized.items(), key=lambda item: -item[1])),
        "quantized_operator_instances": sum(quantized.values()),
        "linear_like_operators": dict(sorted(linear_like.items(), key=lambda item: -item[1])),
        "constant_tensors_by_dtype": dict(const_tensors),
        "constant_elements_by_dtype": dict(const_elements),
        "constant_bytes_by_dtype": dict(const_bytes),
        "delegates_by_backend": dict(delegates),
        "delegate_bytes_by_backend": dict(delegate_bytes),
        "delegate_detail": delegate_detail,
        "backend_delegate_blobs": len(blob_sizes),
        "backend_delegate_blob_bytes": sum(blob_sizes.values()),
        "segments": len(segment_sizes),
        "segment_bytes": sum(segment_sizes),
        "named_data_store": named_data,
        "mutable_data_bytes": mutable_bytes,
        "returned_attributes": [n for n in dir(raw) if not n.startswith("__")],
        # Derived from config.json and then *checked* against the fp32 store, which is why the
        # per-layer counts below are not the ones this function originally assumed (8 and 7). The
        # store holds 205 weights and the earlier guess implied 186; the architecture, not the
        # guess, is authoritative.
        "expected_weights": {
            "full_attention_layers": 6,
            "per_full_attention_layer": {
                "q_proj (gated, 2x)": 8 * 256 * 2 * 2048,
                "k_proj": 2 * 256 * 2048,
                "v_proj": 2 * 256 * 2048,
                "o_proj": 8 * 256 * 2048,
            },
            "linear_attention_layers": 18,
            "per_linear_attention_layer": {
                "in_proj_qkv (fused 3x16x128)": 6144 * 2048,
                "in_proj_z (output gate)": 2048 * 2048,
                "out_proj": 2048 * 2048,
                "a_proj": 16 * 2048,
                "b_proj": 16 * 2048,
                "conv1d (depthwise, kernel 4)": 6144 * 1 * 4,
            },
            "mlp_per_layer": {"gate": 6144 * 2048, "up": 6144 * 2048, "down": 2048 * 6144},
            "layers": 24,
            "lm_head_tied": 248320 * 2048,
            "total_named_weights": 6 * 4 + 18 * 6 + 24 * 3 + 1,
        },
    }


@app.function(image=probe_image, timeout=60 * 20, cpu=2.0)
def probe_mediatek_sdk() -> dict:
    """Establish, with evidence, whether the NeuroPilot SDK is obtainable without a login.

    The MediaTek branch cannot produce a binary without `mtk_converter`, so this decides between
    a real export and a documented BLOCKED_SDK_ACCESS. It is recorded either way; a gated SDK is a
    finding about the deployment target, not a reason to go quiet.
    """
    probe = _run(
        f"curl -sS -L -o /dev/null -w '%{{http_code}} %{{url_effective}}' '{MTK_SDK_URL}'",
        check=False,
    )
    pip_probe = _run("pip download mtk-converter --no-deps -d /tmp/mtk", check=False)
    return {
        "stage": "probe_mediatek_sdk",
        "target": "mediatek-neuropilot",
        "sdk_url": MTK_SDK_URL,
        "http_probe": probe["stdout_tail"].strip(),
        "pip_available": pip_probe["returncode"] == 0,
        "pip_probe_tail": pip_probe["stderr_tail"][-1500:],
        "required_components": [
            "mtk_converter (cp310 wheel)",
            "mtk_neuron (linux x86_64 wheel)",
            "libneuronusdk_adapter.mtk.so",
            "libneuron_buffer_allocator.so",
            "NeuronAdapter.h",
        ],
        "supported_chips": ["Dimensity 9300", "Dimensity 9400"],
        "notes": (
            "ExecuTorch has no qwen3_5 model definition for MediaTek; its llm_models cover "
            "Qwen2/2.5/3 dense only. There is no host emulator for this backend, so even a "
            "successful build could not be behaviourally evaluated without a D9300/D9400 device."
        ),
    }


@app.local_entrypoint()
def main(target: str = "cpu", quantize: str = "", pt2e: str = "qnn_16a4w", artifact: str = ""):
    out = ROOT / "results/quantization/executorch"
    out.mkdir(parents=True, exist_ok=True)

    def save(name: str, payload: dict) -> None:
        path = out / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}  status={payload.get('status', 'n/a')}")

    if target == "cpu":
        save(f"export_cpu_{quantize or 'fp32'}", export_cpu.remote(quantize))
    elif target == "qnn":
        save(f"export_qnn_{pt2e}", export_qnn.remote(pt2e))
    elif target == "mediatek-probe":
        save("probe_mediatek_sdk", probe_mediatek_sdk.remote())
    elif target == "audit":
        result = audit_pte.remote(artifact)
        save(f"audit_{artifact.replace('.pte', '')}", result)
        print(json.dumps({k: v for k, v in result.items() if k != "operators"}, indent=2)[:3000])
    else:
        raise SystemExit(f"unknown target: {target}")
