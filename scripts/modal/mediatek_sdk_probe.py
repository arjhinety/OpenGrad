#!/usr/bin/env python3
"""Establish, with evidence, whether the MediaTek NeuroPilot SDK is obtainable without a login.

Kept as its own Modal app deliberately. Folded into the ExecuTorch export app it would sit behind a
40-minute source build for what is two minutes of curl, and the answer gates whether the MediaTek
branch can produce a binary at all: `mtk_converter` is what turns a model into a MediaTek
representation, and nothing substitutes for it.

A gated SDK is a finding about the deployment target, not a reason to go quiet. Either outcome is
recorded.

Usage:
    modal run scripts/modal/mediatek_sdk_probe.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

NEUROPILOT_DOCS = "https://neuropilot.mediatek.com/resources/public/npexpress/en/docs/npexpress"
NEUROPILOT_PORTAL = "https://neuropilot.mediatek.com/"

# The components backends/mediatek/README.md requires before anything can be built or run.
REQUIRED_COMPONENTS = (
    "mtk_converter (cp310 manylinux wheel) — preprocesses the model into a MediaTek representation",
    "mtk_neuron (linux x86_64 wheel) — converts the model to device binaries",
    "libneuronusdk_adapter.mtk.so — target-dependent execution",
    "libneuron_buffer_allocator.so — DMA buffer allocation",
    "NeuronAdapter.h — copied into backends/mediatek/runtime/include/api/",
)

SUPPORTED_CHIPS = ("MediaTek Dimensity 9300", "MediaTek Dimensity 9400")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2] if len(here.parents) >= 3 else here.parent


ROOT = _repo_root()

app = modal.App("opengrad-mediatek-probe")
image = modal.Image.debian_slim(python_version="3.12").apt_install("curl")


def _probe(url: str) -> dict:
    result = subprocess.run(
        [
            "curl", "-sS", "-L", "-o", "/dev/null",
            "-w", "%{http_code} %{num_redirects} %{url_effective}",
            "--max-time", "60", url,
        ],
        text=True,
        capture_output=True,
    )
    parts = result.stdout.strip().split(" ", 2)
    return {
        "url": url,
        "http_code": parts[0] if parts else None,
        "redirects": parts[1] if len(parts) > 1 else None,
        "final_url": parts[2] if len(parts) > 2 else None,
        "curl_returncode": result.returncode,
        "stderr": result.stderr.strip()[-400:],
    }


@app.function(image=image, timeout=60 * 10, cpu=2.0)
def probe() -> dict:
    http = [_probe(NEUROPILOT_DOCS), _probe(NEUROPILOT_PORTAL)]
    pip = subprocess.run(
        ["pip", "download", "mtk-converter", "--no-deps", "-d", "/tmp/mtk"],
        text=True,
        capture_output=True,
    )
    pip_neuron = subprocess.run(
        ["pip", "download", "mtk-neuron", "--no-deps", "-d", "/tmp/mtk"],
        text=True,
        capture_output=True,
    )

    reachable = any(item["http_code"] == "200" for item in http)
    on_pypi = pip.returncode == 0 or pip_neuron.returncode == 0
    obtainable = on_pypi  # only a real wheel unblocks the binary step

    return {
        "stage": "probe_mediatek_sdk",
        "target": "mediatek-neuropilot",
        "http_probes": http,
        "docs_reachable": reachable,
        "pypi": {
            "mtk_converter_returncode": pip.returncode,
            "mtk_converter_stderr": pip.stderr.strip()[-600:],
            "mtk_neuron_returncode": pip_neuron.returncode,
            "mtk_neuron_stderr": pip_neuron.stderr.strip()[-600:],
            "available": on_pypi,
        },
        "sdk_obtainable_without_login": obtainable,
        "status": "SDK_AVAILABLE" if obtainable else "BLOCKED_SDK_ACCESS",
        "required_components": list(REQUIRED_COMPONENTS),
        "supported_chips": list(SUPPORTED_CHIPS),
        "model_support": (
            "ExecuTorch has no qwen3_5 model definition for MediaTek. examples/mediatek/models/"
            "llm_models covers Qwen2/2.5/3 dense only (modeling_qwen.py), and there is no "
            "export_llm integration for this backend."
        ),
        "evaluability": (
            "No host emulator exists for the MediaTek backend, unlike Qualcomm's x86 HTP "
            "emulator. Even a successful build could not be behaviourally evaluated without a "
            "Dimensity 9300/9400 device."
        ),
    }


@app.local_entrypoint()
def main():
    result = probe.remote()
    out = ROOT / "results/quantization/executorch"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "probe_mediatek_sdk.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)[:2500])
    print(f"\nwrote {path.relative_to(ROOT)}  status={result['status']}")
