"""Hardware probe capturing accelerator specifications, VRAM limits, and precision capabilities."""

from __future__ import annotations

import importlib
import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HardwareProbeResult:
    gpu_available: bool
    gpu_name: str | None
    is_a100: bool
    a100_variant: str | None  # "A100_80GB", "A100_40GB", or None
    total_vram_gb: float
    compute_capability: tuple[int, int] | None
    cuda_driver_version: str | None
    cuda_runtime_version: str | None
    bf16_supported: bool
    fp8_native_supported: bool  # False on Ampere/A100; True on Hopper (H100/H200)
    pytorch_version: str | None
    transformers_version: str | None
    trl_version: str | None
    vllm_available: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "is_a100": self.is_a100,
            "a100_variant": self.a100_variant,
            "total_vram_gb": round(self.total_vram_gb, 2),
            "compute_capability": list(self.compute_capability)
            if self.compute_capability
            else None,
            "cuda_driver_version": self.cuda_driver_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "bf16_supported": self.bf16_supported,
            "fp8_native_supported": self.fp8_native_supported,
            "pytorch_version": self.pytorch_version,
            "transformers_version": self.transformers_version,
            "trl_version": self.trl_version,
            "vllm_available": self.vllm_available,
            "details": self.details,
        }

    def render_summary(self) -> str:
        lines = [
            "# Accelerator Hardware Probe Summary",
            "",
            f"- **GPU Available:** {'YES' if self.gpu_available else 'NO'}",
            f"- **Device Name:** {self.gpu_name or 'N/A'}",
            f"- **A100 Variant:** {self.a100_variant or ('Not an A100' if self.gpu_available else 'N/A')}",
            f"- **Total VRAM:** {self.total_vram_gb:.1f} GB",
            f"- **Compute Capability:** {self.compute_capability[0]}.{self.compute_capability[1]}"
            if self.compute_capability
            else "- **Compute Capability:** N/A",
            f"- **CUDA Driver:** {self.cuda_driver_version or 'N/A'}",
            f"- **BF16 Support:** {'YES (Native hardware)' if self.bf16_supported else 'NO'}",
            f"- **Native FP8 Support:** {'YES (Hopper+)' if self.fp8_native_supported else 'NO (Ampere architecture)'}",
            f"- **PyTorch:** {self.pytorch_version or 'Not installed'}",
            f"- **Transformers:** {self.transformers_version or 'Not installed'}",
            f"- **TRL:** {self.trl_version or 'Not installed'}",
            "",
        ]
        return "\n".join(lines)


def probe_hardware() -> HardwareProbeResult:
    """Probe hardware using PyTorch / NVML / nvidia-smi with explicit A100 40GB vs 80GB detection."""
    gpu_available = False
    gpu_name = None
    is_a100 = False
    a100_variant = None
    vram_gb = 0.0
    compute_cap = None
    cuda_driver = None
    cuda_runtime = None
    bf16_support = False
    fp8_support = False

    # Check via torch if available
    torch_mod = None
    torch_version = None
    try:
        torch_mod = importlib.import_module("torch")
        torch_version = getattr(torch_mod, "__version__", None)
        if torch_mod.cuda.is_available():
            gpu_available = True
            gpu_name = torch_mod.cuda.get_device_name(0)
            compute_cap = torch_mod.cuda.get_device_capability(0)
            vram_bytes = torch_mod.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024**3)
            cuda_runtime = getattr(torch_mod.version, "cuda", None)
            bf16_support = bool(compute_cap[0] >= 8)  # Ampere (sm_80+) supports BF16
            fp8_support = bool(compute_cap[0] >= 9)  # Hopper (sm_90+) supports FP8
    except (ImportError, AttributeError, RuntimeError):
        pass

    # If torch was unable to get device info, probe via nvidia-smi
    if not gpu_available:
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
            res = subprocess.check_output(cmd, text=True).strip()
            if res:
                parts = [p.strip() for p in res.splitlines()[0].split(",")]
                if len(parts) >= 3:
                    gpu_available = True
                    gpu_name = parts[0]
                    vram_mb = float(parts[1])
                    vram_gb = vram_mb / 1024.0
                    cuda_driver = parts[2]
                    if "A100" in gpu_name:
                        compute_cap = (8, 0)
                        bf16_support = True
                        fp8_support = False
        except (subprocess.SubprocessError, FileNotFoundError, ValueError, IndexError):
            pass

    # Analyze A100 variant
    if gpu_name and "A100" in gpu_name:
        is_a100 = True
        if vram_gb >= 70.0:
            a100_variant = "A100_80GB"
        else:
            a100_variant = "A100_40GB"

    # Check transformers and trl
    transformers_version = None
    try:
        t_mod = importlib.import_module("transformers")
        transformers_version = getattr(t_mod, "__version__", None)
    except ImportError:
        pass

    trl_version = None
    try:
        trl_mod = importlib.import_module("trl")
        trl_version = getattr(trl_mod, "__version__", None)
    except ImportError:
        pass

    vllm_available = False
    try:
        importlib.import_module("vllm")
        vllm_available = True
    except ImportError:
        pass

    return HardwareProbeResult(
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        is_a100=is_a100,
        a100_variant=a100_variant,
        total_vram_gb=vram_gb,
        compute_capability=compute_cap,
        cuda_driver_version=cuda_driver,
        cuda_runtime_version=cuda_runtime,
        bf16_supported=bf16_support,
        fp8_native_supported=fp8_support,
        pytorch_version=torch_version,
        transformers_version=transformers_version,
        trl_version=trl_version,
        vllm_available=vllm_available,
    )
