from opengrad.hardware.probe import HardwareProbeResult, probe_hardware


def test_hardware_probe_runs_and_returns_result() -> None:
    result = probe_hardware()
    assert isinstance(result, HardwareProbeResult)
    d = result.to_dict()
    assert "gpu_available" in d
    assert "total_vram_gb" in d
    assert "bf16_supported" in d

    summary = result.render_summary()
    assert "Hardware Probe Summary" in summary
    assert "BF16 Support" in summary


def test_hardware_probe_detects_live_a100() -> None:
    result = probe_hardware()
    if result.gpu_available and result.gpu_name and "A100" in result.gpu_name:
        assert result.is_a100 is True
        assert result.a100_variant in {"A100_80GB", "A100_40GB"}
        assert result.bf16_supported is True
        assert result.fp8_native_supported is False  # Ampere does not have native FP8
