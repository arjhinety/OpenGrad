import json
from pathlib import Path

import yaml
from jsonschema import validate as validate_json

ROOT = Path(__file__).parents[2]


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_frozen_heldout_manifest_is_contract_valid_and_pinned():
    manifest = _json("reports/evaluation/behavioral-heldout-v1.manifest.json")
    validate_json(manifest, _json("registry/evaluation_manifest.schema.json"))
    assert manifest["status"] == "FROZEN_PRE_GPU"
    assert manifest["frozen"] is True
    contract = manifest["model_renderer_contract"]
    assert contract["model_revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc"
    assert (
        contract["template_hash"]
        == "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
    )


def test_baseline_and_m2_are_explicitly_pre_gpu_and_unresolved():
    baseline = yaml.safe_load(
        (ROOT / "configs/experiments/tool_calling/qwen35_2b_baseline.yaml").read_text()
    )
    evaluation = yaml.safe_load(
        (ROOT / "configs/evaluation/tool_calling/qwen35_2b_baseline.yaml").read_text()
    )
    residual = yaml.safe_load(
        (ROOT / "configs/data/tool_calling/residual_policy_v1.yaml").read_text()
    )
    assert baseline["status"] == "PLANNED" and baseline["seed"] == 0
    assert evaluation["status"] == "FROZEN_PRE_GPU"
    assert residual["resolution"] == "UNRESOLVED_UNTIL_BASELINE"
    assert residual["weights_are_not_final"] is True


def test_runtime_registry_does_not_claim_vendor_support():
    registry = yaml.safe_load((ROOT / "registry/runtime_components.yaml").read_text())
    validate_json(registry, _json("registry/runtime_components.schema.json"))
    assert {item["id"] for item in registry["components"]} == {
        "nvidia-modelopt",
        "vllm-speculators",
        "dspark",
    }
    # The ModelOpt entry now has an interface, but "INTERFACE_ONLY" is still not a
    # vendor-support claim: nothing is SUPPORTED_AFTER_VERIFICATION.
    statuses = {item["id"]: item["status"] for item in registry["components"]}
    assert statuses["nvidia-modelopt"] == "INTERFACE_ONLY"
    assert statuses["vllm-speculators"] == "REFERENCE_ONLY"
    assert statuses["dspark"] == "REFERENCE_ONLY"
    assert "SUPPORTED_AFTER_VERIFICATION" not in statuses.values()
    assert all(item["platforms"]["nvidia"] == "NOT_TESTED" for item in registry["components"])


def test_gpu_preflight_record_reflects_the_executed_boundary_smoke():
    """The preflight is no longer a `NOT_RUN` placeholder: the boundary smoke has run, so the
    record must name the observed accelerator and a compatibility basis (grounded in the receipt)
    rather than assert vendor support."""
    config = yaml.safe_load((ROOT / "configs/hardware/gpu_preflight_v1.yaml").read_text())
    validate_json(config, _json("registry/gpu_preflight.schema.json"))
    assert config["status"] == "READY"
    assert config["compatibility"]["result"] == "COMPATIBLE"
    assert config["observed"]["device_count"] == 1
    assert config["observed"]["devices"][0]["name"] == "NVIDIA A100-SXM4-80GB"
    assert "qwen_gpu_smoke.json" in config["compatibility"]["basis"]
    smoke = _json("reports/hardware/qwen_gpu_smoke.json")
    assert smoke["status"] == "PASS"
    assert smoke["hardware"]["gpu_name"] == "NVIDIA A100-SXM4-80GB"
