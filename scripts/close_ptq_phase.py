#!/usr/bin/env python3
"""Freeze the PTQ phase into one immutable, self-verifying provenance record.

This does not recompute anything. It collects the evidence the phase already produced, verifies
that every artifact it references carries a hash, assigns release roles, and writes a closure
manifest plus `reports/PTQ_PHASE_CLOSURE.md`.

Two rules this script exists to enforce:

* **No artifact may be described as satisfying the secondary release bar when it does not.** Both
  accepted rungs failed that bar. Their roles below are release *roles*, not a claim that the bar
  was met, and the original per-rung failure list travels with each one.
* **Frozen verdicts stay visible.** If a later QAD artifact outperforms these, that is recorded as
  a new experiment family; it does not edit or supersede these rows.

Release roles are deliberately distinct from gate verdicts:

| role | meaning |
|---|---|
| `REFERENCE` | the measurement everything else is compared against |
| `RECOMMENDED_RELEASE` | conservative default ship candidate |
| `MEMORY_OPTIMIZED_RELEASE` | highest-compression artifact that passed the primary gate |
| `REJECTED_ACCURACY` | scored and failed the primary gate |
| `EXPERIMENTAL` | reserved for QAD; no PTQ artifact carries it |

Usage:
    python scripts/close_ptq_phase.py
    python scripts/close_ptq_phase.py --verify   # re-check an existing closure, write nothing
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "results/quantization/gguf"
OUT_MANIFEST = ROOT / "manifests/quantization/ptq_phase_closure_v1.json"
OUT_REPORT = ROOT / "reports/PTQ_PHASE_CLOSURE.md"

LADDER = ("Q2_K", "Q3_K_M", "IQ3_M", "IQ4_XS", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
GIB = 1024**3

# Assigned after the ladder was scored, from the frozen verdicts — not a re-ranking of them.
# Q8_0 and Q6_K both passed the primary gate and both FAILED the secondary release bar; the roles
# distinguish what each is for, and the failures are carried alongside so the distinction cannot
# be read as a pass.
ROLES = {
    "bf16": "REFERENCE",
    "Q8_0": "RECOMMENDED_RELEASE",
    "Q6_K": "MEMORY_OPTIMIZED_RELEASE",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance degrades to None outside a git checkout
        return None


def collect() -> dict:
    prepare = load(G / "prepare.json")
    imatrix = load(G / "imatrix.json")
    tokenizer = load(G / "inspect_tokenizer.json")
    parity = load(G / "engine_parity_verdict.json")
    criteria = load(G / "release_selection_criteria_v1.json")
    verdict = load(G / "ladder_verdict.json")
    reference = load(ROOT / "results/quantization/m1_v2_reference.json")
    gate = load(ROOT / "results/quantization/quantization_preservation_v1.json")
    baseline = load(G / "score_m1-v2-bf16_confirmatory.json")

    missing = [
        name for name, doc in {
            "prepare.json": prepare, "imatrix.json": imatrix,
            "inspect_tokenizer.json": tokenizer, "engine_parity_verdict.json": parity,
            "release_selection_criteria_v1.json": criteria, "ladder_verdict.json": verdict,
            "m1_v2_reference.json": reference, "quantization_preservation_v1.json": gate,
            "score_m1-v2-bf16_confirmatory.json": baseline,
        }.items() if doc is None
    ]
    if missing:
        raise SystemExit(f"cannot close the phase, missing evidence: {missing}")

    artifacts = []
    unhashed = []
    artifacts.append({
        "artifact": "m1-v2-bf16.gguf",
        "quantization": "BF16",
        "role": ROLES["bf16"],
        "gate_decision": baseline["gate_vs_frozen_vllm_reference"]["decision"],
        "bytes": prepare["gguf_bytes"],
        "gib": round(prepare["gguf_bytes"] / GIB, 3),
        "compression_vs_bf16": 1.0,
        "sha256": prepare["gguf_sha256"],
        "metrics": baseline["metrics"],
        "secondary_release_bar": "not applicable (reference, not a release candidate)",
    })

    by_rung = {r["rung"]: r for r in verdict["rungs"]}
    for name in LADDER:
        rung = by_rung.get(name)
        quant = load(G / f"quantize_{name}.json")
        score = load(G / f"score_m1-v2-{name}_confirmatory.json")
        if rung is None or not rung.get("present") or score is None:
            artifacts.append({
                "artifact": f"m1-v2-{name}.gguf", "quantization": name,
                "role": "NOT_SCORED", "gate_decision": "NOT_SCORED",
                "bytes": (quant or {}).get("artifact_bytes"),
                "sha256": (quant or {}).get("artifact_sha256"),
            })
            continue
        decision = score["gate_vs_frozen_vllm_reference"]["decision"]
        passed = str(decision).upper() in {"PTQ_ACCEPTED", "PASS", "PASSED", "ACCEPTED"}
        role = ROLES.get(name, "REJECTED_ACCURACY" if not passed else "ACCEPTED_NO_ROLE")
        sha = score.get("artifact_sha256") or (quant or {}).get("artifact_sha256")
        if not sha:
            unhashed.append(name)
        loss = score.get("quantization_loss_vs_llamacpp_bf16", {})
        agreement = loss.get("output_agreement", {})
        artifacts.append({
            "artifact": f"m1-v2-{name}.gguf",
            "quantization": name,
            "role": role,
            "gate_decision": decision,
            "failed_gate_dimensions": [
                c["dimension"] for c in score["gate_vs_frozen_vllm_reference"]["checks"]
                if not c["passed"]
            ],
            "bytes": score.get("artifact_bytes") or (quant or {}).get("artifact_bytes"),
            "gib": round((score.get("artifact_bytes") or 0) / GIB, 3),
            "compression_vs_bf16": round(
                prepare["gguf_bytes"] / (score.get("artifact_bytes") or 1), 3
            ),
            "sha256": sha,
            "quantizer_command": (quant or {}).get("command"),
            "metrics": score["metrics"],
            "retention_vs_llamacpp_bf16": loss.get("relative_retention"),
            "decision_agreement_vs_llamacpp_bf16": agreement.get("decision_agreement"),
            "exact_output_agreement": agreement.get("byte_identical_rate"),
            "decision_flips": agreement.get("decision_flips"),
            "flip_effects": agreement.get("flip_effects"),
            # Carried on every accepted artifact so a release role can never be misread as a pass.
            "secondary_release_bar": (
                "FAILED: " + "; ".join(rung.get("release_bar_failures") or [])
                if rung.get("release_bar_failures")
                else ("PASSED" if rung.get("meets_release_bar") else "not evaluated (gate failed)")
            ),
        })

    if unhashed:
        raise SystemExit(f"refusing to close: artifacts without a sha256: {unhashed}")

    return {
        "phase": "ptq",
        "phase_status": "CLOSED",
        "policy_version": gate.get("policy_version", "quantization_preservation_v1"),
        "closure_manifest_version": "ptq_phase_closure_v1",
        "immutability": (
            "Every value here is copied from evidence the phase already wrote. Nothing was "
            "regenerated to produce cleaner hashes. Later QAD results form a separate experiment "
            "family and must not edit these rows."
        ),
        "source_model": {
            "repo": prepare["model_repo"],
            "revision": prepare["model_revision"],
            "checkpoint": prepare["checkpoint"],
            "weights_sha256": prepare["source_weight_sha256"],
            "weights_bytes": prepare["source_weight_bytes"],
            "config_sha256": prepare["source_config_sha256"],
        },
        "tokenizer": {
            "gguf_pre_type": (tokenizer.get("keys_matched") or {}).get("tokenizer.ggml.pre"),
            "gguf_model": (tokenizer.get("keys_matched") or {}).get("tokenizer.ggml.model"),
            "revision": reference.get("tokenizer_revision"),
            "template_hash": reference["evaluation"]["template_hash"],
            "source_file_sha256": (parity.get("provenance") or {}).get("source_file_sha256"),
            "add_bos_token": (parity.get("provenance") or {}).get("special_tokens", {}).get(
                "add_bos_token"
            ),
        },
        "runtime": {
            "engine": "llama.cpp",
            "tag": prepare["llama_cpp_tag"],
            "commit": prepare["llama_cpp_commit"],
            "converter_command": prepare["converter_command"],
            "gguf_metadata": prepare.get("gguf_metadata", {}).get("keys_matched"),
            "gpu": "A100-80GB (Modal)",
            "note": "throughput figures are datacentre reference numbers, never device numbers",
        },
        "calibration": {
            "corpus_sha256": imatrix["calibration_sha256"],
            "imatrix_sha256": imatrix["imatrix_sha256"],
            "imatrix_bytes": imatrix["imatrix_bytes"],
            "parse_special": imatrix["parse_special"],
            "corpus": "manifests/quantization/m1_v2_imatrix_calibration_v2.txt",
            "contamination": "training-side only; zero overlap with DEV or confirmatory ids",
        },
        "evaluation": {
            "partition": "confirmatory",
            "examples": reference["evaluation"]["confirmatory_examples"],
            "fingerprint": reference["evaluation"]["confirmatory_fingerprint"],
            "manifest_sha256": reference["evaluation"]["manifest_sha256"],
            "renderer": reference["evaluation"]["renderer"],
            "frozen_reference_runtime": reference["runtime"]["recorded_engine"],
            "frozen_reference_caveat": (
                "aggregate metrics only; per-example vLLM predictions were never preserved, so no "
                "per-example vLLM-to-llama.cpp agreement is claimed anywhere in this phase"
            ),
        },
        "generation_config": {
            "temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 0,
            "max_new_tokens": 512, "cache_prompt": False, "slot_context": 5760,
            "n_parallel": 8,
        },
        "gates": {
            "primary": {
                "policy": "quantization_preservation_v1",
                "thresholds": gate.get("thresholds"),
                "status": "APPLIED UNCHANGED against the frozen vLLM BF16 aggregate reference",
            },
            "secondary_release_bar": {
                "policy": criteria["policy_version"],
                "frozen_before": criteria["frozen_before"],
                "result": "NO RUNG MET IT",
                "note": (
                    "Q6_K and Q8_0 passed the primary gate and both failed this bar. No artifact "
                    "in this phase satisfies it and none may be described as if it does."
                ),
            },
            "engine_tokenizer_parity": {
                "status": parity["engine_tokenizer_parity"],
                "exact_matches": parity["exact_matches"],
                "checked": parity["checked"],
                "mismatches": parity["mismatches"],
                "materiality": parity["tokenizer_materiality"],
                "note": "FAILED and deliberately not redefined; llama.cpp left unpatched",
            },
            "quantization_baseline": parity["quantization_baseline"],
        },
        "artifacts": artifacts,
        "selection": verdict["selection"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "repo_commit": git("rev-parse", "HEAD"),
            "repo_dirty": bool(git("status", "--porcelain")),
            "modal_volume": "opengrad-quant",
            "container_images": {
                "gguf_study": "nvidia/cuda 12.8.1 devel + llama.cpp b10919 built with "
                              "CMAKE_CUDA_ARCHITECTURES=80",
                "note": "image digests are not exposed by the Modal CLI; the llama.cpp commit and "
                        "CUDA version are the reproducible pins",
            },
        },
    }


def render(manifest: dict) -> str:
    rows = []
    for a in manifest["artifacts"]:
        if a["role"] == "NOT_SCORED":
            rows.append(f"| `{a['quantization']}` | — | `NOT_SCORED` | `NOT_SCORED` | — |")
            continue
        m = a["metrics"]
        rows.append(
            f"| `{a['quantization']}` | {a['gib']:.2f} | `{a['role']}` | "
            f"`{a['gate_decision']}` | f1 {m['call_f1']:.4f} · prec {m['call_precision']:.4f} · "
            f"over-call {m['over_call_rate']:.4f} · clarify {m['clarification_accuracy']:.4f} |"
        )
    accepted = [a for a in manifest["artifacts"] if a["role"] in ROLES.values()]
    bar_rows = "\n".join(
        f"| `{a['quantization']}` | `{a['role']}` | {a['secondary_release_bar']} |"
        for a in accepted if a["quantization"] != "BF16"
    )
    hash_rows = "\n".join(
        f"| `{a['artifact']}` | {a.get('bytes') or 0:,} | `{a.get('sha256') or 'n/a'}` |"
        for a in manifest["artifacts"]
    )
    src = manifest["source_model"]
    cal = manifest["calibration"]
    rt = manifest["runtime"]
    par = manifest["gates"]["engine_tokenizer_parity"]

    return f"""# PTQ phase closure — M1-v2 GGUF

**Phase status: `CLOSED`.** Every number below is copied from evidence this phase already wrote.
Nothing was regenerated to produce cleaner hashes, and no threshold was moved.

## Release roles

Roles describe what each artifact is *for*. They are not gate verdicts and they are not a claim
that the secondary release bar was met — **no artifact in this phase met that bar.**

| rung | GiB | role | primary gate | key behaviour |
|---|---:|---|---|---|
{chr(10).join(rows)}

### Secondary release bar — failed by both accepted rungs

| rung | role | secondary bar |
|---|---|---|
{bar_rows}

`Q5_K_M` and every rung below it were **scored and failed the primary gate**. They must not be
described as behaviourally equivalent to BF16.

One result that makes the reason for a multi-metric gate concrete: **`Q4_K_M` posts `call_f1`
0.7594, above BF16's 0.7572**, while `over_call_rate` degrades 0.1553 → 0.2512 and
`clarification_accuracy` collapses 0.7682 → 0.6388. **`call_f1` alone must never be used as the
quantization acceptance criterion.**

## Engine tokenizer parity — `{par['status']}`

{par['exact_matches']}/{par['checked']} exact, {par['mismatches']} mismatches, materiality
**`{par['materiality']}`**. Not redefined, not marked as passing, and stock llama.cpp was not
patched. Detail: [`QUANTIZATION_ENGINE_PARITY.md`](QUANTIZATION_ENGINE_PARITY.md).
The six divergent examples are now permanent regression fixtures — see
[`TOKENIZER_DIVERGENCE_REGRESSION.md`](TOKENIZER_DIVERGENCE_REGRESSION.md).

## Provenance

| | |
|---|---|
| model repo | `{src['repo']}` |
| model revision | `{src['revision']}` |
| checkpoint | `{src['checkpoint']}` |
| weights sha256 | `{src['weights_sha256']}` |
| config.json sha256 | `{src['config_sha256']}` |
| tokenizer pre-type | `{manifest['tokenizer']['gguf_pre_type']}` |
| chat template hash | `{manifest['tokenizer']['template_hash']}` |
| llama.cpp | `{rt['tag']}` / `{rt['commit']}` |
| converter command | `{rt['converter_command']}` |
| imatrix sha256 | `{cal['imatrix_sha256']}` |
| calibration corpus sha256 | `{cal['corpus_sha256']}` |
| imatrix `--parse-special` | `{cal['parse_special']}` |
| confirmatory fingerprint | `{manifest['evaluation']['fingerprint']}` |
| decoding | greedy, `temperature 0.0`, `top_k 1`, `top_p 1.0`, seed 0, `cache_prompt false` |
| GPU | {rt['gpu']} |

### Artifact hashes

| artifact | bytes | sha256 |
|---|---:|---|
{hash_rows}

## Standing caveats

- The frozen vLLM BF16 reference is **aggregate only**. Its per-example predictions were never
  preserved, so no per-example vLLM ↔ llama.cpp agreement is claimed anywhere in this phase.
- Throughput figures are A100-80GB reference numbers. They are **not** device numbers and no
  mobile behavioural verdict is claimed from them.
- The confirmatory partition is internal, not an untouched external benchmark.
- CPU/mobile behavioural validation belongs to OpenWeights and has **not** been claimed.

Machine-readable closure: `manifests/quantization/ptq_phase_closure_v1.json`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    manifest = collect()

    if args.verify:
        existing = load(OUT_MANIFEST)
        if existing is None:
            raise SystemExit("no closure manifest to verify")
        drift = [
            key for key in ("source_model", "calibration", "runtime", "evaluation", "gates")
            if existing.get(key) != manifest.get(key)
        ]
        if drift:
            raise SystemExit(f"closure manifest drifted from live evidence in: {drift}")
        print("closure manifest matches live evidence")
        return 0

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_REPORT.write_text(render(manifest), encoding="utf-8")
    print(f"wrote {OUT_MANIFEST.relative_to(ROOT)}")
    print(f"wrote {OUT_REPORT.relative_to(ROOT)}")
    for a in manifest["artifacts"]:
        print(f"  {a['quantization']:<8} {a['role']:<26} {a['gate_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
