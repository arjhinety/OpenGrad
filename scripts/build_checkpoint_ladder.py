#!/usr/bin/env python3
"""Resolve the real checkpoint ladder for the general-capability diagnosis.

Stage identities are taken from artifacts that already exist -- the frozen quantization reference
names its own parent (`m0_sft_canonical_v2_final::checkpoint-1800`, weight sha `7144579a...`), and
the run registry names the stage experiments. Nothing here invents a stage: a repo that cannot be
resolved on the Hub is recorded MISSING_CHECKPOINT rather than substituted.

Only small metadata files are fetched (config, tokenizer config, chat template, the safetensors
index). Weights are never downloaded locally; artifact identity comes from the Hub-reported LFS
sha256 of each weight shard, which is the same digest the H200 job recomputes after download.

Usage:
    python scripts/build_checkpoint_ladder.py
    python scripts/build_checkpoint_ladder.py --verify   # fail if the ladder file drifted
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/benchmarks/checkpoint_ladder.json"
REFERENCE = ROOT / "results/quantization/m1_v2_reference.json"

# Stage identity comes from the frozen reference and the run registry, not from naming guesses.
# `subfolder` is the real on-repo layout: the DPO repos publish one directory per saved step.
# `pinned_revision` overrides the branch head where an earlier artifact already froze a commit --
# the current checkpoint MUST be read at the revision the frozen reference names, because that
# repo's main has since moved for README-only commits.
LADDER = [
    {
        "stage": "BASE",
        "order": 0,
        "repo": "Qwen/Qwen3.5-2B",
        "subfolder": "",
        "pinned_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "experiment_id": None,
        "checkpoint_id": None,
        "role": "upstream pretrained/instruct base; no OpenGrad post-training",
    },
    {
        "stage": "M0_SFT",
        "order": 1,
        "repo": "arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CanonicalV2-Final",
        "subfolder": "",
        "pinned_revision": None,
        "experiment_id": "m0_sft_canonical_v2_final",
        "checkpoint_id": "checkpoint-1800",
        "role": "supervised fine-tuning on ToolPolicy-Canonical-v2; direct parent of M1-v2",
    },
    {
        "stage": "M1_DPO_CURRENT",
        "order": 2,
        "repo": "arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2",
        "subfolder": "dpo-checkpoint-30",
        "pinned_revision": "f33d20308982f37deb459076f489e794d5521ee3",
        "experiment_id": "m1_dpo_canonical_v2_final_v2",
        "checkpoint_id": "dpo-checkpoint-30",
        "role": "promoted preference-tuned checkpoint; the model under diagnosis",
    },
    {
        "stage": "M1_DPO_HISTORICAL",
        "order": 3,
        "repo": "arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO",
        "subfolder": "checkpoint-300",
        "pinned_revision": None,
        "experiment_id": "qwen35_2b_m1_dpo",
        "checkpoint_id": "checkpoint-300",
        "role": (
            "earlier DPO lineage on a different SFT parent (CorpusV2, not CanonicalV2-Final). "
            "SUPPLEMENTARY only -- it is not on the base->M0->M1-v2 path, so a delta against it "
            "mixes two changes and cannot localise a stage."
        ),
        "supplementary": True,
    },
]

META_FILES = ["config.json", "tokenizer_config.json", "generation_config.json"]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_meta(api: HfApi, repo: str, subfolder: str, pinned: str | None) -> dict:
    """Resolve one repo/subfolder's identity. Returns availability plus every hash we can pin."""
    try:
        info = api.model_info(repo, revision=pinned, files_metadata=True)
    except RepositoryNotFoundError:
        return {"available": False, "unavailable_reason": "MISSING_CHECKPOINT: repo not found on the Hub"}
    except HfHubHTTPError as exc:
        return {"available": False, "unavailable_reason": f"HUB_ERROR: {exc.__class__.__name__}: {exc}"}

    revision = info.sha
    prefix = f"{subfolder}/" if subfolder else ""
    all_files = sorted(s.rfilename for s in info.siblings)
    files = sorted(f[len(prefix):] for f in all_files if f.startswith(prefix) and "/" not in f[len(prefix):])
    if not files:
        return {
            "available": False,
            "revision": revision,
            "unavailable_reason": (
                f"MISSING_CHECKPOINT: subfolder {subfolder!r} holds no files at revision {revision}"
            ),
        }

    def dl(name: str) -> Path:
        return Path(hf_hub_download(repo, f"{prefix}{name}", revision=revision))

    # Weight identity: the Hub's own LFS digests, one per shard, folded into a stable ladder hash.
    shards = {}
    for s in info.siblings:
        if not s.rfilename.startswith(prefix) or ".safetensors" not in s.rfilename:
            continue
        if s.rfilename.endswith(".json") or s.lfs is None:
            continue
        digest = getattr(s.lfs, "sha256", None) or (
            s.lfs.get("sha256") if isinstance(s.lfs, dict) else None
        )
        if digest:
            shards[s.rfilename[len(prefix):]] = digest
    weight_identity = (
        sha256_text("\n".join(f"{k}:{v}" for k, v in sorted(shards.items()))) if shards else None
    )

    out = {
        "available": True,
        "revision": revision,
        "revision_is_pinned": pinned is not None,
        "branch_head": api.model_info(repo).sha,
        "file_count": len(files),
        "files": files,
        "weight_shards": shards,
        "weight_shard_count": len(shards),
        "weight_identity_sha256": weight_identity,
        "weight_identity_note": (
            "sha256 over 'filename:lfs_sha256' lines for every safetensors shard, sorted. "
            "Not a hash of the tensor bytes concatenated; it is a stable identity for the shard set."
        ),
        "single_shard_sha256": next(iter(shards.values())) if len(shards) == 1 else None,
        "has_chat_template_file": "chat_template.jinja" in files,
    }

    for name in META_FILES:
        if name not in files:
            out[f"{name}_sha256"] = None
            continue
        raw = dl(name).read_bytes()
        out[f"{name}_sha256"] = hashlib.sha256(raw).hexdigest()
        if name == "config.json":
            cfg = json.loads(raw)
            text_cfg = cfg.get("text_config") or cfg
            out["architectures"] = cfg.get("architectures")
            out["num_hidden_layers"] = text_cfg.get("num_hidden_layers")
            out["hidden_size"] = text_cfg.get("hidden_size")
            out["vocab_size"] = text_cfg.get("vocab_size")
            out["torch_dtype"] = cfg.get("torch_dtype") or cfg.get("dtype")
            out["is_multimodal_config"] = "text_config" in cfg or "vision_config" in cfg
        if name == "tokenizer_config.json":
            tcfg = json.loads(raw)
            out["add_bos_token"] = tcfg.get("add_bos_token")
            out["bos_token"] = tcfg.get("bos_token")
            out["eos_token"] = tcfg.get("eos_token")
            inline = tcfg.get("chat_template")
            if isinstance(inline, str):
                out["chat_template_sha256"] = sha256_text(inline)
                out["chat_template_source"] = "tokenizer_config.json:chat_template"

    if out.get("chat_template_sha256") is None and out["has_chat_template_file"]:
        out["chat_template_sha256"] = sha256_text(dl("chat_template.jinja").read_text(encoding="utf-8"))
        out["chat_template_source"] = "chat_template.jinja"

    # Tokenizer identity. `tokenizer.json` alone is the comparable quantity across stages: the base
    # repo additionally ships vocab.json/merges.txt for slow-tokenizer consumers, and including
    # those would report a difference that does not exist in the tokenizer actually used.
    tok_files = [f for f in files if f.startswith("tokenizer") or f in ("vocab.json", "merges.txt")]
    tok_digests = {f: hashlib.sha256(dl(f).read_bytes()).hexdigest() for f in tok_files}
    out["tokenizer_files"] = tok_digests
    out["tokenizer_json_sha256"] = tok_digests.get("tokenizer.json")
    out["tokenizer_identity_sha256"] = sha256_text(
        "\n".join(f"{k}:{v}" for k, v in sorted(tok_digests.items()))
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="compare against the committed ladder")
    args = ap.parse_args()

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    api = HfApi()

    entries = []
    for spec in LADDER:
        print(f"resolving {spec['stage']:<20} {spec['repo']}", flush=True)
        meta = fetch_meta(api, spec["repo"], spec["subfolder"], spec["pinned_revision"])
        entry = {**spec, **meta}
        entries.append(entry)
        state = "available" if meta["available"] else meta["unavailable_reason"]
        print(f"  -> {state}", flush=True)

    by_stage = {e["stage"]: e for e in entries}

    # Cross-checks against evidence that already exists, so the ladder cannot silently disagree
    # with the frozen reference this whole study is anchored to.
    checks = {}
    cur = by_stage["M1_DPO_CURRENT"]
    m0 = by_stage["M0_SFT"]
    base = by_stage["BASE"]
    checks["current_revision_matches_frozen_reference"] = (
        cur.get("revision") == reference["hf_publication_revision"]
    )
    checks["current_chat_template_matches_frozen_reference"] = (
        cur.get("chat_template_sha256") == reference["evaluation"]["template_hash"]
    )
    checks["current_weights_match_h200_run"] = (
        cur.get("single_shard_sha256")
        == "903f9b115d418d1988c0d5e1dd4bda7e04809a54a6bc7e34431982fbee2262f6"
    )
    checks["current_config_matches_h200_run"] = (
        cur.get("config.json_sha256")
        == "88bf86c270d616198909ed1eefef8d8c21ac1fa13f62e947f20f8e1ebd02c211"
    )
    checks["m0_chat_template_matches_current"] = (
        m0.get("chat_template_sha256") == cur.get("chat_template_sha256")
    )
    checks["m0_tokenizer_json_matches_current"] = (
        m0.get("tokenizer_json_sha256") == cur.get("tokenizer_json_sha256")
    )
    checks["base_tokenizer_json_matches_current"] = (
        base.get("tokenizer_json_sha256") == cur.get("tokenizer_json_sha256")
    )
    checks["base_chat_template_matches_current"] = (
        base.get("chat_template_sha256") == cur.get("chat_template_sha256")
    )
    checks["m0_architecture_matches_current"] = (
        m0.get("architectures") == cur.get("architectures")
    )
    # Measured, not assumed: the Hub's LFS digest for the published M0 shard equals the
    # parent_weight_sha256 the frozen M1-v2 reference recorded for its training-time parent. The
    # published M0 repo therefore IS the checkpoint M1-v2 was trained from, byte for byte, which
    # is what makes the M0 -> M1 edge a clean isolation of the preference stage.
    checks["m0_weights_match_frozen_reference_parent"] = (
        m0.get("single_shard_sha256") == reference["parent_weight_sha256"]
    )
    # The base repo is the multimodal-capable release; the post-trained stages are text-only
    # causal LMs. This is recorded rather than asserted, because it is a real property of the
    # upstream artifact and it constrains how the BASE rung may be loaded.
    checks["base_architecture_matches_current"] = (
        base.get("architectures") == cur.get("architectures")
    )

    payload = {
        "schema_version": 1,
        "run": "h200-capability-diagnosis-v1",
        "purpose": (
            "Localise the stage at which the OpenWeights sentinel refusal behaviour first becomes "
            "observable. The decisive path is BASE -> M0_SFT -> M1_DPO_CURRENT; every stage on it "
            "shares one tokenizer and one chat template, which is what makes the deltas readable."
        ),
        "primary_path": ["BASE", "M0_SFT", "M1_DPO_CURRENT"],
        "anchored_to": {
            "frozen_reference": "results/quantization/m1_v2_reference.json",
            "declared_parent": reference["parent_checkpoint"],
            "declared_parent_weight_sha256": reference["parent_weight_sha256"],
            "declared_current_revision": reference["hf_publication_revision"],
        },
        "consistency_checks": checks,
        "known_cross_stage_differences": [
            {
                "id": "BASE_IS_MULTIMODAL_ARCHITECTURE",
                "stages": ["BASE"],
                "observed": (
                    "Qwen/Qwen3.5-2B publishes architectures=['Qwen3_5ForConditionalGeneration'] "
                    "with a vision_config (depth 24, hidden 1024) and image/video token ids. "
                    "M0_SFT and M1_DPO_CURRENT publish ['Qwen3_5ForCausalLM'] with a flat "
                    "text-only config. The text_config of the base is shape-identical to the "
                    "post-trained config: 24 layers, hidden 2048, vocab 248320, "
                    "full_attention_interval 4, linear_conv_kernel_dim 4."
                ),
                "handling": (
                    "The BASE rung is evaluated as published, text-only, with no images supplied. "
                    "The vision tower is not exercised. The language stack is NOT extracted and "
                    "re-saved as a causal LM, because a reconstructed artifact is not the original."
                ),
                "residual_risk": (
                    "A multimodal wrapper can differ from a text-only wrapper in prompt handling "
                    "even with no image present. This is a caveat on BASE deltas, not on the "
                    "M0->M1 delta, where both sides are Qwen3_5ForCausalLM."
                ),
            },
            {
                "id": "BASE_TOKENIZER_PRETOKENIZER_REGEX_DIFFERS",
                "stages": ["BASE", "M0_SFT", "M1_DPO_CURRENT"],
                "observed": (
                    "tokenizer.json byte digests differ between the base and the post-trained "
                    "stages. The vocab (248044 entries) and the merge list are IDENTICAL once "
                    "serialization format is normalised (base stores merges as space-joined "
                    "strings, the post-trained stages as 2-element arrays). Two real differences "
                    "remain: (1) the Split pre-tokenizer regex -- base uses the Qwen3.5 form "
                    "'[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+ ... ?[^\\s\\p{L}\\p{M}\\p{N}]+' which "
                    "is aware of Unicode combining marks, while the post-trained stages carry the "
                    "older Qwen2 form without \\p{M}; (2) ByteLevel trim_offsets is false on base "
                    "and true on the post-trained stages. The post-trained stages also register 7 "
                    "extra added_tokens (ids 248070-248076, audio/tts placeholders) absent from "
                    "the base's added_tokens list."
                ),
                "likely_cause": (
                    "The post-trained tokenizers were re-serialised by the training stack, which "
                    "rebuilt the fast tokenizer from an older Qwen2-family template. NOT an "
                    "intentional change, and NOT verified here -- stated as the probable "
                    "mechanism, not as a finding."
                ),
                "handling": (
                    "Each checkpoint is evaluated with its own tokenizer, because the tokenizer is "
                    "part of the artifact. The number of benchmark prompts whose token ids differ "
                    "between the base and post-trained tokenizers is measured separately on CPU "
                    "and reported; if it is zero the difference is behaviourally inert for this "
                    "study and the BASE delta is readable."
                ),
                "relates_to": "reports/TOKENIZER_DIVERGENCE_REGRESSION.md",
            },
        ],
        "checkpoints": entries,
        "missing": [e["stage"] for e in entries if not e["available"]],
        "notes": [
            "No intermediate DPO step exists between M0_SFT and M1_DPO_CURRENT that is published; "
            "dpo-checkpoint-30 is the promoted one and the only DPO artifact on the Hub for this "
            "lineage, so the M0->M1 delta is the whole preference stage, not one step of it.",
            "parent_weight_sha256 in the frozen reference is a digest of the training-time "
            "checkpoint. It was initially assumed this need not equal the published M0 repo's "
            "shard digest. It does: both are "
            "7144579aeecec8b4de25f193ab63085efdf8d9d76b85ed915352291b0152277a, verified twice -- "
            "from the Hub's LFS metadata here, and by recomputing sha256 over the downloaded file "
            "inside the H200 container. The published M0 is the exact parent of M1-v2.",
            "Two consistency checks are EXPECTED to fail: base_architecture_matches_current and "
            "base_tokenizer_json_matches_current. Both are documented in "
            "known_cross_stage_differences and are properties of the upstream artifact, not "
            "errors in this ladder. Every check on the M0->M1 edge passes, which is the edge that "
            "isolates the preference stage.",
        ],
        "expected_failing_checks": [
            "base_architecture_matches_current",
            "base_tokenizer_json_matches_current",
        ],
    }

    if args.verify:
        if not OUT.exists():
            print("FAIL: ladder file does not exist", file=sys.stderr)
            return 1
        prior = json.loads(OUT.read_text(encoding="utf-8"))
        drift = [
            e["stage"]
            for e, p in zip(entries, prior["checkpoints"])
            if e.get("revision") != p.get("revision")
            or e.get("weight_identity_sha256") != p.get("weight_identity_sha256")
        ]
        if drift:
            print(f"FAIL: ladder drifted for {drift}", file=sys.stderr)
            return 1
        print("ladder verified: no drift")
        return 0

    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("\nconsistency checks:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("\nladder:")
    for e in entries:
        tag = " (supplementary)" if e.get("supplementary") else ""
        if e["available"]:
            print(f"  {e['stage']:<20} {e['revision'][:12]}  {e['repo']}{tag}")
        else:
            print(f"  {e['stage']:<20} MISSING  {e['unavailable_reason']}")
    failed = [k for k, v in checks.items() if not v]
    unexpected = [k for k in failed if k not in payload["expected_failing_checks"]]
    if unexpected:
        print(f"\nUNEXPECTED consistency failure(s): {unexpected} -- inspect before evaluating.")
        return 1
    if failed:
        print(f"\n{len(failed)} expected failure(s), documented in known_cross_stage_differences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
