#!/usr/bin/env python3
"""Upload a large exported artifact from the Modal volume straight to the Hugging Face Hub.

The `.pte` files live on the Modal volume and are gigabytes each. Routing them through the
developer's machine would mean downloading and re-uploading the same bytes, so this pushes them
from where they already are.

The token is passed as an **ephemeral** `Secret.from_dict`, which exists for the duration of the run
and is not persisted as a named Modal secret. Small handoff files (prompts, scorer, model card) are
uploaded from the host instead, so the only thing this function needs the credential for is the
single large blob.

Usage:
    modal run scripts/modal/upload_artifact.py \\
        --repo experimentalmachines/QwenGrad-... --volume-path /vol/executorch/qwen3_5_2b_8da4w.pte
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import modal

VOL = "/vol"

app = modal.App("opengrad-artifact-upload")
volume = modal.Volume.from_name("opengrad-quant", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub[hf_transfer]", "hf_transfer"
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2] if len(here.parents) >= 3 else here.parent


ROOT = _repo_root()


@app.function(
    image=image,
    volumes={VOL: volume},
    timeout=60 * 180,
    cpu=8.0,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("OPENGRAD_HF_TOKEN", "")})],
)
def upload(repo_id: str, volume_path: str, path_in_repo: str | None = None) -> dict:
    """Push one file, verifying its hash on the way out so provenance survives the transfer."""
    from huggingface_hub import HfApi

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("no HF_TOKEN reached the container; set OPENGRAD_HF_TOKEN locally")
    source = Path(volume_path)
    if not source.is_file():
        raise RuntimeError(f"missing artifact on the volume: {volume_path}")

    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    # hf_transfer gives multi-threaded uploads; without it a multi-GiB push is needlessly serial.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(source),
        path_in_repo=path_in_repo or source.name,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add {source.name} ({source.stat().st_size:,} bytes)",
    )
    return {
        "repo_id": repo_id,
        "file": source.name,
        "path_in_repo": path_in_repo or source.name,
        "bytes": source.stat().st_size,
        "sha256": digest.hexdigest(),
        "url": f"https://huggingface.co/{repo_id}",
    }


@app.local_entrypoint()
def main(repo: str, volume_path: str, path_in_repo: str = ""):
    result = upload.remote(repo, volume_path, path_in_repo or None)
    print(f"uploaded {result['file']}  {result['bytes']:,} bytes")
    print(f"  sha256 {result['sha256']}")
    print(f"  {result['url']}")
