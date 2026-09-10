# Dataset Manifests & Provenance Specification

**Building in Public.** Datasets in OpenGrad are immutable experimental artifacts, not untracked folders of JSON files.

---

## 1. The Dataset Manifest Contract

Defined in `src/opengrad/data/manifest.py`, every dataset split carries an immutable JSON manifest:

```json
{
  "dataset_name": "OpenGrad-ToolPolicy-Canonical-v1",
  "source": "arrochi112/OpenGrad-ToolPolicy-Canonical-v1",
  "upstream_revision": "bb295d8a4ad64f7e8161044ad2fa34f873ede418",
  "adapter": "canonical",
  "schema_version": "1.0",
  "split": "train",
  "raw_record_count": 213951,
  "valid_record_count": 213951,
  "rejected_record_count": 0,
  "deduplicated_count": 0,
  "final_count": 213951,
  "checksum": "181b3fba1d722cddfab9026ffeb0ab47783398e96dac06be6b727183b017a911",
  "deterministic_fingerprint": "d7a4...",
  "tokenizer": "Qwen/Qwen3.5-2B",
  "contamination_status": "CLEAN_LEVEL_1_TO_5",
  "license_constraints": "CC-BY-4.0 / Apache-2.0"
}
```

---

## 2. Order-Independent Deterministic Fingerprinting

Fingerprints are calculated over sorted canonical conversation hashes:
$$\text{Fingerprint} = \text{SHA256}\left(\bigoplus_{i=1}^N \text{hash}(x_i)\right)$$
If a single token, role, or tool argument changes, the fingerprint changes, preventing accidental mutation of frozen training corpora.

---

## 3. Preflight Manifest Verification

During `opengrad preflight <config>`, the manifest validator checks:
1. Every path referenced in `datasets.manifest_paths` exists.
2. The local dataset SHA-256 matches the manifest checksum.
3. No training manifest contains records overlapping with frozen evaluation held-outs.
