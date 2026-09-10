# Agent & Harness Integration Guide

**Building in Public.** OpenGrad is designed to be fully operable by human researchers and autonomous AI agents (such as DeepSeek Harness plugins, coding agents, or terminal copilots) through stable commands, deterministic schemas, and machine-readable output.

---

## 1. Design Philosophy

OpenGrad does not depend on any specific agent harness internally. Instead, it exposes a stable, predictable CLI and API boundary. Agents interact through commands that support the `--json` flag, emitting clean JSON without requiring agents to parse pretty terminal tables or ANSI colors.

```text
DeepSeek Harness / AI Agent
         │
         ▼
  OpenGrad CLI / API
  ├── opengrad preflight <config> --json
  ├── opengrad train <config> --json
  ├── opengrad evaluate <target> --suite <suite> --json
  ├── opengrad compare <base> <cand> --json
  ├── opengrad failures <run> --json
  ├── opengrad checkpoint list --json
  ├── opengrad experiment show <id> --json
  ├── opengrad experiment diff <a_id> <b_id> --json
  ├── opengrad promote <ckpt_id> --json
  └── opengrad doctor --json
```

---

## 2. Agent Permissions Model

Actions are partitioned into three impact tiers:

### Tier 1: Read-Only (Unrestricted)
Safe for any inspection turn:
- `opengrad doctor --json`
- `opengrad validate-data <path> --json`
- `opengrad inspect-template --json`
- `opengrad benchmark list`
- `opengrad checkpoint list --json`
- `opengrad experiment list --json`
- `opengrad experiment diff <a_id> <b_id> --json`
- `opengrad failures <run_dir> --json`

### Tier 2: Safe Write (Sandboxed Execution)
Requires workspace write permissions:
- `opengrad benchmark dry-run`
- `opengrad preflight <config> --json`
- `opengrad evaluate <target> --suite <suite> --dry-run --json`
- `opengrad train <config> --dry-run --json`
- `opengrad contamination-scan --benchmark <bm>`

### Tier 3: High Impact (Resource & Publication Bounds)
Operations that commit compute time or promote models:
- `opengrad train <config>` (GPU training execution)
- `opengrad evaluate <target> --suite full_post_training` (Full GPU evaluation matrix)
- `opengrad promote <checkpoint_id> --reason "<justification>"`
- `opengrad reject <checkpoint_id> --reason "<justification>"`

---

## 3. Stable Machine-Readable Error Codes

When an operation fails, OpenGrad emits a documented error code:

| Error Code | Meaning |
| :--- | :--- |
| `CONFIG_INVALID` | Experiment configuration failed schema validation. |
| `DATASET_SCHEMA_INVALID` | Training/evaluation dataset records violate canonical schema. |
| `CHECKSUM_MISMATCH` | Dataset manifest checksum does not match data on disk. |
| `CONTAMINATION_FAILURE` | Critical benchmark overlap detected in training manifest. |
| `TOKENIZER_MISMATCH` | Model and tokenizer revisions or configurations conflict. |
| `BASELINE_NOT_FOUND` | Designated baseline checkpoint or run directory does not exist. |
| `BENCHMARK_VERSION_MISMATCH`| Attempted comparison of disparate benchmark revisions. |
| `PATH_NOT_WRITABLE` | Insufficient disk space or invalid run directory path. |
| `ALGORITHM_UNSUPPORTED` | Requested training algorithm is unrecognized. |
| `CHECKPOINT_NOT_FOUND` | Requested checkpoint ID is not registered in the registry. |
