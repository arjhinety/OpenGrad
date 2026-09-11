# OpenGrad DPO and On-Policy Distillation (OPD) Implementation Report

**Building in Public.** Every gradient is a hypothesis. Every checkpoint is evidence.

---

## 1. Executive Summary

This report documents the implementation of the complete Direct Preference Optimization (DPO) and On-Policy Distillation (OPD) foundations within OpenGrad. The platform now supports SFT, DPO, and On-Policy Distillation underneath a unified post-training experiment operating system.

All components have been designed and verified on a single **NVIDIA A100-SXM4-80GB**, with complete offline CPU test suites and deterministic mock pathways enabling safe execution in GPU-free CI environments.

---

## 2. What Existed Before vs. What Was Reused vs. What Was Added

### What Existed Before:
- Initial SFT training scripts and canonical data materialization pipelines.
- Multi-tier benchmark evaluation system (Tiers A–E, speculative decoding, OpenWeights mobile testing).
- Baseline experiment lifecycle contracts (`runs/<experiment-id>/`).

### What Was Reused:
- `Qwen35_2BRenderer` and `parse_qwen_native_output` for chat template rendering and tool call parsing.
- `ExperimentStore` and `ExperimentLedger` for immutable run directory contracts and append-only event logging.
- `CheckpointRegistry` for registering and tracking model checkpoints.
- Benchmark harnesses and `RegressionEngine` for non-regression verification.

### What Was Added:
1. **Hardware Probe (`src/opengrad/hardware/probe.py`)**:
   - Detects GPU name, compute capability, VRAM (79.2 GB), CUDA runtime (12.4), PyTorch, Transformers, and TRL.
   - Accurately identifies `A100_80GB` and validates BF16 support.
2. **Tokenizer Compatibility Gate (`src/opengrad/distillation/tokenizer_gate.py`)**:
   - Compares Qwen3.5-2B vs. Qwen3.8-27B across vocab size (248,064), special tokens, tool markup tokens, and representative text/call tokenizations.
   - Command: `opengrad distill validate-teacher`.
3. **Synthetic DPO Pipeline (`src/opengrad/preferences/`)**:
   - Candidate generation ($N=4$ candidates per training prompt).
   - Stage 1: `DeterministicJudge` scoring decisions, tool selection, argument grounding, and clarification behavior.
   - Stage 2: `OpenAIJudge` for ambiguous semantic boundaries, featuring hard budget stops (`max_requests`, `max_cost_usd`), structured JSON output, and persistent disk caching.
   - Quality gate (`chosen != rejected`, tokenization check, no held-out leakage).
   - Commands: `opengrad preference inspect`, `generate`, `validate`, `build`.
4. **DPO Trainer Backend (`src/opengrad/training/dpo.py`)**:
   - Enforces frozen SFT reference policy semantics (prevents accidental fallback to base model).
   - Tracks reward margin, chosen/rejected rewards, and preference accuracy.
5. **On-Policy Distillation Engine (`src/opengrad/distillation/`, `src/opengrad/training/distillation.py`)**:
   - Prompt-state extraction from canonical trajectories (`src/opengrad/distillation/prompts.py`).
   - On-policy rollout generation (`src/opengrad/distillation/rollouts.py`) tracking `policy_staleness_steps`.
   - `TeacherAdvantageEvaluator` verifying teacher capability delta before training (`TEACHER_GAP_INSUFFICIENT` fail-closed gate).
   - Single-A100 execution modes (Mode A: Co-resident in 80GB VRAM, Mode B: Teacher server, Mode C: Bounded-staleness alternating).
   - Tool-call loss masking (loss applied only to assistant tokens; prompt, system, tools, and observations masked).
   - Commands: `opengrad distill build-prompts`, `smoke`, `train`, `opengrad rollout inspect`, `stats`.
6. **Secret Safety & Hygiene**:
   - Added `.env.example`.
   - Enhanced `scripts/repo/check_publication_hygiene.py` with secret key scanning (`sk-`, `OPENAI_API_KEY=`).

---

## 3. Dataset Boundaries & Held-Out Firewall

- **Canonical SFT Dataset**: `arrochi112/OpenGrad-ToolPolicy-Canonical-v1` (213,951 records) is used **only** as a candidate prompt/context pool. Its assistant targets are never simply renamed to "chosen".
- **Real Preference Dataset**: When2Call preference data is maintained separately.
- **Held-Out Evaluation Firewall**: Training prompts, synthetic DPO candidate prompts, and OPD rollout prompt states explicitly exclude all examples in `reports/evaluation/behavioral-heldout-v2.manifest.json` and external benchmark suites (BFCL, tau3, ACEBench).

---

## 4. Hardware Probe & Single-A100 Execution Mode

Live hardware probe results:
- **Device:** `NVIDIA A100-SXM4-80GB`
- **Total VRAM:** 79.2 GB
- **Compute Capability:** 8.0 (Ampere)
- **BF16 Support:** YES (Native)
- **FP8 Support:** NO (Ampere architecture; Hopper+ required for native FP8)
- **Selected OPD Mode:** `MODE_A_CO_RESIDENT`
  - Student Qwen3.5-2B in BF16: ~12 GB (weights + optimizer + KV cache)
  - Teacher Qwen3.8-27B in BF16: ~58 GB (frozen weights + KV cache)
  - Combined peak VRAM: ~70–74 GB, safely fitting within 79.2 GB A100 VRAM without quantization.

---

## 5. Tokenizer Compatibility Results

Running `opengrad distill validate-teacher`:
- Student: `Qwen/Qwen3.5-2B` (Vocab: 248,064)
- Teacher: `Qwen/Qwen3.8-27B` (Vocab: 248,064)
- Special tokens (`<|im_start|>`, `<|im_end|>`): Identical
- Tool markup tokens (`<tool_call>`, `</tool_call>`): Identical
- Representative text and function call tokenizations: Identical
- **Verdict:** **`TOKENIZER_COMPATIBLE`**

---

## 6. Verification & Test Results

- **PyTest:** All tests passing.
- **Ruff & Mypy:** Clean, zero warnings, strict static type checking enforced.
- **Registry Validation:** `registry validation: OK`.
- **System Doctor (`opengrad doctor`):**
  - Python: PASS (3.12.3)
  - Disk: PASS (61 GB free)
  - Android Studio: INSTALLED (`/opt/android-studio`)
  - Android SDK: INSTALLED (`/opt/android-sdk`)
  - Pixel 7 AVD: PROVISIONED (`/root/.android/avd/pixel_phone.avd`)
  - Benchmark Registry: OK

---

## 7. Known Limitations

1. **A100 Single-GPU Concurrency**: While Mode A (`MODE_A_CO_RESIDENT`) fits both student and teacher on the A100 80GB, training with batch size $> 2$ or sequence lengths $> 4096$ requires enabling gradient checkpointing and activation offloading.
2. **OpenAI API Key**: Semantic adjudication falls back to deterministic mock when `OPENAI_API_KEY` is not present in the environment.
3. **Android Emulator**: The provisioned Pixel 7 phone AVD runs under software emulation due to the host QEMU container lacking nested KVM (`/dev/kvm`).

---

## 8. Post-Training Commands for Experiments

### Execute DPO Experiment:
```bash
# 1. Preflight check
opengrad preflight configs/training/dpo/qwen35_2b_dpo_a100.yaml

# 2. Launch DPO training against promoted SFT checkpoint
opengrad train configs/training/dpo/qwen35_2b_dpo_a100.yaml

# 3. Evaluate candidate DPO model on full benchmark suite
opengrad evaluate checkpoints/qwen35_2b_dpo_a100 --suite full_post_training

# 4. Compare candidate against SFT baseline
opengrad compare runs/qwen35_2b_m0_sft/eval runs/qwen35_2b_dpo_a100/eval
```

### Execute On-Policy Distillation Experiment:
```bash
# 1. Validate tokenizer compatibility
opengrad distill validate-teacher

# 2. Extract eligible prompt states
opengrad distill build-prompts --profile residual --count 500

# 3. Run distillation smoke check (VRAM & teacher advantage)
opengrad distill smoke

# 4. Launch On-Policy Distillation training
opengrad distill train configs/training/distillation/qwen35_2b_qwen38_27b_a100.yaml

# 5. Inspect rollout history and staleness metrics
opengrad rollout stats --file runs/qwen35_2b_qwen38_27b_a100/rollouts/rollout_history.jsonl

# 6. Evaluate and check for regression vs. DPO / SFT
opengrad evaluate checkpoints/qwen35_2b_qwen38_27b_a100 --suite agent_transfer
```
