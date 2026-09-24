#!/usr/bin/env bash
# One-command MediaTek export for Qwen3.5, ready for the moment the NeuroPilot wheels exist.
#
# It checks every prerequisite before doing anything, because each missing piece fails later with a
# less obvious error: a missing mtk_converter surfaces deep inside the conversion, and a missing
# NeuronAdapter.h surfaces as a compile error in the runtime build.
#
# Usage:
#   ./export_qwen3_5.sh /path/to/dpo-checkpoint-30 [output_dir]

set -euo pipefail

CHECKPOINT="${1:?usage: $0 <checkpoint-dir> [output-dir]}"
OUTPUT_DIR="${2:-./qwen3_5_2b_mtk}"
: "${EXECUTORCH_ROOT:?set EXECUTORCH_ROOT to your ExecuTorch checkout}"

fail() { echo "BLOCKED: $*" >&2; exit 1; }

echo "== prerequisites =="

python - <<'PY' || fail "mtk_converter is not installed. It is portal-gated: register at
       https://neuropilot.mediatek.com/ and install the cp310 wheel. Without it there is no path
       from a PyTorch module to a MediaTek binary."
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("mtk_converter") else 1)
PY
echo "  mtk_converter  OK"

python - <<'PY' || fail "mtk_neuron is not installed (NeuroPilot Express)."
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("mtk_neuron") else 1)
PY
echo "  mtk_neuron     OK"

PYVER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[ "$PYVER" = "3.10" ] || echo "  WARNING: python $PYVER; the mtk_converter wheel is cp310"

[ -f "$EXECUTORCH_ROOT/backends/mediatek/runtime/include/api/NeuronAdapter.h" ] \
  || fail "NeuronAdapter.h is missing. Copy it to
       \$EXECUTORCH_ROOT/backends/mediatek/runtime/include/api/"
echo "  NeuronAdapter  OK"

[ -n "${ANDROID_NDK:-}" ] || fail "ANDROID_NDK is unset (needs r26.3.11579264)."
echo "  ANDROID_NDK    OK"

MODELS="$EXECUTORCH_ROOT/examples/mediatek/models/llm_models"
for f in configuration_qwen3_5.py modeling_qwen3_5.py; do
  [ -f "$MODELS/$f" ] || fail "$f is not installed. Copy it into $MODELS/"
done
echo "  model defs     OK"

[ -f "$CHECKPOINT/model.safetensors" ] || fail "no model.safetensors in $CHECKPOINT"
echo "  checkpoint     OK"

echo
echo "== export =="
mkdir -p "$OUTPUT_DIR"

# The DeltaNet layer is unimplemented and modeling_qwen3_5.py raises rather than emitting a model
# that silently produces wrong numbers. This runner is deliberately not tolerant of that: a
# "successful" export of a half-implemented model is the outcome to avoid, not to work around.
cd "$EXECUTORCH_ROOT"
python -m examples.mediatek.model_export_scripts.qwen \
  --model_name qwen3_5_2b \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_len 5760 \
  "${@:3}"

echo
echo "== next =="
echo "An export is not a result. Before trusting it:"
echo "  1. per-layer numerics vs HF Qwen3_5ForCausalLM"
echo "  2. full-model logits vs the BF16 reference"
echo "  3. state threading: one-pass vs split-prompt must agree"
echo "  4. frozen confirmatory set -> score_generations.py -> quantization_preservation_v1"
echo "See IMPLEMENTATION_SPEC.md."
