#!/usr/bin/env bash
# Ascend A2/A3 (910B/910C) NPU cross-framework run (scaffold). Same harness and
# report shape as the NVIDIA rounds, but on Huawei Ascend NPUs. One script serves
# both NPUs — pick the chip with DBF_HARDWARE_PROFILE.
#
# The runner auto-detects Ascend via `npu-smi info`, records device=npu, and
# appends `--device npu` to the SGLang-Diffusion and vLLM-Omni serve commands
# (and sets ASCEND_RT_VISIBLE_DEVICES instead of CUDA_VISIBLE_DEVICES). No code
# change is needed for a first run; set the values below to the box.
#
# Prerequisites (mirror scripts/run_b200_cross_framework_20260912.sh):
#   - sglang importable from its own checkout, vLLM-Omni in its isolated venv,
#     both built with Ascend NPU support
#   - an HF token in $DBF_STATE_DIR/.hftoken for the gated model licences
#
#   DBF_REPO_DIR=/path/to/diffusion-bench-framework \
#   DBF_HARDWARE_PROFILE=a2 bash scripts/run_ascend.sh   # Ascend 910B
#   DBF_REPO_DIR=/path/to/diffusion-bench-framework \
#   DBF_HARDWARE_PROFILE=a3 bash scripts/run_ascend.sh   # Ascend 910C
#
# Results land in $DBF_LOG_DIR; merge an NPU run with the matching GPU run and
# publish exactly as the NVIDIA rounds do (the report splits each framework into
# GPU/NPU columns when a merge carries both devices).
set -u

export DBF_STATE_DIR="${DBF_STATE_DIR:-/workspace/npu}"
export DBF_REPO_DIR="${DBF_REPO_DIR:?set DBF_REPO_DIR to this repo's checkout on the NPU box}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-${DBF_STATE_DIR}/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${DBF_HF_HOME}}"
export DBF_FRAMEWORKS="${DBF_FRAMEWORKS:-sglang vllm-omni}"
export DBF_HARDWARE_PROFILE="${DBF_HARDWARE_PROFILE:-a2}"
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-5400}"

# Ascend device ids, like CUDA_VISIBLE_DEVICES ids; the runner maps this to
# ASCEND_RT_VISIBLE_DEVICES on NPU.
GPUS="${GPUS:-0}"
# Tag defaults to the chip (ascenda2 / ascenda3) so one script serves both.
TAG="${TAG:-ascend${DBF_HARDWARE_PROFILE}}"

# A small image-only starter set; extend to the cases your NPU can serve.
CASES="${CASES:-
zimage_turbo_t2i_1024
qwen_image_2512_t2i_1024
}"

echo "=== Ascend NPU cross-framework run START $(date -Is) ==="
echo "=== repo=$DBF_REPO_DIR frameworks=[$DBF_FRAMEWORKS] hw=$DBF_HARDWARE_PROFILE npu=$GPUS ==="
port="${BASE_PORT:-45001}"
for case_id in $CASES; do
    echo "=== $(date -Is) case $case_id ==="
    bash "${DBF_REPO_DIR}/scripts/devbox_run_cases.sh" \
        "${TAG}_${case_id}" "$GPUS" "$port" "single_e2e throughput" "$case_id"
    grep -h "^RESULT" "${DBF_LOG_DIR}/run_${TAG}_${case_id}.log" 2>/dev/null
    port=$((port + 4))
done
echo "=== Ascend NPU cross-framework run DONE $(date -Is) ==="
