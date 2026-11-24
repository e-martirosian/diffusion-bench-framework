#!/usr/bin/env bash
# Ascend A2/A3 (910B/910C) NPU cross-framework run (scaffold). Same harness and
# report shape as the NVIDIA rounds, but on Huawei Ascend NPUs. One script serves
# both NPUs — pick the chip with DBF_HARDWARE_PROFILE.
#
# The runner auto-detects Ascend via `npu-smi info`, records device=npu, and
# appends `--device npu` to the vLLM-Omni serve command only (sglang detects
# npu/gpu itself; its parser rejects --device), plus ASCEND_RT_VISIBLE_DEVICES
# instead of CUDA_VISIBLE_DEVICES. No code
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
export DBF_REPO_DIR="${DBF_REPO_DIR:?set DBF_REPO_DIR to this repo\'s checkout on the NPU box}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-${DBF_STATE_DIR}/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${DBF_HF_HOME}}"
# No HF access? Point DBF_MODEL_PATHS at a JSON file mapping repo id -> local
# model dir (see configs/model_paths.example.json); the runner uses it verbatim
# and skips HuggingFace entirely.
export DBF_MODEL_PATHS="${DBF_MODEL_PATHS:-${DBF_STATE_DIR}/model_paths.json}"
export DBF_FRAMEWORKS="${DBF_FRAMEWORKS:-sglang vllm-omni}"
export DBF_HARDWARE_PROFILE="${DBF_HARDWARE_PROFILE:-a2}"
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-5400}"

# Ascend device ids, like CUDA_VISIBLE_DEVICES ids; the runner maps this to
# ASCEND_RT_VISIBLE_DEVICES on NPU.
GPUS="${GPUS:-0}"
# Tag defaults to the chip (ascenda2 / ascenda3) so one script serves both.
TAG="${TAG:-ascend${DBF_HARDWARE_PROFILE}}"

# Every case whose model has a model_paths.json entry AND whose sglang cell is
# `supported` in its case file (55 of 64, ordered by cases/_order.json).
# Excluded -- sglang `unsupported` there, so no sglang round can ever run them:
# qwen_image_2512_1024 ("MUSA-only checkpoint"), qwen_image_edit_1024,
# flux2_dev_nvfp4_t2i_1024, wan22_t2v_a14b_nvfp4, comfy_ideogram4_t2i,
# kandinsky6_vsr, hunyuan3d2, pi05_base, flux3_action_droid.
# Multi-GPU cases (num_gpus 2/4) need GPUS to cover them; on a smaller GPUS
# they fail fast with a recorded device-count error, not silently.
CASES="${CASES:-
flux3_klein_base_t2i_1024
qwen_image_21_t2i_1024
qwen_image_21_t2i_1024_2gpu
qwen_image_21_edit_1024
zimage_turbo_t2i_1024
ideogram4_t2i_1024_2gpu_tp
wan22_t2v_a14b_720p
ltx2.5_twostage_t2v_2gpus
cosmos3_nano_t2i_720p
cosmos3_nano_t2v_720p_189f
cosmos3_nano_i2v_720p_189f
minimax_h3_t2va_5s
minimax_h3_t2va_5s_4gpu
minimax_h3_t2va_480p_1gpu
minimax_h3_ref2va_5s
minimax_h3_fasth3_v2_t2va_5s
flux1_dev_t2i_1024
flux2_dev_t2i_1024
flux2_klein_4b_t2i_1024
flux3_image_klein_t2i_1024
qwen_image_t2i_1024
qwen_image_edit_2509_1024
qwen_image_edit_2511_1024
qwen_image_layered_1024
glm_image_t2i_1024
zimage_t2i_1024
ernie_image_t2i_1024
hunyuan_image_30_t2i_1024
hunyuan_image_30_instruct_t2i_1024
joyai_image_edit_1024
joyai_echo_1024
llada_image_t2i_1024
hidream_o1_image_t2i_1024
anima_base_t2i_1024
ltx2_t2v_720p
ltx23_t2v_720p
vdn_minimax_h3_t2v
helios_base_t2v
helios_distilled_t2v
helios_mid_t2v
pika_h3_flex_t2v
fasth3_omni_t2v
wan21_t2v_13b_720p
wan21_t2v_14b_720p
wan21_i2v_14b_480p
wan22_i2v_a14b_720p
wan22_ti2v_5b_720p
wan22_ti2v_5b_fullattn
turbowan21_t2v_13b
hunyuanvideo_t2v_720p
fasthunyuan_t2v
longlive2_t2v
mova_360p_t2v
lingbot_video_moe_t2v
lingbot_world_t2v
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
