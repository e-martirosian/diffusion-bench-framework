#!/usr/bin/env bash
# The 2026-11-16 round on one RunPod 4xB300 (Blackwell Ultra, 288 GB each; no H200
# could be placed: the whole H200 pool had one free card): every case except the
# consumer one, one harness revision, latest-vs-latest, all six frameworks.
#
# Wraps scripts/run_b200_cross_framework_20260912.sh. One box, the two halves in
# turn (never at once: shared host and PCIe), every framework on the cells' own GPUs:
#   ROLE=image: image cases and the 2-GPU LTX video (LTX-2.5).
#   ROLE=video: MiniMax-H3, Cosmos3 189f and Wan2.2.
# minimax_h3_t2va_480p_1gpu is the single-32 GB consumer case and is not run here.
#
# Image lmsysorg/sglang:dev; sglang runs from a checkout of the round's pin through
# DBF_SGLANG_PYTHONPATH (the result records that sha); framework venvs, HF cache,
# logs and the harness live on /workspace (ephemeral: copy results off per half).
# Competitors were installed by scripts/setup_round_20261116.sh at the versions it
# records (vllm 0.33.0, vllm-omni/LightX2V/ComfyUI/FastVideo at main, tensorrt-llm
# 1.4.0rc3).
#
#   SGLANG_PIN=<sha> ROLE=image bash scripts/run_b300_full_20261116.sh
#   SGLANG_PIN=<sha> ROLE=video bash scripts/run_b300_full_20261116.sh
#
# Then merge both halves' results and publish -- dry run first:
#   scripts/merge_and_publish_run.sh "<logs>/run_b300*1116_*.json" \
#       b300x4-full-20261116 "4xB300 cross-framework (latest-vs-latest)" \
#       "4x NVIDIA B300 288GB" scripts/run_b300_full_20261116.sh
set -u

ROLE="${ROLE:?ROLE=image or ROLE=video}"
export DBF_STATE_DIR="${DBF_STATE_DIR:-/workspace/r1116}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-${DBF_STATE_DIR}/dbf}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-${DBF_STATE_DIR}/hf}"
export DBF_HF_TOKEN_FILE="${DBF_HF_TOKEN_FILE:-${DBF_STATE_DIR}/.hftoken}"
export DBF_SGLANG_PYTHONPATH="${DBF_SGLANG_PYTHONPATH:-${DBF_STATE_DIR}/sgl/python}"
export DBF_FRAMEWORKS="${DBF_FRAMEWORKS:-sglang vllm-omni lightx2v trtllm-visual comfyui fastvideo}"
export DBF_HARDWARE_PROFILE="${DBF_HARDWARE_PROFILE:-b300}"
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-21600}"
export DIFFUSION_BENCH_COMFYUI_WORKSPACE="${DIFFUSION_BENCH_COMFYUI_WORKSPACE:-${DBF_STATE_DIR}/comfy-ws}"
case "${ROLE}" in
  image)
    export GPUS="${GPUS:-0,1,2,3}" TAG="${TAG:-b300img1116}" BASE_PORT="${BASE_PORT:-61000}"
    export CASES="${CASES:-
qwen_image_21_t2i_1024
qwen_image_21_t2i_1024_2gpu
qwen_image_21_edit_1024
zimage_turbo_t2i_1024
cosmos3_nano_t2i_720p
flux3_klein_base_t2i_1024
ideogram4_t2i_1024_2gpu_tp
ltx2.5_twostage_t2v_2gpus
}" ;;
  video)
    export GPUS="${GPUS:-0,1,2,3}" TAG="${TAG:-b300vid1116}" BASE_PORT="${BASE_PORT:-62000}"
    export CASES="${CASES:-
minimax_h3_t2va_5s
minimax_h3_ref2va_5s
minimax_h3_t2va_5s_4gpu
minimax_h3_fasth3_v2_t2va_5s
cosmos3_nano_t2v_720p_189f
cosmos3_nano_i2v_720p_189f
wan22_t2v_a14b_720p
}" ;;
  *) echo "ROLE must be image or video" >&2; exit 1 ;;
esac

cd "${DBF_REPO_DIR}" || exit 1
sglang_tree="${DBF_SGLANG_PYTHONPATH%/python}"
if [[ "$(git -C "${sglang_tree}" rev-parse HEAD)" != "${SGLANG_PIN:?set SGLANG_PIN to the round sglang sha}"* ]]; then
  echo "sglang tree is at $(git -C "${sglang_tree}" rev-parse --short HEAD), not the round pin ${SGLANG_PIN:0:12}" >&2
  exit 1
fi
# The runner clears the cards between cases. A job that takes the lock (the A/B and plan runners do) and
# starts meanwhile would not wait for an unlocked matrix and would kill its servers: this round's matrix
# held no lock and was only kept apart by launching those jobs after it had finished.
# shellcheck source=scripts/gpu_job_lock.sh
source "${DBF_REPO_DIR}/scripts/gpu_job_lock.sh"
gpu_lock_acquire
echo "=== ${ROLE} | $(git log --oneline -1) | sglang $(git -C "${sglang_tree}" log --oneline -1) ==="
bash "${DBF_REPO_DIR}/scripts/run_b200_cross_framework_20260912.sh"   # not exec: the EXIT trap must release the lock
