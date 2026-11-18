#!/usr/bin/env bash
# The 2026-11-18 supplementary experiments, one after another on one box, each step under the GPU lock (queued behind the PR A/B plan):
#   1. cuDNN SDPA microbench in four torch/cuDNN combinations (the cuDNN wheel is swapped inside the two scratch venvs, then restored)
#   2. the competitor fairness arms, round 1 (scripts/plan_b300_fair_r1_20261118.txt)
#   3. the vLLM-Omni commit bucket for Z-Image (O0 O1 N1 N2 P C) and #10120 alone for Qwen-Image-2.1 (P C)
#
#   DBF_STATE_DIR=/workspace/r1117 setsid -f bash scripts/chain_attribution_20261118.sh > /workspace/r1117/logs-ab/chain.log 2>&1 < /dev/null
set -u

STATE="${DBF_STATE_DIR:?DBF_STATE_DIR}"
export DBF_STATE_DIR="${STATE}"
EXTRA="${STATE}/extra"
LOGS="${STATE}/logs-ab"
CUDNN_OLD=9.20.0.48
CUDNN_NEW=9.24.0.43
mkdir -p "${LOGS}"

# shellcheck source=scripts/gpu_job_lock.sh
source "${STATE}/dbf/scripts/gpu_job_lock.sh"

bench() {  # <label> <venv>
  CUDA_VISIBLE_DEVICES=0 "$2/bin/python" "${EXTRA}/bench_cudnn_sdpa.py" --iters 30 > "${LOGS}/cudnn_$1.json" 2> "${LOGS}/cudnn_$1.err"
  echo "CUDNN $1 $(tail -c 600 "${LOGS}/cudnn_$1.json" | tr -d '\n' | cut -c1-520)"
}
swap() {  # <venv> <cudnn version>
  uv pip install -q --python "$1/bin/python" --no-deps "nvidia-cudnn-cu13==$2" 2>&1 | tail -1
}

echo "=== CHAIN START $(date -u +%FT%TZ)"
gpu_lock_acquire
V213="${STATE}/venv-torch2130"; V214="${STATE}/venv-torch2140"
for pass in 1 2; do
  bench "t213_c920_p${pass}" "${V213}"
  bench "t214_c924_p${pass}" "${V214}"
  swap "${V214}" "${CUDNN_OLD}"; bench "t214_c920_p${pass}" "${V214}"; swap "${V214}" "${CUDNN_NEW}"
  swap "${V213}" "${CUDNN_NEW}"; bench "t213_c924_p${pass}" "${V213}"; swap "${V213}" "${CUDNN_OLD}"
done
gpu_lock_release
echo "=== CUDNN DONE $(date -u +%FT%TZ)"

DBF_STATE_DIR="${STATE}" PLAN="${EXTRA}/plan_b300_fair_r1_20261118.txt" PLAN_TAG=fair1 bash "${STATE}/dbf/scripts/run_plan_ab.sh" > "${LOGS}/fair1.log" 2>&1
echo "=== FAIR1 DONE $(date -u +%FT%TZ)"

DBF_STATE_DIR="${STATE}" ROUNDS=3 CASES=zimage_turbo_t2i_1024:0,1 bash "${EXTRA}/run_vllm_omni_bucket_20261118.sh" > "${LOGS}/bucket_z.log" 2>&1
echo "=== BUCKET Z DONE $(date -u +%FT%TZ)"
DBF_STATE_DIR="${STATE}" ROUNDS=3 CASES=qwen_image_21_t2i_1024:0 ONLY="P C" bash "${EXTRA}/run_vllm_omni_bucket_20261118.sh" > "${LOGS}/bucket_q.log" 2>&1
echo "=== BUCKET Q DONE $(date -u +%FT%TZ)"
echo "=== CHAIN DONE $(date -u +%FT%TZ)"
