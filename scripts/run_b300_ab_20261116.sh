#!/usr/bin/env bash
# Competitor A/B pass of the 2026-11-16 round (RunPod 4xB300), before the matrix. Each arm is
# a candidate command from the round's recipe audit (scripts/check_competitor_recipes.py), measured
# against the profile B300 currently selects ("b300"). A winner is promoted into the case's Blackwell
# profile, then scripts/run_b300_full_20261116.sh measures every cell through its profile.
#
#   bash scripts/run_b300_ab_20261116.sh
#   WAVE=1 bash scripts/run_b300_ab_20261116.sh          # only the lines after `wave 1`
#   AB_DRY_RUN=1 bash scripts/run_b300_ab_20261116.sh    # list them, run nothing
#
# Same machinery as scripts/run_b300_ab_20261022.sh: WAVE selects a section, AB_CASES (a regex on
# "<case> <framework>") narrows it, every line keeps its port. Arms print as
# `ARM <case> <framework> <arm> r<k> RESULT ...`; results land in
# $DBF_LOG_DIR/run_<TAG>_w<wave>_<framework>_<arm>_r<k>_<case>.json.
set -u

export DBF_STATE_DIR="${DBF_STATE_DIR:-/workspace/r1116}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-${DBF_STATE_DIR}/dbf}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs-ab}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-${DBF_STATE_DIR}/hf}"
export DBF_HF_TOKEN_FILE="${DBF_HF_TOKEN_FILE:-${DBF_STATE_DIR}/.hftoken}"
export DBF_SGLANG_PYTHONPATH="${DBF_SGLANG_PYTHONPATH:-${DBF_STATE_DIR}/sgl/python}"
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-7200}"
export DIFFUSION_BENCH_COMFYUI_WORKSPACE="${DIFFUSION_BENCH_COMFYUI_WORKSPACE:-${DBF_STATE_DIR}/comfy-ws}"

if [[ -z "${AB_DRY_RUN:-}" ]]; then
  # shellcheck source=scripts/gpu_job_lock.sh
  source "${DBF_REPO_DIR}/scripts/gpu_job_lock.sh"
  gpu_lock_acquire
fi

wave=1
wave() { wave=$1; }  # the section the following arms lines belong to

arms() {  # <case> <framework> <rounds> <arms...>; WAVE=<n> / AB_CASES=<regex> select a subset
  local case_id=$1 fw=$2 rounds=$3
  shift 3
  if [[ -n "${WAVE:-}" && "${wave}" != "${WAVE}" ]] \
    || [[ -n "${AB_CASES:-}" && ! "${case_id} ${fw}" =~ ${AB_CASES} ]]; then
    port=$((port + 100))
    return
  fi
  if [[ -n "${AB_DRY_RUN:-}" ]]; then
    echo "PLAN wave=${wave} port=${port} ${case_id} ${fw} rounds=${rounds} arms=$*"
  else
    # the wave is in the tag: two waves arming one cell would otherwise overwrite each other's results
    ARMS_CASE="${case_id}" ARMS_FW="${fw}" ARMS="$*" ROUNDS="${rounds}" \
      bash "${DBF_REPO_DIR}/scripts/run_competitor_arms.sh" "${TAG}_w${wave}_${fw}" "${port}"
  fi
  port=$((port + 100))
}

export ARMS_HW=b300 GPUS=0,1,2,3 TAG=ab1116b300
port=40000  # 100 ports per arms line; keep every base under 65535
echo "=== AB START $(date -Is) | $(git -C "${DBF_REPO_DIR}" log --oneline -1 2>/dev/null) | WAVE=${WAVE:-all} ==="
wave 1
# Options the competitors' recipes gained since the last round, one round each (the cells are ~30-75 s a request).
# vLLM-Omni #4637: Cosmos3 pipeline parallel, validated for T2V/I2V. 4 GPUs, so PP2 pairs with CFG 2 or Ulysses 2.
arms cosmos3_nano_t2v_720p_189f vllm-omni 1 b300 cand-pp2-cfg2 cand-pp2-usp2
# TRT-LLM #20176: MiniMax-H3 DiT tensor parallelism on two GPUs (the Ulysses profile is the baseline).
arms minimax_h3_t2va_5s trtllm-visual 1 b300 cand-tp2 cand-tp2-vae2
echo "=== AB DONE $(date -Is) ==="
