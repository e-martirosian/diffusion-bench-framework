#!/usr/bin/env bash
# Interleaved harness arms from a plan file, on one devbox: a fresh server per arm, one GPU-lock hold per
# invocation, a host/GPU state snapshot before and after every arm, a 1 Hz nvidia-smi trace and MP4 md5s.
# The state they record is what explains a drift between arms of the same cell. Generalised from
# scripts/b300_supp_20261102/run_supp_ab.sh, which is hard-wired to that round's box and install specs.
#
#   DBF_STATE_DIR=/workspace/r1116 PLAN=<file> PLAN_TAG=<tag> [PLAN_PORT=<base port>] \
#     setsid -f bash scripts/run_plan_ab.sh > <state>/logs-ab/<tag>.log 2>&1 < /dev/null
#
# A plan line is "<arm> <framework> <case> <modes> <gpus> <profile|-> [ENV=VAL ...]"; '#' starts a comment.
# modes join with '+' (single_e2e+throughput). A '-' profile is auto-selection for the hardware class
# (DBF_HARDWARE_PROFILE, default b300); any other value selects that profile by name for the arm's framework.
# Results: $DBF_LOG_DIR/run_<tag>_<arm>_<case>.json, one `ARM ...` line per run. The same layout variables
# as scripts/devbox_run_cases.sh apply (DBF_REPO_DIR, DBF_LOG_DIR, DBF_VENV_ROOT, DBF_HF_HOME,
# DBF_HF_TOKEN_FILE, DBF_SGLANG_PYTHONPATH), defaulting to the round's state dir.
set -u

STATE="${DBF_STATE_DIR:?DBF_STATE_DIR, e.g. /workspace/r1116}"
export DBF_STATE_DIR="${STATE}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-${STATE}/dbf}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${STATE}/logs-ab}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${STATE}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-${STATE}/hf}"
export DBF_HF_TOKEN_FILE="${DBF_HF_TOKEN_FILE:-${STATE}/.hftoken}"
export DBF_SGLANG_PYTHONPATH="${DBF_SGLANG_PYTHONPATH:-${STATE}/sgl/python}"
export DBF_HARDWARE_PROFILE="${DBF_HARDWARE_PROFILE:-b300}"
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-7200}"
export DIFFUSION_BENCH_KEEP_VIDEO_DIR="${DBF_LOG_DIR}/kept-mp4"

TAG="${PLAN_TAG:?PLAN_TAG}"; PLAN="${PLAN:?PLAN file}"
mkdir -p "${DBF_LOG_DIR}" "${DIFFUSION_BENCH_KEEP_VIDEO_DIR}"

# the sglang tree must be the pinned one, not the image's /sgl-workspace
resolved="$(PYTHONPATH="${DBF_SGLANG_PYTHONPATH}" python3 -c 'import os, sglang; print(os.path.realpath(sglang.__file__))')"
[[ "${resolved}" == "$(realpath "${DBF_SGLANG_PYTHONPATH}")/"* ]] || { echo "sglang resolves to ${resolved}"; exit 1; }

# shellcheck source=scripts/gpu_job_lock.sh
source "${DBF_REPO_DIR}/scripts/gpu_job_lock.sh"
gpu_lock_acquire
SMI=""
trap '[[ -n "${SMI}" ]] && kill "${SMI}" 2>/dev/null; gpu_lock_release_if_mine' EXIT

snap() {  # <file> <label>: host and GPU state that could explain a drift between arms
  {
    echo "--- $2 $(date -u +%FT%TZ)"
    uptime
    echo "loadavg: $(cat /proc/loadavg)"
    top -bn1 | sed -n '3p'
    free -g | sed -n '1,2p'
    nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,clocks_throttle_reasons.active --format=csv,noheader
    echo "compute-apps:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
    echo "top-cpu:"; ps -eo pid,pcpu,etime,comm --sort=-pcpu | head -6
  } >> "$1" 2>&1
}

echo "=== PLAN START $(date -Is) tag=${TAG} plan=${PLAN} harness $(git -C "${DBF_REPO_DIR}" log --oneline -1 | cut -c1-70) sglang_tree=$(git -C "${DBF_SGLANG_PYTHONPATH%/python}" rev-parse HEAD)"
port="${PLAN_PORT:-61100}"
while read -r arm fw case_id modes gpus prof envs; do
  [[ -z "${arm}" || "${arm}" == \#* ]] && continue
  modes="${modes//+/ }"
  [[ "${prof}" == - ]] && prof=""
  t="${TAG}_${arm}"
  t0=$(date +%s)
  echo "=== ARM_START ${arm} ${fw} ${case_id} modes=[${modes}] gpus=${gpus} profile=${prof:-auto} port=${port} $(date -u +%FT%TZ)"
  snap "${DBF_LOG_DIR}/state_${t}.txt" PRE
  nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv,noheader -lms 1000 > "${DBF_LOG_DIR}/smi_${t}.csv" &
  SMI=$!
  # sglang reads only SGLANG_BENCH_SGLANG_PROFILE; every other framework its DIFFUSION_BENCH_<FRAMEWORK>_PROFILE
  fw_env="DIFFUSION_BENCH_$(tr 'a-z-' 'A-Z_' <<<"${fw}")_PROFILE"
  if [[ "${fw}" == sglang ]]; then sp="${prof}"; fp=""; else sp=""; fp="${prof}"; fi
  env ${envs} DBF_FRAMEWORKS="${fw}" SGLANG_BENCH_SGLANG_PROFILE="${sp}" "${fw_env}=${fp}" DIFFUSION_BENCH_FRAMEWORK_PROFILE="" \
    bash "${DBF_REPO_DIR}/scripts/devbox_run_cases.sh" "${t}" "${gpus}" "${port}" "${modes}" "${case_id}" < /dev/null
  kill "${SMI}" 2>/dev/null; SMI=""
  snap "${DBF_LOG_DIR}/state_${t}.txt" POST
  out=$(python3 - "${DBF_LOG_DIR}/run_${t}_${case_id}.json" "${fw}" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError) as exc:
    print(f"no-result({exc.__class__.__name__})"); sys.exit()
parts = []
sha = ((d.get("sglang_runtime") or {}).get("git_commit") or "")[:10]
for r in d.get("results", []):
    if r.get("framework") == sys.argv[2]:
        m, fm = r.get("metrics") or {}, r.get("framework_metadata") or {}
        parts.append(f"single={r.get('latency_s')} samples={m.get('latency_samples_s')} srv={m.get('server_latency_s')} startup={m.get('server_startup_s')} profile={fm.get('profile')}({fm.get('profile_source')}) sha={sha}"
                     + (f" error={(r.get('error') or '')[:200]!r}" if r.get("error") else ""))
for r in d.get("throughput_results", []):
    if r.get("framework") == sys.argv[2]:
        m, fm = r.get("metrics") or {}, r.get("framework_metadata") or {}
        parts.append(f"tput rps={m.get('throughput_rps')} dur={m.get('duration_s')} p50={m.get('latency_p50_s')} done={m.get('completed_requests')}/{m.get('num_requests')} profile={fm.get('profile')}({fm.get('profile_source')}) sha={sha}"
                     + (f" error={(r.get('error') or '')[:200]!r}" if r.get("error") else ""))
print(" | ".join(parts) or "no-row")
PYEOF
)
  echo "ARM ${arm} ${fw} ${case_id} $(( $(date +%s) - t0 ))s ${out}"
  python3 "${DBF_REPO_DIR}/scripts/probe_mp4_encode.py" --group --since "${t0}" --label "${arm}" \
    "${DBF_REPO_DIR}/outputs" "${DIFFUSION_BENCH_KEEP_VIDEO_DIR}" >> "${DBF_LOG_DIR}/mp4_probe_${TAG}.jsonl" 2>/dev/null
  find "${DBF_REPO_DIR}/outputs" "${DIFFUSION_BENCH_KEEP_VIDEO_DIR}" -name '*.mp4' -newermt "@${t0}" -exec md5sum {} + 2>/dev/null | sed "s#^#${arm} #" >> "${DBF_LOG_DIR}/md5_${TAG}.txt"
  port=$((port + 10))
done < "${PLAN}"
echo "=== PLAN DONE $(date -Is) tag=${TAG} ==="
