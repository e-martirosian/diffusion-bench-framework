#!/usr/bin/env bash
# run_plan_ab.sh must hand the case runner each arm's framework, profile, modes, GPUs and port.
#
# An arm's profile reaches the harness only through an environment variable, and a wrong or empty one
# does not fail: the harness auto-selects, and the arm measures the wrong command under the right name
# (the 2026-11-16 H3 throughput pair exists because auto-selection compared sglang U2 with vLLM-Omni
# TP2). run_b300_ab_20261022.sh once lost every arm's environment to a comment between a `\`
# continuation and its command. The check runs the script against a stub runner that reports what it got.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/repo/scripts" "$TMP/state/sgl/python/sglang" "$TMP/bin"
cp scripts/gpu_job_lock.sh "$TMP/repo/scripts/"
: > "$TMP/state/sgl/python/sglang/__init__.py"
cat > "$TMP/repo/scripts/devbox_run_cases.sh" <<'EOF'
#!/usr/bin/env bash
echo "STUB tag=$1 gpus=$2 port=$3 modes=[$4] case=$5 fw=${DBF_FRAMEWORKS:-} hw=${DBF_HARDWARE_PROFILE:-} sg=[${SGLANG_BENCH_SGLANG_PROFILE-unset}] vo=[${DIFFUSION_BENCH_VLLM_OMNI_PROFILE-unset}] gen=[${DIFFUSION_BENCH_FRAMEWORK_PROFILE-unset}] extra=[${EXTRA_FLAG:-}]"
EOF
: > "$TMP/repo/scripts/probe_mp4_encode.py"
for tool in nvidia-smi top free; do printf '#!/usr/bin/env bash\nexit 0\n' > "$TMP/bin/$tool"; done
chmod +x "$TMP/bin/"*
cat > "$TMP/plan.txt" <<'EOF'
# comment line and a blank one are skipped

d1 sglang minimax_h3_t2va_5s throughput 0,1 dp2-throughput
v1 vllm-omni minimax_h3_t2va_5s throughput 0,1 replica2-throughput EXTRA_FLAG=on
a1 sglang minimax_h3_t2va_5s single_e2e+throughput 0,1 -
EOF

out="$(PATH="$TMP/bin:$PATH" DBF_STATE_DIR="$TMP/state" DBF_REPO_DIR="$TMP/repo" DBF_LOG_DIR="$TMP/logs" \
    GPU_LOCK_DIR="$TMP/lock" PLAN="$TMP/plan.txt" PLAN_TAG=t PLAN_PORT=61100 \
    bash scripts/run_plan_ab.sh 2>&1)"

fail=0
expect() {  # <label> <fixed string the STUB line must contain>
    if grep '^STUB ' <<<"$out" | grep -qF -- "$2"; then
        echo "  ok   $1"
    else
        echo "  FAIL $1: no STUB line contains: $2"
        fail=1
    fi
}
expect "sglang arm selects its profile by name" "tag=t_d1 gpus=0,1 port=61100 modes=[throughput] case=minimax_h3_t2va_5s fw=sglang hw=b300 sg=[dp2-throughput] vo=[unset] gen=[]"
expect "vllm-omni arm selects its profile through its own variable, plus the arm's extra env" "tag=t_v1 gpus=0,1 port=61110 modes=[throughput] case=minimax_h3_t2va_5s fw=vllm-omni hw=b300 sg=[] vo=[replica2-throughput] gen=[] extra=[on]"
expect "'-' leaves selection to the hardware class and '+' joins modes" "tag=t_a1 gpus=0,1 port=61120 modes=[single_e2e throughput] case=minimax_h3_t2va_5s fw=sglang hw=b300 sg=[] vo=[unset] gen=[]"
stubs="$(grep -c '^STUB ' <<<"$out")"
if [ "$stubs" -ne 3 ]; then echo "  FAIL expected 3 arms, ran $stubs"; fail=1; fi
exit "$fail"
