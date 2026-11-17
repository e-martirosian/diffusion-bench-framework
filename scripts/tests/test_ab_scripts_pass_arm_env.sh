#!/usr/bin/env bash
# Every A/B script must hand its arm runner the case, framework, arms and rounds.
#
# run_b300_ab_20261022.sh set them as a prefix to `bash run_competitor_arms.sh`
# but put a comment line between the `\` continuation and the command. The
# comment ended the assignment statement, so the assignments became plain shell
# variables, the runner started with an empty environment and died on its first
# line (`ARMS_CASE: parameter null or not set`) -- for every arm of every wave.
# Nothing in the script said so: the output was one error per arm.
#
# The check runs each script against a stub runner that reports what it received.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/repo/scripts" "$TMP/state"
cp scripts/gpu_job_lock.sh "$TMP/repo/scripts/"
cat > "$TMP/repo/scripts/run_competitor_arms.sh" <<'EOF'
#!/usr/bin/env bash
echo "STUB case=${ARMS_CASE:-} fw=${ARMS_FW:-} arms=${ARMS:-} rounds=${ROUNDS:-}"
EOF

fail=0
for script in scripts/run_*_ab_*.sh; do
    out="$(DBF_STATE_DIR="$TMP/state" DBF_REPO_DIR="$TMP/repo" DBF_LOG_DIR="$TMP/logs" \
        GPU_LOCK_DIR="$TMP/lock-$(basename "$script")" bash "$script" 2>&1)"
    stubs="$(grep -c '^STUB ' <<<"$out")"
    empty="$(grep '^STUB ' <<<"$out" | grep -cE 'case= |fw= |arms= |rounds=$')"
    if [ "$stubs" -gt 0 ] && [ "$empty" -eq 0 ]; then
        echo "  ok   $(basename "$script"): $stubs arm line(s), each with its case, framework, arms and rounds"
    else
        echo "  FAIL $(basename "$script"): $stubs arm line(s), $empty with an empty field"
        grep '^STUB ' <<<"$out" | head -2
        fail=1
    fi
done
exit "$fail"
