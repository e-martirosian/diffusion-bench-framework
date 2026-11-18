#!/usr/bin/env bash
# bisect_sglang_cell.sh lands on the first slow commit, skips a commit that cannot be measured, and can bisect a speedup.
# A wrong answer here names the wrong PR as a regression, so the search is pinned against a history whose answer is known.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
STATE="$TMP/state"
mkdir -p "$STATE/dbf/scripts" "$STATE/logs-ab"
git init -q "$STATE/sgl-full"
g() { git -C "$STATE/sgl-full" -c user.email=t@t -c user.name=t "$@"; }
mkdir -p "$STATE/sgl-full/python/sglang"
# commit i writes latency L_i; c0 is the good endpoint. c1..c9: slow from c6 on; c4 cannot be measured.
lat=(0.500 0.501 0.499 0.500 fail 0.502 0.530 0.529 0.531 0.530)
for i in "${!lat[@]}"; do
  echo "${lat[i]}" > "$STATE/sgl-full/python/sglang/latency.txt"
  g add -A && g commit -q --allow-empty -m "c$i"
  eval "c$i=$(g rev-parse HEAD)"
done
# the stub "measures" by reading the checked-out tree's latency file
cat > "$STATE/dbf/scripts/run_plan_ab.sh" <<'EOF'
#!/usr/bin/env bash
read -r arm fw case_id modes gpus prof envs < "${PLAN}"
tree="${envs#DBF_SGLANG_PYTHONPATH=}"
value="$(cat "${tree}/sglang/latency.txt")"
[[ "${value}" == fail ]] && exit 0
printf '{"results": [{"framework": "sglang", "latency_s": %s}]}\n' "${value}" > "${DBF_STATE_DIR}/logs-ab/run_${PLAN_TAG}_${arm}_${case_id}.json"
EOF

run() { DBF_STATE_DIR="$STATE" BISECT_CASE=cell BISECT_GPUS=0 BISECT_TAG="$1" BISECT_GOOD="$2" BISECT_BAD="$3" BISECT_THRESHOLD="$4" BISECT_FASTER_IS="${5:-good}" bash scripts/bisect_sglang_cell.sh 2>&1; }
fail=0
out="$(run a "$c0" "$c9" 0.515)"
if grep -q "^FIRST bad ${c6:0:12}" <<<"$out" && grep -q "^LAST good ${c5:0:12}" <<<"$out"; then echo "  ok   the first slow commit is found, with the last fast one before it"; else echo "  FAIL regression bisect:"; echo "$out" | tail -6; fail=1; fi
if grep -q "fail (skipped)" <<<"$out" || ! grep -q "PROBE ${c4:0:10}" <<<"$out"; then echo "  ok   an unmeasurable commit is skipped or never needed"; else echo "  FAIL unmeasurable commit: $out"; fail=1; fi
probes="$(grep -c '^PROBE ' <<<"$out")"
if (( probes >= 2 && probes <= 5 )); then echo "  ok   $probes probes for 9 candidates"; else echo "  FAIL $probes probes"; fail=1; fi
# a speedup: the history reversed in meaning (newer is faster), bisected with BISECT_FASTER_IS=bad
git -C "$STATE/sgl-full" worktree prune
lat2=(0.530 0.531 0.529 0.500 0.501)
for i in "${!lat2[@]}"; do
  echo "${lat2[i]}" > "$STATE/sgl-full/python/sglang/latency.txt"
  g add -A && g commit -q --allow-empty -m "s$i"
  eval "s$i=$(g rev-parse HEAD)"
done
out="$(run b "$s0" "$s4" 0.515 bad)"
if grep -q "^FIRST bad ${s3:0:12}" <<<"$out"; then echo "  ok   a speedup is bisected to the first fast commit"; else echo "  FAIL speedup bisect:"; echo "$out" | tail -5; fail=1; fi
exit "$fail"
