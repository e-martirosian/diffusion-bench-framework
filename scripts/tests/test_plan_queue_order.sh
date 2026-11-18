#!/usr/bin/env bash
# run_plan_queue.sh runs plans in name order, each once, picks up a plan added while it runs, and stops on STOP.
# A queue that re-ran a finished plan or skipped a late one would overwrite or lose a night of arms.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/state/dbf/scripts" "$TMP/state/queue"
cat > "$TMP/state/dbf/scripts/run_plan_ab.sh" <<'EOF'
#!/usr/bin/env bash
echo "RAN tag=${PLAN_TAG} plan=$(basename "${PLAN}")" >> "${DBF_STATE_DIR}/ran.txt"
# the first plan adds a later one and the stop marker while the queue is running
if [[ "${PLAN_TAG}" == first ]]; then
  : > "${DBF_STATE_DIR}/queue/15_added.plan"
  : > "${DBF_STATE_DIR}/queue/STOP"
fi
EOF
: > "$TMP/state/queue/20_second.plan"
: > "$TMP/state/queue/10_first.plan"
printf 'echo "RAN script" >> "%s/state/ran.txt"\n' "$TMP" > "$TMP/state/queue/30_third.sh"

out="$(DBF_STATE_DIR="$TMP/state" bash scripts/run_plan_queue.sh 2>&1)"
got="$(tr '\n' '|' < "$TMP/state/ran.txt")"
want="RAN tag=first plan=10_first.plan|RAN tag=added plan=15_added.plan|RAN tag=second plan=20_second.plan|RAN script|"
fail=0
if [[ "$got" == "$want" ]]; then echo "  ok   plans ran once each, in name order, including the one added mid-run"; else echo "  FAIL order: $got"; fail=1; fi
if grep -q "QUEUE END" <<<"$out"; then echo "  ok   the queue ended on STOP"; else echo "  FAIL no QUEUE END"; fail=1; fi
markers="$(ls "$TMP/state/queue" | grep -c '\.done$')"
if [[ "$markers" == 4 ]]; then echo "  ok   one done marker per item"; else echo "  FAIL $markers done markers"; fail=1; fi
exit "$fail"
