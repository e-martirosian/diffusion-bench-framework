#!/usr/bin/env bash
# Bisect sglang main for the commit that moved one cell's latency, on one devbox, through the harness: every probe is
# a fresh server on a checkout of the candidate commit, measured like any arm (scripts/run_plan_ab.sh).
#
#   DBF_STATE_DIR=/workspace/r1118 BISECT_CASE=cosmos3_nano_t2i_720p BISECT_GPUS=0,1 \
#     BISECT_GOOD=<sha> BISECT_BAD=<sha> BISECT_THRESHOLD=0.5195 BISECT_TAG=c3i \
#     bash scripts/bisect_sglang_cell.sh
#
# BISECT_THRESHOLD is a single_e2e median in seconds between the two endpoints' values: a probe at or below it counts
# as good (set BISECT_FASTER_IS=bad when the newer side is the faster one). Measure both endpoints on this box first
# and leave the threshold well clear of the arm-to-arm noise. Commits are the first-parent ones that touch
# BISECT_PATHS (default python/sglang) in good..bad. Needs $DBF_STATE_DIR/sgl-full (scripts/setup_sglang_trees.sh).
# Prints `PROBE <sha> <latency> good|bad <subject>` per step and `FIRST <good|bad> <sha> ...` at the end.
set -u

STATE="${DBF_STATE_DIR:?DBF_STATE_DIR}"
export DBF_STATE_DIR="${STATE}"
CASE="${BISECT_CASE:?BISECT_CASE}"; GPUS="${BISECT_GPUS:?BISECT_GPUS}"
GOOD="${BISECT_GOOD:?BISECT_GOOD}"; BAD="${BISECT_BAD:?BISECT_BAD}"
THRESHOLD="${BISECT_THRESHOLD:?BISECT_THRESHOLD}"; TAG="${BISECT_TAG:?BISECT_TAG}"
FASTER_IS="${BISECT_FASTER_IS:-good}"
PATHS="${BISECT_PATHS:-python/sglang}"
FULL="${STATE}/sgl-full"
TREE="${STATE}/sgl-bisect-${TAG}"
REPO="${DBF_REPO_DIR:-${STATE}/dbf}"
LOGS="${DBF_LOG_DIR:-${STATE}/logs-ab}"

commits=()  # no mapfile: macOS ships bash 3.2
while IFS= read -r sha; do commits+=("${sha}"); done < <(git -C "${FULL}" rev-list --first-parent --reverse "${GOOD}..${BAD}" -- ${PATHS})
echo "BISECT ${CASE} good=${GOOD:0:10} bad=${BAD:0:10} threshold=${THRESHOLD} candidates=${#commits[@]}"
[[ ${#commits[@]} -gt 0 ]] || { echo "no commits in range"; exit 1; }
[[ -d "${TREE}" ]] || git -C "${FULL}" -c http.https://github.com/.extraheader= worktree add -q --detach "${TREE}" "${GOOD}" || exit 1

probe() {  # <sha> -> prints the median latency, or "fail"
  local sha=$1 plan
  git -C "${TREE}" -c http.https://github.com/.extraheader= checkout -q --detach "${sha}" || { echo fail; return; }
  plan="$(mktemp)"
  echo "p-${sha:0:10} sglang ${CASE} single_e2e ${GPUS} - DBF_SGLANG_PYTHONPATH=${TREE}/python" > "${plan}"
  PLAN="${plan}" PLAN_TAG="bis${TAG}" bash "${REPO}/scripts/run_plan_ab.sh" > "${LOGS}/bis${TAG}_${sha:0:10}.log" 2>&1
  rm -f "${plan}"
  python3 - "${LOGS}/run_bis${TAG}_p-${sha:0:10}_${CASE}.json" <<'PY'
import json, sys
try:
    rows = [r for r in json.load(open(sys.argv[1]))["results"] if r["framework"] == "sglang" and r.get("latency_s")]
    print(rows[0]["latency_s"] if rows else "fail")
except (OSError, ValueError, KeyError):
    print("fail")
PY
}

lo=-1                     # index of the last commit known to behave like GOOD (-1: GOOD itself)
hi=$(( ${#commits[@]} - 1 ))   # index of the first commit known to behave like BAD (BAD is the last candidate)
while (( hi - lo > 1 )); do
  mid=$(( (lo + hi) / 2 ))
  sha="${commits[mid]}"
  lat="$(probe "${sha}")"
  if [[ "${lat}" == fail ]]; then
    echo "PROBE ${sha:0:10} fail (skipped) $(git -C "${FULL}" log -1 --format=%s "${sha}" | cut -c1-80)"
    # a commit that cannot be measured is dropped from the candidates
    unset 'commits[mid]'; commits=("${commits[@]}"); hi=$(( hi - 1 ))
    continue
  fi
  if python3 -c "import sys; faster = float('${lat}') <= float('${THRESHOLD}'); sys.exit(0 if faster == ('${FASTER_IS}' == 'good') else 1)"; then verdict=good; lo=${mid}; else verdict=bad; hi=${mid}; fi
  echo "PROBE ${sha:0:10} ${lat} ${verdict} $(git -C "${FULL}" log -1 --format='%cs %s' "${sha}" | cut -c1-90)"
done
echo "FIRST bad ${commits[hi]:0:12} $(git -C "${FULL}" log -1 --format='%cs %s' "${commits[hi]}" | cut -c1-110)"
if (( lo >= 0 )); then echo "LAST good ${commits[lo]:0:12} $(git -C "${FULL}" log -1 --format='%cs %s' "${commits[lo]}" | cut -c1-110)"; else echo "LAST good ${GOOD:0:12} (the good endpoint)"; fi
