#!/usr/bin/env bash
# Work through a directory of plans on a devbox, in name order, picking up files added while it runs, so the box
# does not sit idle between an analysis and the next plan. A `<name>.plan` runs through scripts/run_plan_ab.sh with
# PLAN_TAG=<name minus its NN_ prefix>; a `<name>.sh` is run as is (it takes the GPU lock itself). Each finished item
# gets a `<name>.done` marker and a `QUEUE DONE <name>` line; the runner exits when the directory holds a `STOP` file
# and nothing is left to run.
#
#   DBF_STATE_DIR=/workspace/r1118 setsid -f bash scripts/run_plan_queue.sh > /workspace/r1118/logs-ab/queue.log 2>&1 < /dev/null
#   # then drop 10_first.plan, 20_second.plan, ... into $DBF_STATE_DIR/queue; `touch $DBF_STATE_DIR/queue/STOP` to end it
set -u

STATE="${DBF_STATE_DIR:?DBF_STATE_DIR}"
export DBF_STATE_DIR="${STATE}"
QUEUE="${STATE}/queue"
LOGS="${DBF_LOG_DIR:-${STATE}/logs-ab}"
mkdir -p "${QUEUE}" "${LOGS}"
echo "=== QUEUE START $(date -u +%FT%TZ) dir=${QUEUE}"
while :; do
  next=""
  for item in "${QUEUE}"/*.plan "${QUEUE}"/*.sh; do
    [[ -e "${item}" && ! -e "${item%.*}.done" ]] || continue
    if [[ -z "${next}" || "$(basename "${item}")" < "$(basename "${next}")" ]]; then next="${item}"; fi
  done
  if [[ -z "${next}" ]]; then
    [[ -e "${QUEUE}/STOP" ]] && break
    sleep 20
    continue
  fi
  name="$(basename "${next%.*}")"
  tag="${name#[0-9][0-9]_}"
  echo "=== QUEUE RUN ${name} $(date -u +%FT%TZ)"
  if [[ "${next}" == *.plan ]]; then
    PLAN="${next}" PLAN_TAG="${tag}" bash "${STATE}/dbf/scripts/run_plan_ab.sh" > "${LOGS}/${tag}.log" 2>&1
  else
    bash "${next}" > "${LOGS}/${tag}.log" 2>&1
  fi
  echo "rc=$?" > "${next%.*}.done"
  echo "=== QUEUE DONE ${name} $(cat "${next%.*}.done") $(date -u +%FT%TZ)"
done
echo "=== QUEUE END $(date -u +%FT%TZ)"
