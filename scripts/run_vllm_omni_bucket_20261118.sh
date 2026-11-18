#!/usr/bin/env bash
# Which vLLM-Omni change made Z-Image TP2 (and the image cells) faster between 2026-10-22 and 2026-11-16? Interleaved arms, a fresh server per arm:
#   O0 0da1d831e  old stack  (vllm 0.32.0, torch 2.13.0; the 2026-10-22 pin)
#   O1 670aa7ec6  old stack  (the parent of the vllm 0.33.0 alignment commit)
#   N1 4b63581cf  new stack  (vllm 0.33.0, torch 2.14.0; the alignment commit)
#   N2 24e547284  new stack  (the 2026-11-16 pin)
#   P  702917be6  new stack  (the parent of #10120, the image response path rewrite)
#   C  eb8b991ff  new stack  (#10120)
# O1-O0 is vllm-omni code before the stack move, N1-O1 the stack move, N2-N1 code after it, C-P #10120 alone.
# A commit is switched by checking it out in the venv's editable source directory (vllm-omni is pure python), so the venv roots must exist:
# fw-venvs (the round's) and fw-venvs-0da1 (scripts/setup_attribution_20261118.sh, step old). The round's pin is restored on exit.
#
# ONLY="P C" keeps just those arms (labels above), CASES="<case>:<gpus> ..." picks the cells.
#
#   DBF_STATE_DIR=/workspace/r1117 ROUNDS=3 CASES=zimage_turbo_t2i_1024:0,1 setsid -f bash scripts/run_vllm_omni_bucket_20261118.sh > /workspace/r1117/logs-ab/bucket.log 2>&1 < /dev/null
set -u

STATE="${DBF_STATE_DIR:?DBF_STATE_DIR}"
ROUNDS="${ROUNDS:-3}"
CASES="${CASES:-zimage_turbo_t2i_1024:0,1 qwen_image_21_t2i_1024:0}"
ONLY="${ONLY:-O0 O1 N1 N2 P C}"
PIN=24e54728490e4ded447f0a6d0eda538dedf75bb8
NEWROOT="${STATE}/fw-venvs"
OLDROOT="${STATE}/fw-venvs-0da1"
# label root commit
ARMS=(
  "O0 ${OLDROOT} 0da1d831e07fadccc3ab51ef27b7c21623b693c2"
  "O1 ${OLDROOT} 670aa7ec6043571e0a8a16448685322e41d93828"
  "N1 ${NEWROOT} 4b63581cf6527dc5c964558650e996c0a36e027a"
  "N2 ${NEWROOT} ${PIN}"
  "P ${NEWROOT} 702917be6b58afff29cb71cfd012e84e1cffddab"
  "C ${NEWROOT} eb8b991ff692c3c304472044ef18772c65bc7eee"
)
if [[ -n "${ARMS_SPEC:-}" ]]; then  # "<label> <venv root> <commit>" lines replace the list above, e.g. to bisect inside O0..O1
  mapfile -t ARMS <<<"${ARMS_SPEC}"
else
  kept=()
  for arm in "${ARMS[@]}"; do [[ " ${ONLY} " == *" ${arm%% *} "* ]] && kept+=("${arm}"); done
  ARMS=("${kept[@]}")
fi

restore() { git -C "${NEWROOT}/vllm-omni/src/vllm-omni" checkout -q "${PIN}" && echo "restored ${NEWROOT} to ${PIN:0:9}"; }
trap restore EXIT

for root in "${NEWROOT}" "${OLDROOT}"; do
  src="${root}/vllm-omni/src/vllm-omni"
  [[ -d "${src}/.git" ]] || { echo "no source checkout at ${src}"; exit 1; }
  for sha in $(printf '%s\n' "${ARMS[@]}" | awk -v r="${root}" '$2==r {print $3}'); do
    git -C "${src}" cat-file -e "${sha}^{commit}" 2>/dev/null || git -C "${src}" fetch -q origin "${sha}" || { echo "cannot get ${sha:0:9} in ${src}"; exit 1; }
  done
done

plan="$(mktemp)"
for round in $(seq 1 "${ROUNDS}"); do
  if (( round % 2 )); then order=("${ARMS[@]}"); else order=(); for ((i = ${#ARMS[@]} - 1; i >= 0; i--)); do order+=("${ARMS[i]}"); done; fi
  for spec in ${CASES}; do
    case_id="${spec%%:*}"; gpus="${spec##*:}"
    for arm in "${order[@]}"; do
      read -r label root sha <<<"${arm}"
      git -C "${root}/vllm-omni/src/vllm-omni" checkout -q "${sha}" || { echo "checkout ${sha:0:9} failed"; continue; }
      echo "=== BUCKET ${label} ${sha:0:9} root=$(basename "${root}") case=${case_id} round=${round} head=$(git -C "${root}/vllm-omni/src/vllm-omni" rev-parse --short=9 HEAD)"
      echo "bk-${label}-${case_id%%_*}-r${round} vllm-omni ${case_id} single_e2e ${gpus} -" > "${plan}"
      DBF_VENV_ROOT="${root}" PLAN="${plan}" PLAN_TAG="bk${round}" bash "${STATE}/dbf/scripts/run_plan_ab.sh" 2>&1 | grep -E '^(ARM |=== ARM_START)'
    done
  done
done
rm -f "${plan}"
echo "BUCKET_DONE rounds=${ROUNDS}"
