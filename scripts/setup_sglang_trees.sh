#!/usr/bin/env bash
# sglang source trees at several commits on one box, for same-box A/B of sglang against itself (scripts/run_plan_ab.sh
# takes a tree per arm through DBF_SGLANG_PYTHONPATH). One blobless clone, one worktree per commit.
#
#   DBF_STATE_DIR=/workspace/r1118 TREES="old=ca819c11cd new=25c1c43942 main=origin/main" bash scripts/setup_sglang_trees.sh
#
# Each tree lands in $DBF_STATE_DIR/sgl-<name>; an existing one is left alone. Prints `T <name> <sha> <subject>` per tree and
# whether `import sglang` resolves inside it with the box's python (an old tree against a newer image can fail to import).
set -u

STATE="${DBF_STATE_DIR:?DBF_STATE_DIR}"
FULL="${STATE}/sgl-full"
if [[ ! -d "${FULL}/.git" ]]; then
  git -c http.https://github.com/.extraheader= clone -q --filter=blob:none --no-checkout https://github.com/sgl-project/sglang.git "${FULL}" \
    || { echo "T FATAL clone"; exit 1; }
fi
git -C "${FULL}" -c http.https://github.com/.extraheader= fetch -q origin main || echo "T WARN fetch failed, using what the clone has"

for spec in ${TREES:?TREES="name=rev ..."}; do
  name="${spec%%=*}"; rev="${spec#*=}"
  dir="${STATE}/sgl-${name}"
  if [[ ! -d "${dir}" ]]; then
    git -C "${FULL}" -c http.https://github.com/.extraheader= worktree add -q --detach "${dir}" "${rev}" || { echo "T FAIL ${name} ${rev}"; continue; }
  fi
  resolved="$(cd /tmp && PYTHONPATH="${dir}/python" python3 -c 'import os, sglang; print(os.path.realpath(sglang.__file__))' 2>&1 | tail -1)"
  [[ "${resolved}" == "${dir}/python/"* ]] && ok=import-ok || ok="IMPORT-FAIL(${resolved:0:80})"
  echo "T ${name} $(git -C "${dir}" log -1 --format='%h %cs %s' | cut -c1-90) ${ok}"
done
echo "T_DONE"
