#!/usr/bin/env bash
# Install what the 2026-11-18 attribution and fairness arms need beside the round's venvs, never inside them (a venv is not relocatable):
#   fa4      vllm-omni[fa4] (CuTe FA4) into the round's vllm-omni venv; pip freeze before/after is kept so a side effect is visible
#   old      vLLM-Omni 0da1d831e + vllm 0.32.0 (the 2026-10-22 stack) in its own venv root fw-venvs-0da1, as scripts/b300_supp_20261102 did
#   cudnn    two scratch venvs, torch 2.13.0 and torch 2.14.0, for scripts/bench_cudnn_sdpa.py (the cuDNN wheel is swapped inside them)
#
#   DBF_STATE_DIR=/workspace/r1117 STEPS="fa4 old cudnn" setsid -f bash scripts/setup_attribution_20261118.sh > /workspace/r1117/attrib_setup.log 2>&1 < /dev/null
set -u

STATE="${DBF_STATE_DIR:?DBF_STATE_DIR}"
STEPS="${STEPS:-fa4 old cudnn}"
LOGS="${STATE}/logs"
mkdir -p "${LOGS}"
export HF_HOME="${STATE}/hf" HF_HUB_CACHE="${STATE}/hf/hub" HUGGINGFACE_HUB_CACHE="${STATE}/hf/hub"

for step in ${STEPS}; do
  t0=$(date +%s)
  case "${step}" in
    fa4)
      V="${STATE}/fw-venvs/vllm-omni"
      "${V}/bin/pip" freeze > "${LOGS}/freeze_vllm-omni_before_fa4.txt" 2>/dev/null
      # the extra's own pin: vllm-omni's pyproject [project.optional-dependencies] fa4
      spec="$(grep -A1 '^fa4 = \[' "${V}/src/vllm-omni/pyproject.toml" | sed -n 2p | sed -E 's/^[[:space:]]*"(.*)",?$/\1/')"
      echo "A fa4 spec=${spec}"
      DBF_VENV_ROOT="${STATE}/fw-venvs" bash "${STATE}/dbf/scripts/upgrade_framework_stack.sh" vllm-omni "${spec}" > "${LOGS}/install_fa4.log" 2>&1
      rc=$?
      "${V}/bin/pip" freeze > "${LOGS}/freeze_vllm-omni_after_fa4.txt" 2>/dev/null
      echo "A fa4 rc=${rc} $(( $(date +%s) - t0 ))s changed: $(diff "${LOGS}/freeze_vllm-omni_before_fa4.txt" "${LOGS}/freeze_vllm-omni_after_fa4.txt" | grep -c '^[<>]') lines"
      diff "${LOGS}/freeze_vllm-omni_before_fa4.txt" "${LOGS}/freeze_vllm-omni_after_fa4.txt" | grep '^[<>]' | head -12 | sed 's/^/A   /'
      "${V}/bin/python3" -c "import flash_attn.cute, importlib.metadata as m; print('A fa4 import ok', m.version('flash-attn-4'))" 2>&1 | tail -1 | sed 's/^/A /'
      ;;
    old)
      export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${STATE}/fw-venvs-0da1"
      export VLLM_INSTALL_SPEC=vllm==0.32.0
      export VLLM_OMNI_INSTALL_SPEC=git+https://github.com/vllm-project/vllm-omni.git@0da1d831e07fadccc3ab51ef27b7c21623b693c2
      export VLLM_OMNI_VSA_KERNEL_SPEC=fastvideo-kernel==0.3.5
      mkdir -p "${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT}"
      (cd "${STATE}/dbf" && bash scripts/install_comparison_frameworks.sh vllm-omni > "${LOGS}/install_vllm-omni_0da1.log" 2>&1)
      rc=$?
      echo "A old rc=${rc} $(( $(date +%s) - t0 ))s"
      "${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT}/vllm-omni/bin/python" - <<'PY' 2>&1 | sed 's/^/A   /'
import importlib.metadata as m
for p in ["vllm", "vllm-omni", "torch", "nvidia-cudnn-cu13", "nvidia-nccl-cu13", "triton", "flashinfer-python"]:
    try:
        print(p, m.version(p))
    except Exception:
        print(p, "n/a")
PY
      unset SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT VLLM_INSTALL_SPEC VLLM_OMNI_INSTALL_SPEC VLLM_OMNI_VSA_KERNEL_SPEC
      ;;
    cudnn)
      for pair in 2.13.0 2.14.0; do
        v="${STATE}/venv-torch${pair//./}"
        uv venv -q --python 3.12 "${v}" && uv pip install -q --python "${v}/bin/python" "torch==${pair}" > "${LOGS}/install_torch${pair}.log" 2>&1
        rc=$?
        echo "A cudnn torch${pair} rc=${rc} $("${v}/bin/python" -c 'import torch, importlib.metadata as m; print(torch.__version__, "cudnn", m.version("nvidia-cudnn-cu13"))' 2>&1 | tail -1)"
      done
      ;;
  esac
done
echo "A_DONE steps=[${STEPS}]"
