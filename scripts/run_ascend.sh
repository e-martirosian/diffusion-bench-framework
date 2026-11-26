#!/usr/bin/env bash
# rev: 2026-11-26-case-groups2 (verify on the box: `head -2 scripts/run_ascend.sh`
# shows this line; `grep -c 'CASE_GROUPS="wan22' scripts/run_ascend.sh` prints 1
# and `grep -c 'CASE_GROUPS2="qwen' scripts/run_ascend.sh` prints 1;
# dry-run: `DBF_DRY_RUN_GROUPS=1 bash scripts/run_ascend.sh` must print
# "38 cases in 32 groups" and exit without launching anything)
# Ascend A2/A3 (910B/910C) NPU cross-framework run (scaffold). Same harness and
# report shape as the NVIDIA rounds, but on Huawei Ascend NPUs. One script serves
# both NPUs — pick the chip with DBF_HARDWARE_PROFILE.
#
# The runner auto-detects Ascend via `npu-smi info`, records device=npu, and
# appends `--device npu` to the vLLM-Omni serve command only (sglang detects
# npu/gpu itself; its parser rejects --device), plus ASCEND_RT_VISIBLE_DEVICES
# instead of CUDA_VISIBLE_DEVICES. No code
# change is needed for a first run; set the values below to the box.
#
# Prerequisites (mirror scripts/run_b200_cross_framework_20260912.sh):
#   - sglang importable from its own checkout, vLLM-Omni in its isolated venv,
#     both built with Ascend NPU support
#   - an HF token in $DBF_STATE_DIR/.hftoken for the gated model licences
#
#   DBF_REPO_DIR=/path/to/diffusion-bench-framework \
#   DBF_HARDWARE_PROFILE=a2 bash scripts/run_ascend.sh   # Ascend 910B
#   DBF_REPO_DIR=/path/to/diffusion-bench-framework \
#   DBF_HARDWARE_PROFILE=a3 bash scripts/run_ascend.sh   # Ascend 910C
#
# Results land in $DBF_LOG_DIR; merge an NPU run with the matching GPU run and
# publish exactly as the NVIDIA rounds do (the report splits each framework into
# GPU/NPU columns when a merge carries both devices).
set -u

export DBF_STATE_DIR="${DBF_STATE_DIR:-/workspace/npu}"
export DBF_REPO_DIR="${DBF_REPO_DIR:?set DBF_REPO_DIR to this repo\'s checkout on the NPU box}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-${DBF_STATE_DIR}/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${DBF_HF_HOME}}"
# No HF access? Point DBF_MODEL_PATHS at a JSON file mapping repo id -> local
# model dir (see configs/model_paths.example.json); the runner uses it verbatim
# and skips HuggingFace for models. Reference images (image-edit / i2v cases)
# still fetch huggingface.co URLs unless DBF_HF_ENDPOINT (or HF_ENDPOINT)
# points at a mirror.
export DBF_MODEL_PATHS="${DBF_MODEL_PATHS:-${DBF_STATE_DIR}/model_paths.json}"
export DBF_FRAMEWORKS="${DBF_FRAMEWORKS:-sglang vllm-omni}"
export DBF_HARDWARE_PROFILE="${DBF_HARDWARE_PROFILE:-a2}"
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-5400}"

# Ascend device ids, like CUDA_VISIBLE_DEVICES ids; the runner maps this to
# ASCEND_RT_VISIBLE_DEVICES on NPU.
GPUS="${GPUS:-0}"
# Tag defaults to the chip (ascenda2 / ascenda3) so one script serves both.
TAG="${TAG:-ascend${DBF_HARDWARE_PROFILE}}"

# Every case whose model has a model_paths.json entry AND whose sglang cell is
# `supported` in its case file (54 of 64), as CONSECUTIVE SAME-MODEL GROUPS:
# one line = one harness process (DBF_ONE_PROCESS=1), so a shared model
# (qwen21 trio, cosmos3 trio, minimax quartet) downloads once and is removed
# after the group's last case. Excluded -- sglang `unsupported` there, so no
# sglang round can ever run them: qwen_image_2512_1024 ("MUSA-only
# checkpoint"), qwen_image_edit_1024, flux2_dev_nvfp4_t2i_1024,
# wan22_t2v_a14b_nvfp4, comfy_ideogram4_t2i, kandinsky6_vsr, hunyuan3d2,
# pi05_base, flux3_action_droid. Also excluded: flux3_klein_base_t2i_1024 --
# its model (black-forest-labs/flux-3-image-klein-base) has no
# model_paths.json entry; only flux-3-image-klein is mapped, and runs as
# flux3_image_klein_t2i_1024. Multi-GPU cases (num_gpus 2/4) need GPUS to
# cover them; on a smaller GPUS they fail fast with a recorded device-count
# error, not silently. CASE_TIMEOUT covers a GROUP, not each case.
# Precomputed by scripts/group_cases.py -- the runtime grouping ran through
# the box's python3 in both heredoc and file forms without success, so the
# verified output is baked in. Regenerate after changing the case set:
#   python3 scripts/group_cases.py <case ids in run order>

#ideogram4_t2i_1024_2gpu_tp unsupported
#qwen_image_21_t2i_1024 qwen_image_21_t2i_1024_2gpu qwen_image_21_edit_1024
#zimage_turbo_t2i_1024
#cosmos3_nano_t2i_720p cosmos3_nano_t2v_720p_189f cosmos3_nano_i2v_720p_189f
#ltx2.5_twostage_t2v_2gpus
#flux1_dev_t2i_1024
#flux2_klein_4b_t2i_1024
#flux3_image_klein_t2i_1024 - can't repair, because we can't run it
#qwen_image_t2i_1024
#qwen_image_edit_2511_1024
#qwen_image_layered_1024
#glm_image_t2i_1024
#zimage_t2i_1024
#ernie_image_t2i_1024
#joyai_image_edit_1024
#hidream_o1_image_t2i_1024
#anima_base_t2i_1024
#ltx2_t2v_720p
#wan21_t2v_13b_720p
#wan21_i2v_14b_480p
#wan22_ti2v_5b_720p
#wan22_ti2v_5b_fullattn

CASE_GROUPS="wan22_t2v_a14b_720p
minimax_h3_t2va_5s minimax_h3_t2va_5s_4gpu minimax_h3_t2va_480p_1gpu minimax_h3_ref2va_5s minimax_h3_fasth3_v2_t2va_5s
flux2_dev_t2i_1024
qwen_image_edit_2509_1024
hunyuan_image_30_t2i_1024
hunyuan_image_30_instruct_t2i_1024
joyai_echo_1024
llada_image_t2i_1024
ltx23_t2v_720p
vdn_minimax_h3_t2v
helios_base_t2v
helios_distilled_t2v
helios_mid_t2v
pika_h3_flex_t2v
fasth3_omni_t2v
wan21_t2v_14b_720p
wan22_i2v_a14b_720p
turbowan21_t2v_13b
hunyuanvideo_t2v_720p
fasthunyuan_t2v
longlive2_t2v
mova_360p_t2v
lingbot_video_moe_t2v
lingbot_world_t2v"

# CASE_GROUPS2 -- the sglang diffusion support table (2026-11-26), mapped onto
# case ids that exist here and can actually run on the box. CASE_GROUPS is
# left untouched; these groups run after it, sharing the g<N> tags and the
# port stride. Skipped as DUPLICATES of what CASE_GROUPS already runs:
#   LLaDA-Image (llada_image_t2i_1024), LingBot-World V2 (lingbot_world_t2v),
#   LongLive-2 (longlive2_t2v), MOVA (mova_360p_t2v), TurboWan2.1
#   (turbowan21_t2v_13b), TurboWan2.2-I2V-A14B (wan22_i2v_a14b_720p).
# Skipped as sglang-unsupported: ideogram4_t2i_1024_2gpu_tp ("unsupported"
# above), comfy_ideogram4_t2i, kandinsky6_vsr (VSR, not generative),
# pi05_base (robotics policy).
# Skipped -- no case file yet for the table's checkpoint: Cosmos3
# Super/Edge/Nano-Policy-DROID/Super-Image2Video/Super-Text2Image/Nano-Sim-*,
# Kandinsky 6 Pro variants, Nomas (Hy-Nomas), SANA-Video/2.0, SANA-WM,
# SANA 600M/1600M/1.5, Wan 2.2 Animate x2, SenseNova-U1.5, FireRed x2,
# FLUX.2 Klein 9B x2, Krea 2, LongCat x3, Ming x2, MOSS x2, SD3 x6,
# Wan2.1-Fun-InP. Qwen-Image-NVFP4 also has no case (the NVFP4 numeric class
# is sglang-unsupported here), so the base-model qwen_image_t2i_1024 is NOT
# standing in for that row.
# Skipped -- FLUX.3 Image: flux3_klein_base_t2i_1024 is an exact checkpoint
# match but its model has no model_paths.json entry and is gated (header
# note above); flux3_image_klein_t2i_1024 is marked unrunnable above.
# Checkpoint deltas vs the table rows, all same numeric class: the qwen21
# trio serves Qwen/Qwen-Image-2.1 (row is the -Turbo distill),
# wan21_t2v_13b_720p the Wan-AI 1.3B base (row is the FastVideo Fast
# distill), wan21_i2v_14b_480p the 480P checkpoint (row is 720P),
# wan22_ti2v_5b_720p the Wan-AI TI2V-5B base (row is the Fast distill).
# Same-model cases share one line (one harness process, one download);
# CASE_TIMEOUT covers a GROUP. The 2gpu cases need GPUS to cover 2 cards.
CASE_GROUPS2="qwen_image_21_t2i_1024 qwen_image_21_t2i_1024_2gpu qwen_image_21_edit_1024
flux2_klein_4b_t2i_1024
ernie_image_t2i_1024
hidream_o1_image_t2i_1024
wan21_t2v_13b_720p
wan21_i2v_14b_480p
wan22_ti2v_5b_720p
wan22_ti2v_5b_fullattn"

# Consecutive cases sharing one model (the qwen21 trio, the cosmos3 trio, the
# minimax quartet) run as ONE harness process each: the model downloads once
# and is removed after the group's last case. CASE_TIMEOUT applies per GROUP.
export DBF_ONE_PROCESS=1

echo "=== Ascend NPU cross-framework run START $(date -Is) ==="
echo "=== repo=$DBF_REPO_DIR frameworks=[$DBF_FRAMEWORKS] hw=$DBF_HARDWARE_PROFILE npu=$GPUS ==="
# Groups are computed by python and validated: every token must be a known
# case id and the groups must cover CASES exactly. Without the validation a
# degraded grouping (python3 step failed, CASES never reached the script) once
# ran a nonexistent "case 0" and completed rc=0 in two seconds.
# CASE_GROUPS runs first, then CASE_GROUPS2 appends its groups; both share
# the g<N> tags and the port stride. Blank lines are skipped by the loop.
run_groups="${CASE_GROUPS}
${CASE_GROUPS2}"
if [ -z "${CASE_GROUPS}${CASE_GROUPS2}" ]; then
    echo "ERROR: CASE_GROUPS and CASE_GROUPS2 are both empty -- nothing to run" >&2
    exit 1
fi
echo "=== $(echo $run_groups | wc -w) cases in $(printf '%s\n' "$run_groups" | grep -c .) groups ==="
# Verify a freshly deployed copy without launching anything:
#   DBF_DRY_RUN_GROUPS=1 bash scripts/run_ascend.sh
if [ "${DBF_DRY_RUN_GROUPS:-0}" = "1" ]; then
    echo "first CASE_GROUPS groups:"
    printf '%s\n' "$CASE_GROUPS" | head -3
    echo "first CASE_GROUPS2 groups:"
    printf '%s\n' "$CASE_GROUPS2" | head -3
    exit 0
fi
# Port stride between groups. A sglang server binds its HTTP port plus
# derived ports next to it (the ZMQ broker sits at port+1, others nearby), so
# a leaked server covers a RANGE; the stride must exceed that range or the
# next group's HTTP port lands inside it ("could not bind port 45013",
# 2026-11-24: a leaked 4-card wan22 server from an earlier run held the range).
PORT_SPACING="${DBF_PORT_SPACING:-32}"
port="${BASE_PORT:-45001}"
gi=0
while IFS= read -r group; do
    [ -z "$group" ] && continue
    gi=$((gi + 1))
    gtag="${TAG}_g${gi}"
    echo "=== $(date -Is) group $gi: $group ==="
    bash "${DBF_REPO_DIR}/scripts/devbox_run_cases.sh" \
        "$gtag" "$GPUS" "$port" "single_e2e throughput" $group
    grep -h "^RESULT" "${DBF_LOG_DIR}/run_${gtag}.log" 2>/dev/null
    port=$((port + PORT_SPACING))
done <<< "$run_groups"
echo "=== Ascend NPU cross-framework run DONE $(date -Is) ==="
