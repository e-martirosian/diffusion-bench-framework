#!/usr/bin/env bash
# Second pass of the 2026-11-18 experiments: the fairness arms that showed a signal (+ Ref2VA async), then a bisect of vLLM-Omni's
# Z-Image gain between 0da1d831e (O0) and 670aa7ec6 (O1) on the old stack, around the two candidate commits.
#   DBF_STATE_DIR=/workspace/r1117 setsid -f bash scripts/chain2_20261118.sh > /workspace/r1117/logs-ab/chain2.log 2>&1 < /dev/null
set -u
STATE="${DBF_STATE_DIR:?DBF_STATE_DIR}"
export DBF_STATE_DIR="${STATE}"
EXTRA="${STATE}/extra"
LOGS="${STATE}/logs-ab"
echo "=== CHAIN2 START $(date -u +%FT%TZ)"
PLAN="${EXTRA}/plan_b300_fair_r2_20261118.txt" PLAN_TAG=fair2 bash "${STATE}/dbf/scripts/run_plan_ab.sh" > "${LOGS}/fair2.log" 2>&1
echo "=== FAIR2 DONE $(date -u +%FT%TZ)"
OLD="${STATE}/fw-venvs-0da1"
ARMS_SPEC="B0 ${OLD} 0da1d831e07fadccc3ab51ef27b7c21623b693c2
B1 ${OLD} 949907817ee9c15a7fc2733b30b24f82dd9bbaf9
B2 ${OLD} d12349a46990be1e81f7d406ae53db08a7823b9e
B3 ${OLD} 637695435c98bc11a76930cbe81b2671acd51bf2
B4 ${OLD} 7266b8bd2f0058645aaa44d53de108e7cc66c999
B5 ${OLD} 670aa7ec6043571e0a8a16448685322e41d93828" ROUNDS=2 CASES=zimage_turbo_t2i_1024:0,1 \
  bash "${EXTRA}/run_vllm_omni_bucket_20261118.sh" > "${LOGS}/bucket_bisect1.log" 2>&1
echo "=== BISECT1 DONE $(date -u +%FT%TZ)"
echo "=== CHAIN2 DONE $(date -u +%FT%TZ)"
