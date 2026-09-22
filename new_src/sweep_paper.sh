#!/usr/bin/env bash
# Full recompute on the development suite (35 pairs) after the paper-contract rebuild.
# Resumable: every step passes --resume, so re-running skips episodes already recorded.
#
#   AUTHMEM_BASE_URL=http://100.100.21.59:11434/v1 MODEL=qwen2.5:14b bash new_src/sweep_paper.sh
set -u
cd "$(dirname "$0")/.."
MODEL="${MODEL:-qwen2.5:14b}"
RUN=(env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u http_proxy -u https_proxy
     HF_HUB_OFFLINE=1 AUTHMEM_BASE_URL="${AUTHMEM_BASE_URL:-http://100.100.21.59:11434/v1}"
     AUTHMEM_ACTION_MODEL="$MODEL" AUTHMEM_CONSOLIDATOR_MODEL="$MODEL" AUTHMEM_JUDGE_MODEL="${JUDGE:-$MODEL}"
     .venv/bin/python -u -m new_src.run)

step() { echo "=== $(date -u +%H:%M:%S) $*"; "${RUN[@]}" "$@" --resume --quiet 2>&1 | grep -E "overall|fired anyway|Upgrade|Traceback|Error"; }

step null
# Module B: the paper's seven interventions (appendix E.1), then this project's gate arms.
for c in memory-off baseline baseline-attributed sanitizer conservative-join gold-washed gold-prompted \
         gate gate-license-model; do step b --condition "$c"; done
# Module A: write-time collapse.
step a
# Module C: the paper's five conditions (appendix F.2), then the gate arms.
for c in memory-off c-no-label c-naive-join c-predicted c-oracle gate gate-predicted gate-license-model; do
    step c --condition "$c"
done
echo "=== $(date -u +%H:%M:%S) sweep finished"
