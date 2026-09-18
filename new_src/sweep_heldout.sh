#!/usr/bin/env bash
# Full run on the held-out suites: the speech-act families (Q2D, N2D, P2F, G2O) and the 35-pair
# held-out core suite, Module B and Module C. Resumable (--resume).
#
#   AUTHMEM_BASE_URL=http://100.100.21.59:11434/v1 MODEL=qwen2.5:14b bash new_src/sweep_heldout.sh
set -u
cd "$(dirname "$0")/.."
MODEL="${MODEL:-qwen2.5:14b}"
RUN=(env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u http_proxy -u https_proxy
     HF_HUB_OFFLINE=1 AUTHMEM_BASE_URL="${AUTHMEM_BASE_URL:-http://100.100.21.59:11434/v1}"
     AUTHMEM_ACTION_MODEL="$MODEL" AUTHMEM_CONSOLIDATOR_MODEL="$MODEL" AUTHMEM_JUDGE_MODEL="${JUDGE:-$MODEL}"
     .venv/bin/python -u -m new_src.run_heldout)

step() { echo "=== $(date -u +%H:%M:%S) $*"; "${RUN[@]}" "$@" --resume --quiet 2>&1 | grep -E "overall|fired anyway|Traceback|Error|rror:|undefined|supports"; }

# The speech-act families first, so a wiring problem shows up early. The washed rendering is
# undefined for them (see run_heldout), so the paper's washed arms do not apply.
step b --suite speechact --null
for c in memory-off baseline-attributed gold-prompted gate gate-license-model; do
    step b --suite speechact --condition "$c"
done
# Held-out core: null control, Module B (the paper's seven interventions + gate arms), Module C
# (the paper's five conditions + gate arms).
step b --suite core --null
for c in memory-off baseline baseline-attributed sanitizer conservative-join gold-washed gold-prompted \
         gate gate-license-model; do step b --suite core --condition "$c"; done
for c in memory-off c-no-label c-naive-join c-predicted c-oracle gate gate-predicted gate-license-model; do
    step c --suite core --condition "$c"
done
echo "=== $(date -u +%H:%M:%S) held-out sweep finished"
