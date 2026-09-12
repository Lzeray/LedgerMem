#!/usr/bin/env bash
#
# Run Module B and Module C against the Innopolis inference endpoints.
#
#   ./run_api_models.sh            # both endpoints, full suite
#   ./run_api_models.sh a          # only endpoint A
#   SMOKE=3 ./run_api_models.sh    # 3 pairs per condition first, to see it work
#
# Every run passes --resume, so this script is safe to interrupt and re-run: it picks up
# exactly the episodes that are missing and never re-runs a recorded one.

set -uo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Configure here. If preflight says a model id is not served, copy the correct
# id from the list it prints and put it here.
# ---------------------------------------------------------------------------
ENDPOINT_A="http://10.100.11.201:8000/v1"
MODEL_A="Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"

ENDPOINT_B="http://10.100.10.70:8123/v1"
MODEL_B="qwen3.6-35B-A3B"          # <- unverified; preflight will print the real id

# The two conditions being compared: the unprotected baseline and the newest gate.
CONDITIONS=("baseline" "gate-license-model")

PY=".venv/bin/python"
LIMIT=""
[ -n "${SMOKE:-}" ] && LIMIT="--limit ${SMOKE}"

# The personal proxy at 127.0.0.1:10808 must not swallow traffic bound for the
# corporate network — no_proxy only covers localhost.
# HF_HUB_OFFLINE stops sentence-transformers retrying huggingface.co five times on every
# process start. The embedding model is cached locally and needs no network; without this each
# run wastes ~15s per process on name-resolution failures that then succeed from cache anyway.
NOPROXY=(env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u http_proxy -u https_proxy
         HF_HUB_OFFLINE=1)

run_endpoint() {
    local url="$1" model="$2" tag="$3"

    # All three model roles must be set. --model only overrides the ACTION model;
    # Module C's consolidator and judge are read from the environment, and if they are
    # left pointing at a local Ollama tag the remote server answers 404 on every call.
    export AUTHMEM_BASE_URL="$url"
    export AUTHMEM_ACTION_MODEL="$model"
    export AUTHMEM_CONSOLIDATOR_MODEL="$model"
    export AUTHMEM_JUDGE_MODEL="$model"
    # This shell has OLLAMA_BASE_URL and OLLAMA_MODEL set, and new_src/config.py falls back to
    # them when the AUTHMEM_ vars are absent. That is why all three are exported here: running
    # `python -m new_src.run` by hand without them silently talks to the tailnet gpu-box with
    # qwen3:8b instead of the endpoint and model intended.
    # AUTHMEM_API_KEY comes from .env, which is gitignored. Never put it on a command line.

    echo
    echo "############################################################################"
    echo "#  $tag  —  $model"
    echo "#  $url"
    echo "############################################################################"

    if ! "${NOPROXY[@]}" $PY -m new_src.preflight; then
        echo
        echo ">>> preflight failed for $tag — skipping its runs. Fix the endpoint or the"
        echo ">>> model id above and re-run; nothing was recorded, so nothing is lost."
        return 1
    fi

    for condition in "${CONDITIONS[@]}"; do
        for module in b c; do
            echo
            echo "--- module ${module^^} — condition ${condition} ---"
            "${NOPROXY[@]}" $PY -m new_src.run "$module" \
                --condition "$condition" --model "$model" --resume --quiet $LIMIT
        done
    done
}

which="${1:-both}"
[ "$which" = "a" ] || [ "$which" = "both" ] && run_endpoint "$ENDPOINT_A" "$MODEL_A" "ENDPOINT A"
[ "$which" = "b" ] || [ "$which" = "both" ] && run_endpoint "$ENDPOINT_B" "$MODEL_B" "ENDPOINT B"

echo
echo "############################################################################"
echo "#  done. Summaries:  $PY -m new_src.report"
echo "############################################################################"
