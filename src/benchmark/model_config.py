"""
Where the benchmark's models come from — domain-agnostic, like engine.py/metrics.py.

Everything here defaults to a local Ollama instance (unchanged behavior from before this
module existed), but every value is overridable via environment variable so the exact same
code can point at a remote Ollama (e.g. a home GPU box reached over Tailscale) without
editing any runner file — just export these before running:

    export OLLAMA_BASE_URL="http://100.x.y.z:11434/v1"   # or http://<tailscale-hostname>:11434/v1
    export OLLAMA_MODEL="qwen2.5:14b"                     # any model already `ollama pull`-ed there
    export OLLAMA_HELPER_MODEL="qwen2.5:7b"

OLLAMA_MODEL/OLLAMA_HELPER_MODEL just need to name a model the *target* Ollama instance
already has — nothing here restricts which one; "any model" works as long as it's pulled on
whichever host OLLAMA_BASE_URL points at.
"""

import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# MODEL: the main agent model, drives the actual turn loop (tool-calling reliability matters
# most here). HELPER_MODEL: used for paraphrasing, value extraction, and label classification
# (resolver.py, safe_run.py/baseline_run.py's _rewrite_fact/_rewrite_episode) — can be smaller/
# faster since it's not doing the primary reasoning.
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
HELPER_MODEL = os.getenv("OLLAMA_HELPER_MODEL", "qwen2.5:7b")
