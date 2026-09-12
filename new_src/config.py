"""
Every environment-dependent value the AuthMem-Bench re-implementation needs, in one place.

Defaults point at a local Ollama + local Postgres, exactly like src/, but each one is
overridable so the same code can drive a remote GPU box without editing a runner:

    export AUTHMEM_BASE_URL="http://gpu-box:11434/v1"   # the default; a tailnet host
    export AUTHMEM_ACTION_MODEL="qwen2.5:14b"      # the action agent (Module B/C)
    export AUTHMEM_CONSOLIDATOR_MODEL="qwen2.5:14b"  # the memory consolidator (Module A)
    export AUTHMEM_JUDGE_MODEL="qwen2.5:14b"       # Module A's blinded outcome judge
    export AUTHMEM_DB_URL="postgresql://user:pass@host/db"

The paper (arXiv:2608.01679) uses three distinct model roles and they are kept distinct
here even when they default to the same local checkpoint: the consolidator writes memory,
the action model consumes it, and the judge classifies write-time outcomes. Collapsing them
would make Module A's judgement self-graded.
"""

import os

try:  # optional: a .env in the project root is loaded if python-dotenv is installed
    import logging

    from dotenv import find_dotenv, load_dotenv

    # The project's .env holds unrelated free-form lines; dotenv warns about them on every
    # import. Nothing here depends on those lines, so the warning is noise.
    logging.getLogger("dotenv.main").setLevel(logging.ERROR)
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:  # noqa: BLE001 - a missing or unparsable .env must never break a run
    pass

# The default target is the gpu-box over Tailscale, not this machine: the benchmark is many
# thousands of tokens per episode and the local box does not have the headroom (background
# runs here were killed for memory). MagicDNS resolves `gpu-box`; if it ever does not, use the
# tailnet address http://100.100.21.59:11434/v1, or point AUTHMEM_BASE_URL wherever you like —
# nothing else in the package knows the endpoint.
DEFAULT_BASE_URL = "http://gpu-box:11434/v1"

BASE_URL = os.getenv("AUTHMEM_BASE_URL", os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL))
API_KEY = os.getenv("AUTHMEM_API_KEY", "ollama")

ACTION_MODEL = os.getenv("AUTHMEM_ACTION_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:14b"))
CONSOLIDATOR_MODEL = os.getenv("AUTHMEM_CONSOLIDATOR_MODEL", ACTION_MODEL)
JUDGE_MODEL = os.getenv("AUTHMEM_JUDGE_MODEL", ACTION_MODEL)

# Own tables in the same Postgres instance (see memory/models.py) — src/ keeps its own
# semanticMemory/episodicMemory and the two suites can no longer truncate each other.
DB_URL = os.getenv("AUTHMEM_DB_URL", "postgresql://lenaz:lenaz210607@localhost/mydb")

EMBEDDING_MODEL = os.getenv("AUTHMEM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

LOGS_ROOT = os.getenv("AUTHMEM_LOGS_ROOT", "logs_authmem")
