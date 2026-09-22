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

# More than one key for the same endpoint, comma-separated. Hosted free tiers are quoted per
# key and per minute, so two keys are two quotas: requests are dealt out over them in turn and
# a key that answers 429 hands the retry to the next one. Falls back to the single key above,
# which is what a local server wants — there the list is one entry long and nothing rotates.
API_KEYS = [key.strip() for key in os.getenv("AUTHMEM_API_KEYS", "").split(",") if key.strip()] \
    or [API_KEY]

# Hosted APIs fail in ways a local server does not: a per-minute rate limit, a transient
# "high demand" 503, a request that never returns. Left to itself the openai client retries
# twice and then raises, the episode is recorded as a failure, and five of those in a row trip
# the run's breaker — a rate limit stops a whole phase. These three turn that into waiting.
#
#   REQUEST_TIMEOUT        seconds before a single request is abandoned; without it the
#                          client waits ten minutes and the run looks hung.
#   MAX_RETRIES            handled inside the client (429 / 5xx / timeouts, honouring
#                          Retry-After). Raised well above the default of 2 for hosted keys.
#   MIN_REQUEST_INTERVAL   client-side throttle, seconds between requests. Free tiers are
#                          quoted per minute, so spacing requests out is strictly better than
#                          firing them and backing off after the refusal. 0 disables it, which
#                          is what a local server wants.
REQUEST_TIMEOUT = float(os.getenv("AUTHMEM_REQUEST_TIMEOUT", "180"))
MAX_RETRIES = int(os.getenv("AUTHMEM_MAX_RETRIES", "8"))
MIN_REQUEST_INTERVAL = float(os.getenv("AUTHMEM_MIN_REQUEST_INTERVAL", "0"))

# Duty cycle, for when the model runs on the same machine you are sitting at. A sweep is hours
# of continuous inference and a laptop has no room to shed that heat, so the run pauses of its
# own accord instead of relying on the operating system to throttle after the fact.
#
#   DUTY_WORK   seconds of work before a scheduled pause. 0 disables the whole mechanism,
#               which is what any remote endpoint wants — that hardware is not ours to nurse.
#   DUTY_REST   seconds of the pause itself.
#   TEMP_HIGH   pause early if the CPU reaches this, without waiting for the schedule.
#   TEMP_LOW    keep pausing until it has come back down to this, up to DUTY_REST_MAX.
#
# The thermal half is an addition to the schedule, never a replacement for it: a sensor that
# cannot be read, or one that never crosses the threshold, must still leave the machine its
# scheduled rest.
DUTY_WORK = float(os.getenv("AUTHMEM_DUTY_WORK", "0"))
DUTY_REST = float(os.getenv("AUTHMEM_DUTY_REST", "300"))
DUTY_REST_MAX = float(os.getenv("AUTHMEM_DUTY_REST_MAX", "900"))
TEMP_HIGH = float(os.getenv("AUTHMEM_TEMP_HIGH", "85"))
TEMP_LOW = float(os.getenv("AUTHMEM_TEMP_LOW", "72"))

ACTION_MODEL = os.getenv("AUTHMEM_ACTION_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:14b"))
CONSOLIDATOR_MODEL = os.getenv("AUTHMEM_CONSOLIDATOR_MODEL", ACTION_MODEL)
JUDGE_MODEL = os.getenv("AUTHMEM_JUDGE_MODEL", ACTION_MODEL)

# The judge's own endpoint (Module A's judge, Module C's reference labeler). Unset, the judge
# shares the agent's endpoint, as it always did. Set, it gets a separate key pool — typically a
# hosted API behind a proxy while the agent runs on a local Ollama that must not be proxied.
# The paper uses a different, stronger model for these roles than for the action agent; a judge
# that is the same checkpoint as the agent makes the Oracle arm a copy of the Predicted arm.
#
#   AUTHMEM_JUDGE_BASE_URL          OpenAI-compatible endpoint of the judge
#   AUTHMEM_JUDGE_API_KEYS          comma-separated keys for it (rotated)
#   AUTHMEM_JUDGE_PROXY             HTTP proxy the judge's requests go through, if any
#   AUTHMEM_JUDGE_MIN_REQUEST_INTERVAL  per-key spacing, for hosted free tiers
JUDGE_BASE_URL = os.getenv("AUTHMEM_JUDGE_BASE_URL", "")
JUDGE_API_KEYS = [key.strip() for key in os.getenv("AUTHMEM_JUDGE_API_KEYS", "").split(",") if key.strip()] \
    or ["none"]
JUDGE_PROXY = os.getenv("AUTHMEM_JUDGE_PROXY", "") or None
JUDGE_MIN_REQUEST_INTERVAL = float(os.getenv("AUTHMEM_JUDGE_MIN_REQUEST_INTERVAL", "0"))

# The consolidator's own endpoint, on the same pattern as the judge's. Unset, the consolidator
# shares the agent's endpoint, as it always did. The paper consolidates with hosted frontier
# models; a small local checkpoint running the paper's prompt verbatim drops user statements it
# should keep (qwen2.5:14b returned an empty memory list for 11 of 14 dev H+ histories that
# gemini-3.5-flash-lite kept in full), so the write stage is measuring the model, not the setup.
#
#   AUTHMEM_CONSOLIDATOR_BASE_URL          OpenAI-compatible endpoint of the consolidator
#   AUTHMEM_CONSOLIDATOR_API_KEYS          comma-separated keys for it (rotated)
#   AUTHMEM_CONSOLIDATOR_PROXY             HTTP proxy its requests go through, if any
#   AUTHMEM_CONSOLIDATOR_MIN_REQUEST_INTERVAL  per-key spacing, for hosted free tiers
CONSOLIDATOR_BASE_URL = os.getenv("AUTHMEM_CONSOLIDATOR_BASE_URL", "")
CONSOLIDATOR_API_KEYS = [key.strip() for key in os.getenv("AUTHMEM_CONSOLIDATOR_API_KEYS", "").split(",")
                         if key.strip()] or ["none"]
CONSOLIDATOR_PROXY = os.getenv("AUTHMEM_CONSOLIDATOR_PROXY", "") or None
CONSOLIDATOR_MIN_REQUEST_INTERVAL = float(os.getenv("AUTHMEM_CONSOLIDATOR_MIN_REQUEST_INTERVAL", "0"))

# Own tables in the same Postgres instance (see memory/models.py) — src/ keeps its own
# semanticMemory/episodicMemory and the two suites can no longer truncate each other.
DB_URL = os.getenv("AUTHMEM_DB_URL", "postgresql://lenaz:lenaz210607@localhost/mydb")

EMBEDDING_MODEL = os.getenv("AUTHMEM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

LOGS_ROOT = os.getenv("AUTHMEM_LOGS_ROOT", "logs_authmem")
