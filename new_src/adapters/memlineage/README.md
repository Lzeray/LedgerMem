# MemLineage as the memory system under test

An adapter that lets AuthMem-Bench (`new_src/`) run its action-time module against a live
[MemLineage](../../../memlineage) backend instead of the benchmark's own deterministic memory
stub. Nothing under `new_src/bench/` or `new_src/data/` is modified, and nothing in
`memlineage/` is modified either: the episode construction, the action stage, the strict action
predicate, the ASR/TSR metrics and the logging are the benchmark's; the write and read paths
are MemLineage's.

## The question it answers

The gate in `new_src/bench/gate.py` decides from an authority label held in code. MemLineage
has no authority label — it has notes with bodies, a `sources` list per note, and a governed
write path (`dry-run → human commit → audit → undo`). So an agent backed by MemLineage has
prose and a source list to reason from, and nothing intercepts the banking action at all.
Whether that is enough against the paper's seven authority transitions is what these runs
measure.

## What the adapter does, and what it deliberately does not do

MemLineage is used **exactly as documented** — `INTEGRATION.md` for the write governance,
`skills/memlineage/SKILL.md` for the agent-facing read and write actions. No field, label or
check that MemLineage does not ship was added to it.

Per episode:

1. `reset()` — delete every active note, so no episode sees the previous one's memory.
2. **Write**, through the governed path: one `POST /api/v1/changes/dry-run` holding one
   `append_note` per memory record, then `POST /api/v1/changes/{id}/commit` approved by a user
   actor. Each note carries its origin in MemLineage's own `sources` list, per governance rule
   7 ("include `source`/`source_ref` whenever the target action supports it"). The role that
   produced a claim is mapped onto MemLineage's own `SourceType` enum:

   | role in the episode | MemLineage source |
   |---|---|
   | `user` | `message_id` `chat://customer-message` |
   | `assistant` | `message_id` `chat://assistant-message` |
   | `tool` | `doc_id` `tool-result://external-service-response` |
   | `system` | `text` `bank system of record` |

3. **Read**, through a documented read action, and the rows that come back — nothing else —
   become the records the agent is shown.
4. The benchmark's own action stage runs with `policy=direct` and `show_metadata=False`:
   direct because MemLineage does not gate actions, and no metadata because there is no
   authority metadata in MemLineage to display. Displaying one would mean inventing it.

Two choices are deliberately in MemLineage's favour, so that a poor result cannot be blamed on
a stingy integration: the records it is given still name their source in prose
(`source_attributed`, not the paper's washed rendering), and the default read path is
`search_notes`, which is the only read surface that returns both the body and the stored
sources.

One harness artifact is corrected rather than measured: an episode's notes are all written
inside one change set, so their `updated_at` values tie at second resolution and MemLineage's
ordering of them is arbitrary. The adapter displays them in the order it wrote them. That is
client-side bookkeeping — a real workspace accumulates notes minutes apart — and it keeps the
memory block's order fixed, as it is in every condition the run is compared against.

## Conditions

| condition | read path | what the agent sees |
|---|---|---|
| `memlineage` | `search_notes` | note body **and** its stored `sources` — MemLineage's best surface |
| `memlineage-nosources` | `search_notes` | note bodies only, as a summarizing read would hand them over |
| `memlineage-context` | `get_context_bundle` | the skill's generic context read: titles and metadata, no bodies, no sources |

## Running

MemLineage's backend must be up. Its pinned `pydantic==2.9.2` has no wheel for Python 3.14, so
it gets its own virtualenv with unpinned versions — never install it into the benchmark's
`.venv`, which holds torch and sentence-transformers:

```bash
python3 -m venv memlineage/.venv-bench
memlineage/.venv-bench/bin/pip install fastapi uvicorn "sqlalchemy>=2.0" "pydantic>=2.9"

cd memlineage/backend
AFKMS_DB_BACKEND=sqlite AFKMS_REQUIRE_AUTH=false AFKMS_SQLITE_PATH=data/authmem_bench.sqlite3 \
  ../.venv-bench/bin/python -m uvicorn src.app:app --host 127.0.0.1 --port 8077
```

Then, from the project root:

```bash
RUN='env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1 \
     KMS_BASE_URL=http://127.0.0.1:8077 AUTHMEM_BASE_URL=http://gpu-box:11434/v1 \
     AUTHMEM_ACTION_MODEL=qwen2.5:14b .venv/bin/python'

$RUN -m new_src.adapters.memlineage.run --condition memlineage
$RUN -m new_src.adapters.memlineage.run --condition memlineage --suite licence
$RUN -m new_src.adapters.memlineage.run --condition memlineage-nosources
$RUN -m new_src.adapters.memlineage.run --condition memlineage-context
```

`--categories`, `--bases`, `--variants`, `--limit`, `--suite`, `--resume` and `--quiet` behave
as in `new_src.run`. Records land in the same tree as every other condition —
`logs_authmem/<model>/module_b/<condition>/episodes.jsonl` — so `python -m new_src.report`
and any comparison read them without special handling.

The comparison arms are the untouched benchmark, on the same pairs and the same model:

```bash
$RUN -m new_src.run b --condition baseline --model qwen2.5:14b   # washed memory, no metadata
$RUN -m new_src.run b --condition gate     --model qwen2.5:14b   # the authority gate
$RUN -m new_src.run b --condition gate-license --model qwen2.5:14b
```
