"""
MemLineage as the benchmark's memory system — adapter, not a rewrite.

The rule this file follows: **use MemLineage exactly as its own documentation describes it,
and add nothing.** MemLineage has no authority label, no trust level and no gate over
downstream actions; it has notes with bodies, a `sources` list per note, and a governed write
path (dry-run -> human commit). So that is all the adapter uses. Where the benchmark would
normally consult an authority label, this adapter has nothing to consult, and that absence is
the measurement — not a gap to paper over.

Write path (MemLineage's governed one, INTEGRATION.md "How Agent Writes Work" and
skills/memlineage/SKILL.md "Governance Rules"):

    POST /api/v1/changes/dry-run        one `append_note` action per memory record
    POST /api/v1/changes/{id}/commit    approved by a user actor

Each note carries its origin in the `sources` list, because rule 7 of the skill's governance
section says to include source traceability wherever the action supports it. The mapping from
the message role that produced a claim onto MemLineage's own `SourceType` enum
(`text | url | doc_id | message_id`) is the only interpretive step in the adapter, and it is
deliberately generous to MemLineage: a tool result is recorded as an external document, a
customer or assistant utterance as a chat message. Nothing invents a field MemLineage does
not have.

Read path (the documented read action `search_notes`):

    GET /api/v1/notes/search            returns body AND the stored sources per note

This is MemLineage's *best* read surface for this purpose. `get_context_bundle`, the skill's
generic context read, returns notes without their bodies and without their sources at all, so
an agent relying on it would have no memory content beyond note titles; the
`memlineage-context` condition exists to show that, but `notes` is the fair default.
"""

from __future__ import annotations

from dataclasses import dataclass

from new_src.adapters.memlineage.client import MemLineageClient
from new_src.bench.schema import MemoryRecord

#: How the role that introduced a claim is recorded in MemLineage's own source vocabulary.
#: MemLineage's SourceType enum is `text | url | doc_id | message_id`; these are its values,
#: used for what they mean, not extended.
SOURCE_BY_ROLE: dict[str, tuple[str, str]] = {
    "user": ("message_id", "chat://customer-message"),
    "assistant": ("message_id", "chat://assistant-message"),
    "tool": ("doc_id", "tool-result://external-service-response"),
    "system": ("text", "bank system of record"),
}

AGENT_ACTOR = {"type": "agent", "id": "authmem-bench"}
HUMAN_ACTOR = {"type": "user", "id": "benchmark-operator"}
WORKSPACE_TAG = "authmem-bench"


@dataclass
class WriteReceipt:
    """What MemLineage reported back about the governed write, kept for the run record."""

    change_set_id: str
    commit_id: str
    diff: list
    notes_written: int


class MemLineageWorkspace:
    """One MemLineage backend, used as the agent's memory."""

    def __init__(self, client: MemLineageClient | None = None):
        self.client = client or MemLineageClient()
        #: Bodies in the order they were written, used only to display them back in that
        #: order. See `read_notes` for why this is bookkeeping and not a thumb on the scale.
        self._write_order: list[str] = []

    # --- housekeeping ------------------------------------------------------

    def reset(self) -> int:
        """Delete every active note, so an episode never sees the previous episode's memory.

        Uses MemLineage's own note delete endpoint; there is no bulk truncate in its API.
        """
        removed = 0
        while True:
            rows = self.client.get("/api/v1/notes/search?page=1&page_size=100&status=active")["items"]
            if not rows:
                return removed
            for row in rows:
                self.client.delete(f"/api/v1/notes/{row['id']}")
                removed += 1

    # --- write -------------------------------------------------------------

    def write_records(self, records: list[MemoryRecord]) -> WriteReceipt:
        """Write the episode's memory through MemLineage's governed write path.

        One change set holding one `append_note` per record, then a commit by a human actor.
        The commit stands in for the human approval MemLineage requires: approving *that the
        note be stored* is the only decision its pipeline offers, and it says nothing about
        what the note may later authorize. That is precisely the property under test.
        """
        actions = [
            {
                "type": "append_note",
                "payload": {
                    "title": _title(record.text),
                    "body": record.text,
                    "sources": [_source_for(record.role)],
                    "tags": [WORKSPACE_TAG],
                },
            }
            for record in records
        ]
        self._write_order = [record.text for record in records]
        proposal = self.client.post(
            "/api/v1/changes/dry-run",
            {"actions": actions, "actor": AGENT_ACTOR, "tool": "memlineage"},
        )
        change_set_id = proposal["change_set_id"]
        commit = self.client.post(
            f"/api/v1/changes/{change_set_id}/commit",
            {"approved_by": HUMAN_ACTOR, "client_request_id": f"{change_set_id}-commit"},
        )
        return WriteReceipt(
            change_set_id=change_set_id,
            commit_id=commit["commit_id"],
            diff=proposal.get("diff", []),
            notes_written=len(actions),
        )

    # --- read --------------------------------------------------------------

    def read_notes(self) -> list[dict]:
        """`search_notes`, MemLineage's documented note read action.

        Rows come back in the order this adapter wrote them. MemLineage orders by
        `updated_at` descending at second resolution, and an episode's notes are all written
        inside one change set, so their timestamps tie and the surviving order is arbitrary.
        That collision is an artifact of the harness writing a whole history at once — a real
        workspace accumulates notes minutes apart — so leaving it in would add run-to-run
        noise that has nothing to do with MemLineage and would not be comparable with the
        benchmark's own conditions, which display memory in a fixed order. The client already
        knows what it wrote; nothing here reads any authority information.
        """
        rows = self.client.get("/api/v1/notes/search?page=1&page_size=100&status=active")["items"]
        return self._in_write_order(rows, key=lambda row: row.get("body", ""))

    def read_context_bundle(self) -> list[dict]:
        """`get_context_bundle`, the skill's generic context read. Returns note titles and
        metadata only — no body, no sources."""
        bundle = self.client.get("/api/v1/context/bundle?intent=banking&notes_limit=100")
        return self._in_write_order(bundle.get("notes", []), key=lambda row: row.get("title", ""))

    def _in_write_order(self, rows: list[dict], key) -> list[dict]:
        order = {text: index for index, text in enumerate(self._write_order)}

        def position(row):
            value = key(row)
            for text, index in order.items():
                if value and (value == text or text.startswith(value)):
                    return (0, index)
            return (1, row.get("created_at") or "", row.get("id", ""))

        return sorted(rows, key=position)


# --- rendering what MemLineage gives back ----------------------------------


def as_memory_records(rows: list[dict], with_sources: bool, read_path: str = "notes") -> list[MemoryRecord]:
    """Turn MemLineage's read result into the record list the action stage displays.

    The `label` field of a MemoryRecord is required by the benchmark's dataclass but has no
    counterpart in MemLineage, so it is set to the empty string and never displayed: these
    conditions run with `show_metadata=False`, because there is no authority metadata to show.
    """
    records = []
    for row in rows:
        if read_path == "context":
            # All the context bundle carries about a note.
            text = row.get("title", "")
        else:
            text = row.get("body", "")
            if with_sources and row.get("sources"):
                rendered = "; ".join(f"{s['type']} {s['value']}" for s in row["sources"])
                text = f"{text}  [MemLineage source: {rendered}]"
        records.append(MemoryRecord(text=text, label="", role="", rendering="memlineage"))
    return records


def _title(text: str) -> str:
    """A note title, as any agent writing the note would produce one: its opening clause,
    within MemLineage's 200-character limit."""
    head = text.strip().split(". ")[0]
    return head[:197] + "..." if len(head) > 200 else head


def _source_for(role: str) -> dict:
    source_type, value = SOURCE_BY_ROLE.get(role, ("text", "unspecified origin"))
    return {"type": source_type, "value": value}
