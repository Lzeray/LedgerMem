"""
The memory store, with the paper's authority metadata as first-class columns.

Two tables, deliberately separate from src/'s (`semanticMemory`/`episodicMemory`) so the
old suite and this one can coexist in one database:

  am_semantic  — consolidated memory records. Besides the text, each row carries the three
                 things the benchmark actually measures:
                   * label      — the authority label (authorized/attested/unendorsed)
                   * role       — the source role the label was derived from, kept so a
                                  run can be re-scored under a different role policy
                   * rendering  — 'source_attributed' (the record still names its source
                                  condition) or 'washed' (the proposition survives without
                                  it). Module B tests both; a washed record is exactly the
                                  authority collapse the paper is about.
                   * slot_key / slot_value — the structured operative value. The action
                                  gate reads these, never free text, so the defense's
                                  decision is deterministic and never depends on a small
                                  model's extraction (the single largest noise source in
                                  the previous implementation).

  am_episodic  — raw conversation turns, unlabeled, as context only.
"""

from typing import Literal

from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from new_src.config import EMBEDDING_DIM

# The paper's three operational authority classes.
AuthorityLabel = Literal["authorized", "attested", "unendorsed"]

# Source roles the frozen role policy maps from.
SourceRole = Literal["user", "assistant", "tool", "system"]

Rendering = Literal["source_attributed", "washed"]

# What KIND of claim a record makes. The authority label says who is behind a record; the
# claim type says what sort of thing it is. A protected action names which kinds may license
# it, so that an observation cannot stand in for an intention, or a suggestion for a decision.
ClaimType = Literal[
    "fact", "rule", "observation", "suggestion", "recommendation",
    "preference", "decision", "intention", "acknowledgement",
    # Added with the channel model (see bench/taxonomy.py). The first four name speech acts the
    # role-derived mapping could not express at all: a customer quoting someone else, a customer
    # declining, `system` stating a policy rather than a fact about anyone, and a tool carrying
    # an authorization. `other` is what an unparsable or out-of-set classification becomes, and
    # it is in no action's licence list, so it can only ever cause a refusal.
    "quotation", "refusal", "policy", "grant", "question", "other",
]

# The channel a record arrived on. Unlike `role`, it distinguishes the bank's own systems of
# record from outside feeds, which is what decides whether a tool may carry an authorization.
Channel = Literal["user", "system", "trusted_tool", "untrusted_tool", "assistant"]


class Base(DeclarativeBase):
    id: Mapped[int] = mapped_column(autoincrement=True, primary_key=True)
    embedding: Mapped[Vector] = mapped_column(Vector(EMBEDDING_DIM))


class SemanticRecord(Base):
    __tablename__ = "am_semantic"

    fact_text: Mapped[str] = mapped_column(nullable=False)
    label: Mapped[AuthorityLabel] = mapped_column(nullable=False)
    role: Mapped[SourceRole] = mapped_column(nullable=False)
    rendering: Mapped[Rendering] = mapped_column(nullable=False, default="source_attributed")
    # Structured operative value. NULL for background/context facts that no action reads.
    slot_key: Mapped[str] = mapped_column(nullable=True)
    slot_value: Mapped[str] = mapped_column(nullable=True)
    # What kind of claim this is, and what it is about. The authority gate's licence check
    # reads these two; nothing else does.
    claim_type: Mapped[ClaimType] = mapped_column(nullable=True)
    object_ref: Mapped[str] = mapped_column(nullable=True)
    # The customer's own words, stored unaltered, when this record came from them. A licence
    # is only ever granted from verbatim text: a consolidator's paraphrase of a question
    # ("does it make sense to close the account?") must never become an instruction.
    verbatim: Mapped[str] = mapped_column(nullable=True)
    # The channel the claim arrived on, and the ceiling that channel imposes. Written alongside
    # `role` rather than instead of it: a run under the frozen role policy leaves this NULL, and
    # the two label sources stay comparable on the same table.
    channel: Mapped[str] = mapped_column(nullable=True)
    # Which protected actions this record could be asking for, stored comma-delimited with
    # leading and trailing commas (",close_savings_account,") so an exact action can be matched
    # without one name being a substring of another.
    #
    # This is what licenses an action, and it replaces both halves of the old machinery: the
    # claim-type matching and the string test that guessed whether an utterance "named" the
    # action. A record licenses an action when that action is in this list AND the record's
    # label is `authorized` — nothing else is consulted.
    #
    # An empty list is the safe default and the commonest value: a question, a refusal, a policy
    # statement and a bank system's observation all request nothing.
    requests: Mapped[str] = mapped_column(nullable=True)

    def __repr__(self) -> str:
        return (
            f"SemanticRecord(id={self.id!r}, slot={self.slot_key!r}={self.slot_value!r}, "
            f"label={self.label!r}, role={self.role!r}, channel={self.channel!r}, "
            f"claim_type={self.claim_type!r}, requests={self.requests!r}, "
            f"object_ref={self.object_ref!r}, rendering={self.rendering!r}, "
            f"fact_text={self.fact_text!r})"
        )


class EpisodicRecord(Base):
    __tablename__ = "am_episodic"

    content: Mapped[str] = mapped_column(nullable=False)
    role: Mapped[SourceRole] = mapped_column(nullable=False)

    def __repr__(self) -> str:
        return f"EpisodicRecord(id={self.id!r}, role={self.role!r}, content={self.content!r})"
