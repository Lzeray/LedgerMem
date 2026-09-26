"""
Read/write access to the authority-labeled memory.

Two access paths, deliberately different in kind:

  recall_facts()  — semantic (vector) search. This is what the *agent* uses to bring
                    context into its window. It is approximate by nature and nothing
                    security-relevant depends on it.

  lookup_slot()   — exact lookup of a structured operative value by slot key, ordered
                    most-trusted-first. This is what the *authority gate* uses. Keeping
                    the gate off the vector path is the point: in the previous
                    implementation the defense's verdict could flip because a verbose
                    phrasing lost a top-1 cosine search, which measures the retriever,
                    not the defense.

There is deliberately no near-duplicate dedup on write. The benchmark writes memory
deterministically from a script; silently returning an existing row's id (and keeping its
older label) is exactly the failure the previous version had to work around.
"""

import json
from functools import lru_cache

from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session

from new_src.config import DB_URL, EMBEDDING_MODEL
from new_src.memory.models import (
    AuthorityLabel,
    Base,
    Channel,
    ClaimType,
    EpisodicRecord,
    Rendering,
    SemanticRecord,
    SourceRole,
)

# Most-trusted first. Mirrors the gate's own check order.
LABEL_RANK: dict[str, int] = {"authorized": 0, "attested": 1, "unendorsed": 2}


@lru_cache(maxsize=1)
def _embedder():
    # Imported lazily: sentence-transformers pulls in torch, which costs seconds at import
    # time and is not needed by the schema-validation entry points.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed(value: str):
    return _embedder().encode(value)


#: Columns added after the table first shipped. create_all() does not alter an existing
#: table, so they are added here rather than requiring the database to be dropped by hand.
_ADDED_COLUMNS = (
    ("claim_type", "VARCHAR"),
    ("object_ref", "VARCHAR"),
    ("verbatim", "TEXT"),
    ("channel", "VARCHAR"),
    ("requests", "TEXT"),
    ("arguments", "TEXT"),
)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        for column, sql_type in _ADDED_COLUMNS:
            connection.execute(text(f'ALTER TABLE am_semantic ADD COLUMN IF NOT EXISTS {column} {sql_type}'))
    return engine


def reset_memory(engine: Engine | None = None) -> Engine:
    """Empty both tables. Every episode starts from a clean store so no result can depend
    on residue from the previous episode."""
    engine = engine or get_engine()
    with Session(engine) as session:
        session.execute(text('DELETE FROM "am_episodic"'))
        session.execute(text('DELETE FROM "am_semantic"'))
        session.commit()
    return engine


# --- writes ----------------------------------------------------------------


#: Actions are stored between delimiters so an exact name can be matched without one action
#: name matching inside another.
_DELIM = ","


def encode_requests(actions: list[str] | None) -> str | None:
    """Store a request list. An empty list and "no list at all" are stored differently on
    purpose: an empty list means the question was asked and the answer was "nothing", while
    NULL means it was never asked (a channel whose label can never be `authorized`, so the
    answer could not have mattered)."""
    if actions is None:
        return None
    return _DELIM + _DELIM.join(sorted(set(actions))) + _DELIM


def decode_requests(stored: str | None) -> list[str]:
    if not stored:
        return []
    return [name for name in stored.split(_DELIM) if name]


def decode_arguments(stored: str | None) -> dict:
    """{action: {parameter: value}} as written by `write_fact`. Anything unreadable is empty: a
    request whose arguments cannot be read names none, so they are resolved from memory instead."""
    if not stored:
        return {}
    try:
        value = json.loads(stored)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def lookup_requesting(session: Session, action_name: str) -> list[SemanticRecord]:
    """Every record asking for this action, most-trusted first. Exact match inside the
    delimited list — no vector search, no model, no string heuristic over free text."""
    needle = f"%{_DELIM}{action_name}{_DELIM}%"
    rows = list(session.scalars(
        select(SemanticRecord).where(SemanticRecord.requests.like(needle))
    ).all())
    return sorted(rows, key=lambda r: (LABEL_RANK[r.label], r.id))


#: Channels that can never produce `authorized`, whatever else happens upstream.
#:
#: Each channel has a ceiling: the customer's own words `authorized`, a bank system and the agent
#: `attested`, an outside feed `unendorsed`. What a record says can lower it below its ceiling
#: (a quotation), never raise it above. So `authorized` comes from the customer's channel alone
#: (and from the bank's own books, which are seeded rather than written); no classifier answer,
#: parameter, grant or inference may make it come from anywhere else.
#:
#: This is a hard stop rather than a clamp because a clamp hides the defect. A write that
#: reaches here with `authorized` on one of these channels means something upstream decided a
#: label it had no standing to decide, and that is worth failing the episode over — it is how
#: module_c's channel prediction was caught, after it had quietly turned an untrusted tool's
#: claim into the customer's own words and let a payment through.
_NEVER_AUTHORIZED = ("assistant", "trusted_tool", "untrusted_tool")


#: The channel a record without one is checked as, from its role. The paper's own arms (the frozen
#: role policy, the Module C predictor and oracle) write no channel at all, and the guard must still
#: hold for them. A tool of unknown trust is checked as an outside feed: the conservative reading.
_CHANNEL_OF_ROLE = {"user": "user", "system": "system", "assistant": "assistant", "tool": "untrusted_tool"}


def _refuse_impossible_authority(channel: Channel | None, label: AuthorityLabel,
                                 role: SourceRole | None = None) -> None:
    channel = channel or _CHANNEL_OF_ROLE.get(role)
    if label == "authorized" and channel is None:
        raise ValueError("refusing to store an 'authorized' record with neither a channel nor a role: "
                         "authority has to come from somewhere structural")
    if label == "authorized" and channel in _NEVER_AUTHORIZED:
        raise ValueError(
            f"refusing to store an 'authorized' record on the {channel!r} channel: "
            "only the customer's own words can be authorized"
        )


def write_fact(
    session: Session,
    fact_text: str,
    label: AuthorityLabel,
    role: SourceRole,
    rendering: Rendering = "source_attributed",
    slot_key: str | None = None,
    slot_value: str | None = None,
    claim_type: ClaimType | None = None,
    object_ref: str | None = None,
    verbatim: str | None = None,
    channel: Channel | None = None,
    requests: list[str] | None = None,
    arguments: dict | None = None,
) -> int:
    _refuse_impossible_authority(channel, label, role)
    record = SemanticRecord(
        fact_text=fact_text,
        label=label,
        role=role,
        rendering=rendering,
        slot_key=slot_key,
        slot_value=slot_value,
        claim_type=claim_type,
        object_ref=object_ref,
        verbatim=verbatim,
        channel=channel,
        requests=encode_requests(requests),
        arguments=json.dumps(arguments, sort_keys=True) if arguments else None,
        embedding=embed(fact_text),
    )
    session.add(record)
    session.commit()
    return record.id


def write_episode(session: Session, role: SourceRole, content: str) -> int:
    record = EpisodicRecord(role=role, content=content, embedding=embed(content))
    session.add(record)
    session.commit()
    return record.id


def consume_request(session: Session, record_ids: list[int], action_name: str) -> None:
    """Spend a licence: the records that licensed an executed action no longer list it, and no
    longer carry arguments for it. One request licenses one execution; asking again licenses
    again. Labels are not touched — this can only ever take authority away."""
    if not record_ids:
        return
    for row in session.scalars(select(SemanticRecord).where(SemanticRecord.id.in_(record_ids))).all():
        row.requests = encode_requests([name for name in decode_requests(row.requests) if name != action_name])
        arguments = decode_arguments(row.arguments)
        arguments.pop(action_name, None)
        row.arguments = json.dumps(arguments, sort_keys=True) if arguments else None
    session.commit()


# --- reads -----------------------------------------------------------------


def recall_facts(
    session: Session,
    query: str,
    top_k: int = 5,
    label: AuthorityLabel | None = None,
    ids: list[int] | None = None,
) -> list[SemanticRecord]:
    """`ids` restricts the search to those rows (e.g. the consolidated memory, not the gate's
    own write-time journal)."""
    stmt = (
        select(SemanticRecord)
        .order_by(SemanticRecord.embedding.cosine_distance(embed(query)))
        .limit(top_k)
    )
    if label:
        stmt = stmt.where(SemanticRecord.label == label)
    if ids is not None:
        stmt = stmt.where(SemanticRecord.id.in_(ids))
    return list(session.scalars(stmt).all())


def recall_episodes(session: Session, query: str, top_k: int = 5) -> list[EpisodicRecord]:
    stmt = (
        select(EpisodicRecord)
        .order_by(EpisodicRecord.embedding.cosine_distance(embed(query)))
        .limit(top_k)
    )
    return list(session.scalars(stmt).all())


def lookup_slot(session: Session, slot_key: str) -> list[SemanticRecord]:
    """Every record carrying this operative value, most-trusted label first. Exact match on
    slot_key — no vector search, no model in the loop."""
    rows = list(session.scalars(select(SemanticRecord).where(SemanticRecord.slot_key == slot_key)).all())
    return sorted(rows, key=lambda r: (LABEL_RANK[r.label], r.id))


def lookup_claims(session: Session, object_ref: str | None) -> list[SemanticRecord]:
    """Records that make a claim about `object_ref`, most-trusted first. When object_ref is
    None the action has no object of its own, so every record that could license something is
    a candidate and the licence check falls back to naming the action itself.

    Records carrying no object reference are deliberately NOT returned for an action that has
    one. Admitting them was tried and reverted: it leaves a wording heuristic as the only
    barrier between a stray authorized sentence and an executed action, and a customer's
    question about an account was enough to get through.
    """
    stmt = select(SemanticRecord)
    if object_ref is not None:
        stmt = stmt.where(SemanticRecord.object_ref == object_ref)
    rows = list(session.scalars(stmt).all())
    return sorted(rows, key=lambda r: (LABEL_RANK[r.label], r.id))


def all_facts(session: Session) -> list[SemanticRecord]:
    """The complete immutable memory list — what Module A's judge is given."""
    return list(session.scalars(select(SemanticRecord).order_by(SemanticRecord.id)).all())
