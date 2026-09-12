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


def lookup_requesting(session: Session, action_name: str) -> list[SemanticRecord]:
    """Every record asking for this action, most-trusted first. Exact match inside the
    delimited list — no vector search, no model, no string heuristic over free text."""
    needle = f"%{_DELIM}{action_name}{_DELIM}%"
    rows = list(session.scalars(
        select(SemanticRecord).where(SemanticRecord.requests.like(needle))
    ).all())
    return sorted(rows, key=lambda r: (LABEL_RANK[r.label], r.id))


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
) -> int:
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


# --- reads -----------------------------------------------------------------


def recall_facts(
    session: Session,
    query: str,
    top_k: int = 5,
    label: AuthorityLabel | None = None,
) -> list[SemanticRecord]:
    stmt = (
        select(SemanticRecord)
        .order_by(SemanticRecord.embedding.cosine_distance(embed(query)))
        .limit(top_k)
    )
    if label:
        stmt = stmt.where(SemanticRecord.label == label)
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
