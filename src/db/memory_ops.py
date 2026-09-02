
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.orm import Session

from .base import AUTHORITY_LEVELS, ROLES, EpisodicMemory, SemanticMemory

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Semantic memory (persistent facts about the world/user) ---


def store_fact(session: Session, text: str, label: AUTHORITY_LEVELS = "unendorsed", role: ROLES = "system") -> int:
    vec = embedder.encode(text)
    existing = session.execute(
        select(SemanticMemory).where(
            1 - SemanticMemory.embedding.cosine_distance(vec) > 0.95
        )
    ).scalars().first()
    if existing:
        return existing.id
    record = SemanticMemory(role=role, fact_text=text, label=label, embedding=vec)
    session.add(record)
    session.commit()
    return record.id


def recall_facts(session: Session, query: str, label: AUTHORITY_LEVELS | None = None, top_k: int = 3) -> list[SemanticMemory]:
    q_vec = embedder.encode(query)
    stmt = (
        select(SemanticMemory)
        .order_by(SemanticMemory.embedding.cosine_distance(q_vec))
        .limit(top_k)
    )
    if label:
        stmt = stmt.where(SemanticMemory.label == label)
    return list(session.scalars(stmt).all())


# --- Episodic memory (conversation turns from the current run) ---


def store_episode(session: Session, role: ROLES, text: str) -> int:
    vec = embedder.encode(text)
    record = EpisodicMemory(role=role, content=text, embedding=vec)
    session.add(record)
    session.commit()
    return record.id


def recall_episodes(session: Session, query: str, top_k: int = 5) -> list[EpisodicMemory]:
    q_vec = embedder.encode(query)
    stmt = (
        select(EpisodicMemory)
        .order_by(EpisodicMemory.embedding.cosine_distance(q_vec))
        .limit(top_k)
    )
    return list(session.scalars(stmt).all())
