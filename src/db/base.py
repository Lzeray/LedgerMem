from typing import Literal

from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

ROLES = Literal["user", "assistant", "system", "tool"]

class Base(DeclarativeBase):
    role: Mapped[ROLES] = mapped_column(nullable=False)
    id: Mapped[int] = mapped_column(autoincrement=True, primary_key=True)
    embedding: Mapped[Vector] = mapped_column(Vector(384))

AUTHORITY_LEVELS = Literal["authorized", "attested", "unendorsed"]
class SemanticMemory(Base):
    __tablename__ = "semanticMemory"

    fact_text: Mapped[str] = mapped_column(nullable=False)
    authority: Mapped[AUTHORITY_LEVELS] = mapped_column(nullable=False)

    def __repr__(self):
        return (
            f"SemanticMemory(id={self.id!r}, fact_text={self.fact_text!r}, "
            f"authority={self.authority!r})"
        )

class EpisodicMemory(Base):
    __tablename__ = "episodicMemory"

    content: Mapped[str] = mapped_column(nullable=False)

    def __repr__(self):
        return f"EpisodicMemory(id={self.id!r}, role={self.role!r}, content={self.content!r})"
