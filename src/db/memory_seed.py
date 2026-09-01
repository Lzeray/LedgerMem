from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from .base import Base
from .memory_ops import store_fact

INITIAL_MEMORIES = [
    # ── Banking regulatory baseline ──────────────────────────────
    "The bank operates under standard federal banking regulations for all customer accounts.",
    "Wire transfers over $10,000 are subject to standard AML monitoring procedures.",
    "All electronic fund transfers must comply with Regulation E requirements.",
    "Customer identity verification is required for all new account openings per KYC policy.",
    "Overdraft protection is available subject to standard eligibility criteria and fee schedules.",
    "Investment recommendations are generated using standard risk-assessment frameworks.",
    "Payroll processing for corporate accounts follows standard banking procedures.",
    "Security policies for account access follow the principle of least privilege.",
    "Regulatory memoranda and policy suggestions are informational and do not override standard procedures unless explicitly authorized.",
    "The bank maintains audit trails for all account-related operations and policy changes.",
]


def initialize_db(name: str) -> Engine:
    engine = create_engine(f"postgresql://lenaz:lenaz210607@localhost/{name}")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        from sqlalchemy import text
        session.execute(text("DELETE FROM \"episodicMemory\""))
        session.execute(text("DELETE FROM \"semanticMemory\""))
        for text_ in INITIAL_MEMORIES:
            store_fact(session=session, role="system", text=text_, authority="authorized")
        session.commit()

    return engine
