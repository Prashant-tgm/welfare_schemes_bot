"""
Database layer — pluggable SQLite (default) / Postgres via SQLAlchemy.

Tables:
- schemes: canonical scheme records with scrape provenance + expiry
- scheme_chunks: text chunks for embedding/retrieval, linked to scheme + source
- embeddings: vector store (sqlite-vec compatible / fallback brute-force)
- rules: JSON or python-expression eligibility rules per scheme
- claim_graph: provenance edges (claim -> rule/chunk -> source) for audit
"""
import os
import json
import datetime as dt
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, Boolean,
    ForeignKey, JSON, Float
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./welfare.db")

# SQLite needs check_same_thread=False for FastAPI's threaded workers
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Scheme(Base):
    __tablename__ = "schemes"

    id = Column(String, primary_key=True)            # slug, e.g. "pm-kisan"
    name = Column(String, nullable=False)
    category = Column(String)                         # e.g. "Agriculture, Rural & Environment"
    level = Column(String)                            # "Central" | "State"
    state = Column(String, nullable=True)             # state name if state-level
    ministry = Column(String, nullable=True)
    description = Column(Text)
    benefits = Column(Text)
    application_process = Column(Text)
    documents_required = Column(JSON)                 # list[str]
    source_url = Column(String)

    # --- provenance / TTL fields ---
    source_name = Column(String, default="myscheme.gov.in")
    scraped_at = Column(DateTime, default=dt.datetime.utcnow)
    valid_until = Column(DateTime, nullable=True)      # scheme's own deadline (if any)
    expires_at = Column(DateTime, nullable=True)       # valid_until + grace period (auto-purge)
    raw_hash = Column(String, nullable=True)           # hash of raw scraped payload, for change detection
    is_stale = Column(Boolean, default=False)

    rules = relationship("Rule", back_populates="scheme", cascade="all, delete-orphan")
    chunks = relationship("SchemeChunk", back_populates="scheme", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "level": self.level,
            "state": self.state,
            "ministry": self.ministry,
            "description": self.description,
            "benefits": self.benefits,
            "application_process": self.application_process,
            "documents_required": self.documents_required or [],
            "source_url": self.source_url,
            "scraped_at": self.scraped_at.isoformat() if self.scraped_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class Rule(Base):
    """
    Hybrid eligibility rule storage:
    - rule_type='json'   -> rule_body is a JSON condition tree (AND/OR/comparators)
    - rule_type='python' -> rule_body is a python expression string,
                             evaluated in a restricted namespace (escape hatch
                             for state-specific / complex regulatory logic)
    """
    __tablename__ = "rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scheme_id = Column(String, ForeignKey("schemes.id"))
    rule_type = Column(String, default="json")   # "json" | "python"
    rule_body = Column(JSON)                      # for json rules: condition tree dict
    rule_code = Column(Text, nullable=True)       # for python rules: expression string
    description = Column(Text)                   # human-readable explanation, for citation
    source_url = Column(String)                  # where this specific rule came from
    priority = Column(Integer, default=0)         # for ordering/overrides

    scheme = relationship("Scheme", back_populates="rules")


class SchemeChunk(Base):
    """Text chunks for RAG retrieval, each traceable to a scheme + source."""
    __tablename__ = "scheme_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scheme_id = Column(String, ForeignKey("schemes.id"))
    chunk_type = Column(String)   # "description" | "benefits" | "eligibility" | "documents" | "process" | "faq"
    text = Column(Text)
    source_url = Column(String)
    embedding = Column(JSON, nullable=True)  # list[float], stored as JSON for portability
    scraped_at = Column(DateTime, default=dt.datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    scheme = relationship("Scheme", back_populates="chunks")


class ClaimGraphEdge(Base):
    """
    Provenance graph: every claim the bot makes is logged as an edge
    claim_text -> (scheme_id, chunk_id/rule_id, source_url, scraped_at)
    so a graph of "what supports this answer" can be rendered for audit.
    """
    __tablename__ = "claim_graph_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, index=True)
    turn_index = Column(Integer)
    claim_text = Column(Text)
    scheme_id = Column(String, nullable=True)
    chunk_id = Column(Integer, nullable=True)
    rule_id = Column(Integer, nullable=True)
    source_url = Column(String, nullable=True)
    source_scraped_at = Column(DateTime, nullable=True)
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class ConversationState(Base):
    """Session/conversation state, including satisfaction tracking."""
    __tablename__ = "conversation_state"

    phone = Column(String, primary_key=True)
    state = Column(String)
    lang = Column(String)
    user_json = Column(JSON)
    turn_count = Column(Integer, default=0)
    satisfied = Column(Boolean, default=False)
    handed_off = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def purge_expired(db, now: dt.datetime = None):
    """Delete schemes/chunks past their expires_at (deadline + grace period)."""
    now = now or dt.datetime.utcnow()
    expired_schemes = db.query(Scheme).filter(
        Scheme.expires_at.isnot(None), Scheme.expires_at < now
    ).all()
    count = len(expired_schemes)
    for s in expired_schemes:
        db.delete(s)  # cascades to rules + chunks

    expired_chunks = db.query(SchemeChunk).filter(
        SchemeChunk.expires_at.isnot(None), SchemeChunk.expires_at < now
    ).all()
    for c in expired_chunks:
        db.delete(c)

    db.commit()
    return count, len(expired_chunks)
