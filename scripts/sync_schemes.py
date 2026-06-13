"""
Sync Script
===========
Run periodically (cron / Render Cron Job / GitHub Actions schedule) to:

1. Scrape all registered sources (app/scraper/registry.py)
2. Upsert scheme records into the DB, preserving scraped_at/expires_at
3. Re-chunk + re-embed changed schemes for RAG retrieval
4. Purge schemes/chunks past `expires_at` (deadline + grace period)

Usage:
    python scripts/sync_schemes.py                  # full sync, all sources
    python scripts/sync_schemes.py --limit 50        # cap records (testing)
    python scripts/sync_schemes.py --purge-only      # only run expiry purge
    python scripts/sync_schemes.py --source myscheme.gov.in
"""
import argparse
import datetime as dt
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db import init_db, get_session, SessionLocal, Scheme, SchemeChunk, purge_expired
from app.scraper.registry import SOURCES
from app.rules.engine import generate_default_rules
from app.rag.embeddings import embed_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sync")


CHUNK_FIELDS = [
    ("description", "Description"),
    ("benefits", "Benefits"),
    ("application_process", "How to Apply"),
    ("eligibility_text", "Eligibility"),
]


def upsert_scheme(db, record: dict, source_name: str, grace_days: int):
    scheme = db.query(Scheme).get(record["id"])
    now = dt.datetime.utcnow()

    expires_at = record.get("expires_at")
    valid_until = record.get("valid_until")

    if scheme is None:
        scheme = Scheme(id=record["id"])
        db.add(scheme)
        logger.info(f"  + new scheme: {record['id']}")
    else:
        if scheme.raw_hash == record.get("raw_hash"):
            # unchanged — just refresh scraped_at/expiry, skip re-chunking
            scheme.scraped_at = now
            scheme.expires_at = expires_at
            scheme.valid_until = valid_until
            scheme.is_stale = False
            return scheme, False  # changed=False

        logger.info(f"  ~ updated scheme: {record['id']}")

    scheme.name = record["name"]
    scheme.category = record.get("category")
    scheme.level = record.get("level")
    scheme.state = record.get("state")
    scheme.ministry = record.get("ministry")
    scheme.description = record.get("description", "")
    scheme.benefits = record.get("benefits", "")
    scheme.application_process = record.get("application_process", "")
    scheme.documents_required = record.get("documents_required", [])
    scheme.source_url = record.get("source_url")
    scheme.source_name = source_name
    scheme.scraped_at = now
    scheme.valid_until = valid_until
    scheme.expires_at = expires_at
    scheme.raw_hash = record.get("raw_hash")
    scheme.is_stale = False

    return scheme, True  # changed=True


def rechunk_scheme(db, scheme: Scheme, record: dict, expires_at):
    """Delete old chunks, create new ones with embeddings."""
    db.query(SchemeChunk).filter(SchemeChunk.scheme_id == scheme.id).delete()

    for field_key, label in CHUNK_FIELDS:
        text = record.get(field_key) or ""
        if not text.strip():
            continue
        full_text = f"[{scheme.name}] {label}: {text}"
        embedding = embed_text(full_text)
        chunk = SchemeChunk(
            scheme_id=scheme.id,
            chunk_type=field_key,
            text=full_text,
            source_url=scheme.source_url,
            embedding=embedding,
            scraped_at=dt.datetime.utcnow(),
            expires_at=expires_at,
        )
        db.add(chunk)

    # Documents-required chunk
    docs = record.get("documents_required") or []
    if docs:
        text = f"[{scheme.name}] Documents required: {', '.join(docs)}"
        chunk = SchemeChunk(
            scheme_id=scheme.id,
            chunk_type="documents",
            text=text,
            source_url=scheme.source_url,
            embedding=embed_text(text),
            scraped_at=dt.datetime.utcnow(),
            expires_at=expires_at,
        )
        db.add(chunk)


def sync(limit=None, purge_only=False, only_source=None):
    init_db()
    db = SessionLocal()

    if purge_only:
        n_schemes, n_chunks = purge_expired(db)
        logger.info(f"Purged {n_schemes} expired schemes, {n_chunks} expired chunks")
        db.close()
        return

    total_new, total_updated, total_unchanged = 0, 0, 0

    for source in SOURCES:
        if only_source and source["name"] != only_source:
            continue

        logger.info(f"Scraping source: {source['name']}")
        module = source["module"]
        grace_days = source["default_grace_days"]

        for record in module.scrape_all(limit=limit):
            scheme, changed = upsert_scheme(db, record, source["name"], grace_days)
            if changed:
                rechunk_scheme(db, scheme, record, scheme.expires_at)

                # generate default JSON rules from eligibility text
                # (escape hatch: complex schemes can later be overridden with
                #  rule_type='python' entries — see app/rules/engine.py)
                generate_default_rules(db, scheme, record)

                if scheme in db.new:
                    total_new += 1
                else:
                    total_updated += 1
            else:
                total_unchanged += 1

            db.commit()

    n_schemes, n_chunks = purge_expired(db)

    logger.info(
        f"Sync complete. new={total_new} updated={total_updated} "
        f"unchanged={total_unchanged} purged_schemes={n_schemes} purged_chunks={n_chunks}"
    )
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Cap number of records per source")
    parser.add_argument("--purge-only", action="store_true", help="Only run expiry purge, skip scraping")
    parser.add_argument("--source", type=str, default=None, help="Only sync this source name")
    args = parser.parse_args()

    sync(limit=args.limit, purge_only=args.purge_only, only_source=args.source)
