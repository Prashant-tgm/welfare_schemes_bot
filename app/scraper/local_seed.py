"""
Local Seed Source
==================
Implements the same `scrape_all()` interface as app.scraper.myscheme, but
reads from a bundled local JSON file (data/schemes_seed.json) instead of
hitting the live API. Use this:

- For local testing / CI where api.myscheme.gov.in isn't reachable
- As an initial seed so the bot has data on first deploy, before the
  first live sync job runs
- As a fallback source if the live scraper fails entirely

Each record is converted into the same canonical schema as
app.scraper.myscheme.normalize_scheme() output, including provenance
fields (scraped_at, source_url, raw_hash). Since this is curated/manual
data (not freshly scraped), `valid_until`/`expires_at` are left None
(no auto-expiry) unless explicitly set in the seed file.
"""
import json
import os
import hashlib
import datetime as dt

SEED_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "schemes_seed.json")

DEFAULT_GRACE_DAYS = 5


def _hash_payload(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_seed():
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def scrape_all(keyword="", limit=None):
    """Yields normalized scheme dicts from the local seed file."""
    records = _load_seed()
    count = 0
    for r in records:
        elig = r.get("eligibility", {})
        normalized = {
            "id": r["id"].replace("_", "-"),
            "name": r["name"],
            "category": elig.get("category"),
            "level": "Central",
            "state": None,
            "ministry": None,
            "description": r.get("description", ""),
            "benefits": r.get("benefits", ""),
            "eligibility_text": json.dumps(elig),  # structured -> stringified for chunking
            "application_process": r.get("application_process", ""),
            "documents_required": r.get("documents_required", []),
            "source_url": r.get("source_url"),
            "scraped_at": dt.datetime.utcnow(),
            "valid_until": None,
            "expires_at": None,
            "raw_hash": _hash_payload(r),
            "_raw_eligibility": elig,  # passed through for rule generation
        }
        yield normalized
        count += 1
        if limit and count >= limit:
            return
