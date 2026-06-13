"""
Source Registry
================
Pluggable list of scraper sources. Each source exposes:
  - name: str
  - scrape_all(keyword="", limit=None) -> generator of normalized dicts
    (matching the canonical schema in app/db.py::Scheme)

Add new government portals (data.gov.in, state portals, Bhashini scheme
corpora, etc.) by writing a similarly-shaped module and registering it here.
The sync script (scripts/sync_schemes.py) iterates SOURCES uniformly.
"""
from app.scraper import myscheme, local_seed

SOURCES = [
    {
        "name": "local_seed",
        "module": local_seed,
        "default_grace_days": local_seed.DEFAULT_GRACE_DAYS,
    },
    {
        "name": "myscheme.gov.in",
        "module": myscheme,
        "default_grace_days": myscheme.DEFAULT_GRACE_DAYS,
    },
]
