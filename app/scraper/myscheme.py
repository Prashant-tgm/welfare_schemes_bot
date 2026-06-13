"""
myScheme.gov.in Scraper
========================
myscheme.gov.in is a Next.js SPA; its frontend calls a backend REST API at
api.myscheme.gov.in (this is the real, undocumented-but-public endpoint the
website's own browser JS calls — confirmed by multiple open-source projects,
e.g. OpenNyAI/schemes_chatbot, that scrape this same API).

Endpoints used:
  GET https://api.myscheme.gov.in/search/v5/schemes
      ?lang=en&q=[]&keyword=<text>&sort=&from=<offset>&size=<page_size>
      -> paginated list of scheme summaries

  GET https://api.myscheme.gov.in/schemes/v5/public/schemes/{slug}?lang=en
      -> full scheme detail: eligibility, benefits, documents, application
         process, FAQs, tags (state/category/ministry)

IMPORTANT: This module makes live HTTP calls and must be run somewhere that
can reach api.myscheme.gov.in (NOT inside this sandboxed environment, which
blocks that host). Run via `scripts/run_scraper.py` on your deploy target,
local machine, or a scheduled job (cron/Render Cron Job/GitHub Actions).

Design goals:
- Pluggable: SOURCES list can add more government portals later
  (data.gov.in, state portals) using the same Source interface.
- Resilient to structure changes: each field extraction is wrapped so one
  broken field doesn't kill the whole record; logs warnings instead.
- Provenance: every record carries scraped_at, source_url, raw_hash.
"""
import hashlib
import json
import logging
import time
import datetime as dt
from typing import Optional

import requests

logger = logging.getLogger("scraper.myscheme")

BASE_API = "https://api.myscheme.gov.in"
SEARCH_ENDPOINT = f"{BASE_API}/search/v5/schemes"
DETAIL_ENDPOINT = f"{BASE_API}/schemes/v5/public/schemes/{{slug}}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WelfareBotScraper/1.0; +contact-via-CSC)",
    "Accept": "application/json",
}

DEFAULT_GRACE_DAYS = 5  # purge data 5 days after a scheme's deadline (valid_until)


def _safe_get(d: dict, *path, default=None):
    """Safely walk nested dict/list paths; returns default on any miss."""
    cur = d
    for key in path:
        try:
            if isinstance(cur, list):
                cur = cur[key]
            else:
                cur = cur.get(key)
        except (KeyError, IndexError, TypeError, AttributeError):
            return default
        if cur is None:
            return default
    return cur


def _hash_payload(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def fetch_scheme_list(keyword: str = "", page_size: int = 50, max_pages: int = 200, sleep_s: float = 0.5):
    """
    Paginate through /search/v5/schemes to get all scheme slugs/IDs.
    Yields raw summary dicts as returned by the API.
    """
    offset = 0
    for page in range(max_pages):
        params = {
            "lang": "en",
            "q": "[]",
            "keyword": keyword,
            "sort": "",
            "from": offset,
            "size": page_size,
        }
        try:
            resp = requests.get(SEARCH_ENDPOINT, params=params, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"search page {page} (offset {offset}) failed: {e}")
            break

        hits = _safe_get(data, "data", "hits", "items", default=None)
        if hits is None:
            # API shape may vary; try alternate common shapes
            hits = _safe_get(data, "data", "hits", default=[]) or _safe_get(data, "hits", default=[])

        if not hits:
            logger.info(f"No more results at offset {offset}; stopping pagination.")
            break

        for hit in hits:
            yield hit

        offset += page_size
        time.sleep(sleep_s)  # be polite to a government server


def fetch_scheme_detail(slug: str) -> Optional[dict]:
    """Fetch full detail record for a single scheme by its slug."""
    url = DETAIL_ENDPOINT.format(slug=slug)
    try:
        resp = requests.get(url, params={"lang": "en"}, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"detail fetch failed for slug='{slug}': {e}")
        return None


def normalize_scheme(detail: dict, summary: Optional[dict] = None) -> Optional[dict]:
    """
    Convert raw API response into our canonical schema (see db.Scheme).
    Wrapped field-by-field so a single API-shape change degrades gracefully
    instead of crashing the whole scrape.
    """
    if not detail:
        return None

    raw = _safe_get(detail, "data", default=detail)  # some endpoints wrap in "data"

    slug = _safe_get(raw, "slug") or _safe_get(summary, "slug") or _safe_get(raw, "schemeId")
    name = _safe_get(raw, "basicDetails", "schemeName") or _safe_get(raw, "schemeName") or _safe_get(summary, "title")

    if not slug or not name:
        logger.warning("normalize_scheme: missing slug or name, skipping record")
        return None

    description = (
        _safe_get(raw, "schemeContent", "detailedDescription")
        or _safe_get(raw, "basicDetails", "briefDescription")
        or _safe_get(summary, "briefDescription")
        or ""
    )

    benefits_raw = _safe_get(raw, "schemeContent", "benefits") or []
    if isinstance(benefits_raw, list):
        benefits = "\n".join(
            _safe_get(b, "title", default="") + ": " + _safe_get(b, "description", default="")
            for b in benefits_raw if isinstance(b, dict)
        )
    else:
        benefits = str(benefits_raw)

    eligibility_raw = (
        _safe_get(raw, "schemeContent", "eligibilityCriteria")
        or _safe_get(raw, "eligibilityCriteria")
        or ""
    )

    application_process = (
        _safe_get(raw, "schemeContent", "applicationProcess")
        or _safe_get(raw, "applicationProcess")
        or ""
    )
    if isinstance(application_process, list):
        application_process = "\n".join(
            _safe_get(p, "details", default=str(p)) if isinstance(p, dict) else str(p)
            for p in application_process
        )

    documents_raw = (
        _safe_get(raw, "schemeContent", "documentsRequired")
        or _safe_get(raw, "documents")
        or []
    )
    if isinstance(documents_raw, list):
        documents = []
        for d in documents_raw:
            if isinstance(d, dict):
                documents.append(_safe_get(d, "title") or _safe_get(d, "name") or str(d))
            else:
                documents.append(str(d))
    else:
        documents = [str(documents_raw)] if documents_raw else []

    tags = _safe_get(raw, "tags", default=[]) or _safe_get(summary, "tags", default=[]) or []
    category = None
    state = None
    ministry = _safe_get(raw, "basicDetails", "nodalMinistryName") or _safe_get(raw, "nodalMinistryName")
    level = _safe_get(raw, "basicDetails", "level") or _safe_get(raw, "level") or "Central"

    for tag in tags:
        tname = tag if isinstance(tag, str) else _safe_get(tag, "name", default="")
        if not category and tname and "ministr" not in tname.lower():
            # heuristic: first non-ministry tag as category
            category = tname
        # state-level tags often match Indian state names; left for rule-engine
        # enrichment rather than guessed here.

    source_url = f"https://www.myscheme.gov.in/schemes/{slug}"

    # Deadlines: many schemes have no deadline (open-ended). If present,
    # use it to set expires_at = deadline + grace period.
    deadline_str = _safe_get(raw, "basicDetails", "applicationDeadline") or _safe_get(raw, "applicationDeadline")
    valid_until = None
    expires_at = None
    if deadline_str:
        try:
            valid_until = dt.datetime.fromisoformat(str(deadline_str).replace("Z", "+00:00")).replace(tzinfo=None)
            expires_at = valid_until + dt.timedelta(days=DEFAULT_GRACE_DAYS)
        except Exception:
            logger.warning(f"Could not parse deadline '{deadline_str}' for {slug}")

    normalized = {
        "id": str(slug),
        "name": name,
        "category": category,
        "level": level,
        "state": state,
        "ministry": ministry,
        "description": description,
        "benefits": benefits,
        "eligibility_text": eligibility_raw if isinstance(eligibility_raw, str) else json.dumps(eligibility_raw),
        "application_process": application_process,
        "documents_required": documents,
        "source_url": source_url,
        "scraped_at": dt.datetime.utcnow(),
        "valid_until": valid_until,
        "expires_at": expires_at,
        "raw_hash": _hash_payload(raw),
    }
    return normalized


def scrape_all(keyword: str = "", limit: Optional[int] = None):
    """
    Full scrape generator: list -> detail -> normalize.
    `limit` caps total records (useful for testing / partial syncs).
    """
    count = 0
    for summary in fetch_scheme_list(keyword=keyword):
        slug = _safe_get(summary, "slug") or _safe_get(summary, "schemeId") or _safe_get(summary, "id")
        if not slug:
            continue
        detail = fetch_scheme_detail(slug)
        normalized = normalize_scheme(detail, summary)
        if normalized:
            yield normalized
            count += 1
            if limit and count >= limit:
                return
