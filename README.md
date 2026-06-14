# Welfare Scheme Chatbot

A robust, auditable, multilingual welfare-scheme assistant. v2 adds: live
scraping with TTL/expiry, embeddings-based retrieval across ALL synced
schemes (not a hardcoded shortlist), a hybrid (JSON + Python) eligibility
rules engine, a bounded multi-turn satisfaction loop, and a **claim graph**
that proves every bot statement back to a source URL + scrape timestamp.

## test on this by initiating 'hi'
wa.me/+14155238886
## Architecture

```
                    ┌─────────────────────────┐
   Scheduled job →  │ scripts/sync_schemes.py  │
   (cron/Render     │  - scrape sources        │
    Cron Job)       │  - upsert + TTL/expiry    │
                    │  - generate hybrid rules  │
                    │  - re-embed changed chunks│
                    │  - purge expired records  │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │   welfare.db (SQLAlchemy) │
                    │ schemes / rules /         │
                    │ scheme_chunks /           │
                    │ claim_graph_edges /        │
                    │ conversation_state         │
                    └──────────┬───────────────┘
                               │
   WhatsApp ──> FastAPI webhook │
                    │  Phase 1: 6-turn intake    │
                    │   -> hybrid rules engine   │
                    │   -> shortlist + checklist │
                    │  Phase 2: open follow-up   │
                    │   -> RAG retrieval (DB)    │
                    │   -> LLM (grounded prompt) │
                    │   -> satisfaction check    │
                    │      (bounded loop)        │
                    │                            │
                    │  Every claim logged to     │
                    │  claim_graph_edges          │
                    └────────────────────────────┘
                               │
                    GET /admin/graph/{phone}      → JSON provenance graph
                    GET /admin/graph/{phone}/mermaid → Mermaid diagram
```

## Data sources / scraper

`app/scraper/myscheme.py` targets the REST API the myscheme.gov.in frontend
itself calls (`api.myscheme.gov.in/search/v5/...` and
`/schemes/v5/public/schemes/{slug}`). This is the real backend — myscheme is
a Next.js SPA with no server-rendered scheme data, so scraping the HTML page
directly returns nothing useful.

**Important:** `api.myscheme.gov.in` is not reachable from this development
sandbox (network egress allowlist). The scraper code is written against the
documented response shape (per open-source projects that scrape the same
API, e.g. OpenNyAI/schemes_chatbot) but **must be run/tested from an
environment that can reach the host** — your local machine, Render, GitHub
Actions, etc. Every field extraction uses `_safe_get()` so a shape change in
one field degrades gracefully (logs a warning) instead of crashing the sync.

`app/scraper/local_seed.py` provides the same interface reading from
`data/schemes_seed.json` (9 curated schemes, ported from v1) — this is what
ships by default and what the tests run against, so the system works
out-of-the-box without network access.

To add more sources (data.gov.in, state portals): write a module exposing
`scrape_all(keyword="", limit=None)` yielding the canonical dict shape (see
`normalize_scheme()` in `myscheme.py`), and register it in
`app/scraper/registry.py`.

## TTL / expiry

Each `Scheme` and `SchemeChunk` row has `scraped_at`, `valid_until` (the
scheme's own deadline, if any), and `expires_at` = `valid_until + 5 days`
(`DEFAULT_GRACE_DAYS` in `myscheme.py`). `scripts/sync_schemes.py` calls
`purge_expired()` at the end of every run, deleting schemes/chunks past
`expires_at`. Schemes with no deadline (`valid_until=None`) never auto-expire
— most welfare schemes are open-ended.

```bash
python scripts/sync_schemes.py              # full sync (all sources)
python scripts/sync_schemes.py --source local_seed   # one source
python scripts/sync_schemes.py --purge-only # just run expiry cleanup
python scripts/sync_schemes.py --limit 20   # cap records (testing)
```

Run this on a schedule (e.g. daily) via cron, Render Cron Jobs, or GitHub
Actions `schedule:`.

## Hybrid eligibility rules engine

`app/rules/engine.py`. Two rule types per scheme, ALL must pass:

- **`rule_type="json"`** — nested AND/OR/NOT condition trees over fields in
  the user dict (`income <= 250000`, `gender == "female"`, `occupation in
  [...]`, etc.) Handles most regulatory logic.

- **`rule_type="python"`** — a single Python expression, restricted via AST
  whitelist (no imports, no arbitrary attribute access beyond `.get()`, only
  `len/min/max/abs/any/all` builtins). Escape hatch for cross-field logic
  JSON trees can't express cleanly, e.g.:
  ```python
  any(c in user.get('special_categories', []) for c in ['BPL','SC_ST']) \
      or user.get('is_widow', False)
  ```
  This is validated by `_validate_python_rule()` — try `eval_python_rule`
  with e.g. `__import__('os').system(...)` and it's rejected before
  execution.

If a scheme has **zero rules**, `check_scheme_eligibility` returns
`eligible=False` with reason `"no eligibility rules available for this
scheme yet"` — the engine never silently assumes eligibility.

For freshly-scraped schemes (no structured eligibility dict), a conservative
heuristic extractor (`_generate_rules_from_text`) pulls obvious signals
(income caps, age, gender, occupation keywords) from the eligibility text,
tagged `priority=10` and flagged for manual review. Seed data uses the
precise structured generator (`_generate_rules_from_structured`).

## Multi-turn satisfaction loop ("chat doesn't end until satisfied")

After the 6-turn intake produces a shortlist, the conversation enters an
open follow-up loop (`app/conversation.py::handle_followup`):

1. User asks a free-text question
2. RAG retrieves relevant chunks from the DB, builds a grounded prompt, LLM answers
3. Bot asks: *"Did this answer your question? (1=Yes 2=No 3=Talk to a person)"*
4. **Yes** → conversation ends, marked `satisfied=True`
5. **No** → loop back to step 1 (rephrase/clarify)
6. **"talk to a person" / "csc"** → immediate handoff with CSC contact info
7. After `MAX_FOLLOWUP_TURNS` (default 5) unsuccessful rounds → automatic
   handoff to CSC, conversation ends as `handed_off=True`

This is intentionally **bounded**, not a true infinite loop — an unbounded
loop on WhatsApp/SMS would violate the <3s/2G latency goal and could trap
users in unproductive cycles. The cap + human handoff is the safety valve.

## Claim graph (anti-hallucination proof)

`app/graph/claim_graph.py`. Every claim — "user is eligible for X", "document
Y is required", or a free-text RAG answer — is logged as a `ClaimGraphEdge`
row linking:

```
claim_text --> scheme_id --> (rule_id | chunk_id) --> source_url + scraped_at
```

- `GET /admin/graph/{phone}` returns the full graph as JSON (nodes + links)
  plus a `grounding_score` (`grounded_claims / total_claims`). A score < 1.0
  means at least one claim in that conversation had no traceable source —
  this is the audit signal for hallucination.
- `GET /admin/graph/{phone}/mermaid` returns the same graph as a Mermaid
  flowchart (paste into https://mermaid.live or any Mermaid renderer):
  claims (blue) -> schemes (green) / rules (yellow) / chunks (purple) ->
  sources (grey), with an explicit red "UNGROUNDED" node for any
  unsupported claim.

## Setup & local testing

```bash
pip install -r requirements.txt

# Seed the DB from local data (works offline)
python scripts/sync_schemes.py --source local_seed

# Run the API
uvicorn app.main:app --reload --port 8000
```

Walk the conversation:
```bash
P="+919999999999"
curl -s -X POST http://localhost:8000/test/chat -d "{\"phone\":\"$P\",\"message\":\"hi\"}" -H "Content-Type: application/json"
curl -s -X POST http://localhost:8000/test/chat -d "{\"phone\":\"$P\",\"message\":\"3\"}" -H "Content-Type: application/json"   # English
curl -s -X POST http://localhost:8000/test/chat -d "{\"phone\":\"$P\",\"message\":\"1\"}" -H "Content-Type: application/json"   # Farmer
curl -s -X POST http://localhost:8000/test/chat -d "{\"phone\":\"$P\",\"message\":\"45 female\"}" -H "Content-Type: application/json"
curl -s -X POST http://localhost:8000/test/chat -d "{\"phone\":\"$P\",\"message\":\"1\"}" -H "Content-Type: application/json"   # income
curl -s -X POST http://localhost:8000/test/chat -d "{\"phone\":\"$P\",\"message\":\"yes, no\"}" -H "Content-Type: application/json"  # land/house
curl -s -X POST http://localhost:8000/test/chat -d "{\"phone\":\"$P\",\"message\":\"1 2 3\"}" -H "Content-Type: application/json"    # special categories -> shortlist
```

Then inspect provenance:
```bash
curl -s http://localhost:8000/admin/graph/$P | python3 -m json.tool
curl -s http://localhost:8000/admin/graph/$P/mermaid
```

## Enabling live scraping

1. Run from an environment that can reach `api.myscheme.gov.in` (not this sandbox).
2. `python scripts/sync_schemes.py --source myscheme.gov.in --limit 50` (test with a small limit first).
3. Inspect a few resulting `Scheme`/`Rule` rows — the API response shape may
   need small adjustments to `normalize_scheme()` field paths in
   `app/scraper/myscheme.py` (everything is wrapped in `_safe_get` so this is
   a tuning exercise, not a rewrite).
4. Schedule the full sync (no `--limit`) via cron/Render Cron Job.

## Enabling LLM + embeddings

```bash
pip install sentence-transformers groq
export GROQ_API_KEY=...
```
Then implement `call_llm()` in `app/main.py` (template included in comments).
Without `sentence-transformers`, `app/rag/embeddings.py` falls back to a
deterministic hashing-based embedding — functional for testing, lower
retrieval quality than the real multilingual model.

## Database

Default: SQLite (`welfare.db`). For Postgres, set:
```bash
export DATABASE_URL="postgresql://user:pass@host/dbname"
```
SQLAlchemy handles the rest — no code changes needed (see `app/db.py`).
