"""
Test suite — run with: pytest tests/

Covers:
- Hybrid rules engine (JSON trees, python escape hatch, security validator)
- Eligibility shortlist correctness for known personas
- Conversation flow (intake -> results -> followup -> satisfaction loop -> handoff)
- Claim graph provenance + grounding score
- TTL/expiry purge
"""
import os
import sys
import datetime as dt
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db import SessionLocal, Scheme, purge_expired, init_db
from app.rules.engine import eval_json_rule, eval_python_rule, check_scheme_eligibility, _validate_python_rule, generate_default_rules
from app.scraper import local_seed
from app.conversation import Session, State, handle_message, compute_shortlist, build_results_message
from app.graph import claim_graph


@pytest.fixture(scope="module")
def db():
    init_db()
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture(scope="module", autouse=True)
def seed_data(db):
    for record in local_seed.scrape_all():
        existing = db.query(Scheme).get(record["id"])
        if existing:
            continue
        scheme = Scheme(id=record["id"])
        db.add(scheme)
        scheme.name = record["name"]
        scheme.category = record.get("category")
        scheme.level = record.get("level")
        scheme.description = record.get("description", "")
        scheme.benefits = record.get("benefits", "")
        scheme.application_process = record.get("application_process", "")
        scheme.documents_required = record.get("documents_required", [])
        scheme.source_url = record.get("source_url")
        scheme.scraped_at = dt.datetime.utcnow()
        scheme.raw_hash = record.get("raw_hash")
        generate_default_rules(db, scheme, record)
        db.commit()
    yield


# ---------------------------------------------------------------------------
# Rules engine
# ---------------------------------------------------------------------------
def test_json_rule_and():
    rule = {"op": "and", "conditions": [
        {"field": "age", "op": ">=", "value": 18},
        {"field": "occupation", "op": "==", "value": "farmer"},
    ]}
    ok, reasons = eval_json_rule(rule, {"age": 20, "occupation": "farmer"})
    assert ok and not reasons

    ok, reasons = eval_json_rule(rule, {"age": 15, "occupation": "farmer"})
    assert not ok and len(reasons) == 1


def test_json_rule_or_nested():
    rule = {"op": "and", "conditions": [
        {"field": "annual_income", "op": "<=", "value": 250000},
        {"op": "or", "conditions": [
            {"field": "state", "op": "==", "value": "Bihar"},
            {"field": "state", "op": "==", "value": "UP"},
        ]},
    ]}
    ok, _ = eval_json_rule(rule, {"annual_income": 100000, "state": "Bihar"})
    assert ok
    ok, _ = eval_json_rule(rule, {"annual_income": 100000, "state": "Kerala"})
    assert not ok


def test_python_rule_genexp():
    code = "any(c in user.get('special_categories', []) for c in ['BPL','SC_ST']) or user.get('is_widow', False)"
    ok, _ = eval_python_rule(code, {"special_categories": ["BPL"]})
    assert ok
    ok, _ = eval_python_rule(code, {"special_categories": [], "is_widow": True})
    assert ok
    ok, _ = eval_python_rule(code, {"special_categories": [], "is_widow": False})
    assert not ok


def test_python_rule_rejects_malicious():
    malicious_codes = [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "[].__class__.__bases__[0]",
        "user.__class__",
    ]
    for code in malicious_codes:
        with pytest.raises(ValueError):
            _validate_python_rule(code)


def test_no_rules_fails_closed():
    ok, reasons, ids = check_scheme_eligibility([], {"age": 30})
    assert not ok
    assert "no eligibility rules available" in reasons[0]


# ---------------------------------------------------------------------------
# Eligibility shortlist — known personas
# ---------------------------------------------------------------------------
def test_shortlist_bpl_widow_farmer(db):
    user = {
        "special_categories": ["BPL"], "occupation": "farmer", "age": 45,
        "gender": "female", "annual_income": 80000, "land_owner": True,
        "has_pucca_house": False, "residence": "rural",
        "is_widow": True, "has_girl_child_under10": True,
    }
    results = compute_shortlist(db, user)
    eligible_ids = {r["scheme"].id for r in results if r["eligible"]}

    for expected in ["pm-kisan", "ayushman-bharat", "pmay-g", "nrega", "sukanya-samriddhi", "pmfby", "nsap-widow"]:
        assert expected in eligible_ids, f"expected {expected} in {eligible_ids}"

    assert "pm-svanidhi" not in eligible_ids


def test_shortlist_high_income_laborer_no_land(db):
    user = {
        "special_categories": [], "occupation": "laborer", "age": 30,
        "gender": "male", "annual_income": 350000, "land_owner": False,
        "has_pucca_house": False, "residence": "rural",
    }
    results = compute_shortlist(db, user)
    eligible_ids = {r["scheme"].id for r in results if r["eligible"]}

    assert "nrega" in eligible_ids
    assert "ayushman-bharat" not in eligible_ids
    assert "pmay-g" not in eligible_ids
    assert "pm-kisan" not in eligible_ids


# ---------------------------------------------------------------------------
# Conversation flow
# ---------------------------------------------------------------------------
def test_full_conversation_flow_and_handoff(db):
    def fake_llm(prompt):
        return "Apply via the official portal or your nearest CSC."

    session = Session("+91test1")
    flow = ["3", "1", "45 female", "1", "yes, no", "1 2 3"]
    reply = None
    for msg in flow:
        reply = handle_message(db, session, msg, fake_llm)

    assert session.state == State.RESULTS
    assert "Schemes you may be eligible for" in reply

    reply = handle_message(db, session, "How do I apply?", fake_llm)
    assert session.state == State.SATISFACTION_CHECK
    assert "Did this answer your question" in reply

    for _ in range(10):
        if session.state == State.DONE:
            break
        reply = handle_message(db, session, "2", fake_llm)
        if session.state == State.FOLLOWUP:
            reply = handle_message(db, session, "still confused", fake_llm)

    assert session.state == State.DONE
    assert session.handed_off is True


def test_satisfaction_yes_ends_conversation(db):
    def fake_llm(prompt):
        return "Some grounded answer."

    session = Session("+91test2")
    flow = ["3", "1", "45 female", "1", "yes, no", "1 2 3"]
    for msg in flow:
        handle_message(db, session, msg, fake_llm)

    handle_message(db, session, "tell me more", fake_llm)
    assert session.state == State.SATISFACTION_CHECK

    handle_message(db, session, "1", fake_llm)
    assert session.state == State.DONE
    assert session.satisfied is True


# ---------------------------------------------------------------------------
# Claim graph
# ---------------------------------------------------------------------------
def test_claim_graph_grounded(db):
    user = {
        "special_categories": ["BPL"], "occupation": "farmer", "age": 45,
        "gender": "female", "annual_income": 80000, "land_owner": True,
        "has_pucca_house": False, "residence": "rural",
        "is_widow": True, "has_girl_child_under10": True,
    }
    session = Session("+91test3")
    session.user = user
    build_results_message(db, session)

    edges = claim_graph.get_conversation_graph(db, "+91test3")
    assert len(edges) > 0
    score = claim_graph.conversation_grounding_score(edges)
    assert score["grounding_rate"] == 1.0

    mermaid = claim_graph.to_mermaid(edges)
    assert "flowchart LR" in mermaid
    assert "UNGROUNDED" not in mermaid


def test_claim_graph_ungrounded_logged(db):
    session_id = "+91test4"
    claim_graph.log_ungrounded_claim(db, session_id, 0, "Unverifiable claim with no source")
    edges = claim_graph.get_conversation_graph(db, session_id)
    score = claim_graph.conversation_grounding_score(edges)
    assert score["ungrounded_claims"] == 1
    mermaid = claim_graph.to_mermaid(edges)
    assert "UNGROUNDED" in mermaid


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------
def test_expiry_purge(db):
    expired = Scheme(id="test-expired-scheme", name="Expired", expires_at=dt.datetime.utcnow() - dt.timedelta(days=1))
    not_expired = Scheme(id="test-active-scheme", name="Active", expires_at=dt.datetime.utcnow() + dt.timedelta(days=10))
    no_expiry = Scheme(id="test-no-expiry-scheme", name="NoExpiry", expires_at=None)
    db.add_all([expired, not_expired, no_expiry])
    db.commit()

    n_schemes, _ = purge_expired(db)
    assert n_schemes >= 1

    remaining_ids = {s.id for s in db.query(Scheme).all()}
    assert "test-expired-scheme" not in remaining_ids
    assert "test-active-scheme" in remaining_ids
    assert "test-no-expiry-scheme" in remaining_ids
