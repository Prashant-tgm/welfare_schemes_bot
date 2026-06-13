"""
Conversation Engine
====================
Two phases:

1. STRUCTURED INTAKE (6 questions, same as v1) — fast path to a first
   personalized shortlist using the hybrid rules engine against ALL synced
   schemes (not a hardcoded shortlist).

2. OPEN FOLLOW-UP LOOP — after the shortlist, the user can ask free-text
   questions (RAG-grounded) and the bot asks after each answer:
   "Did this answer your question? (yes/no/talk to a person)"
   - If "no", the bot tries again / asks a clarifying question.
   - Capped at MAX_FOLLOWUP_TURNS to respect bandwidth + avoid unbounded
     loops; after the cap, the bot offers a CSC/human handoff and ends
     gracefully rather than looping forever.
   - "talk to a person" or repeated "no" beyond the cap -> handed_off=True,
     conversation ends with contact info, NOT via a true infinite loop.

Every claim made in phase 1 or 2 is logged via app.graph.claim_graph for
provenance/audit (see /admin/graph/{conversation_id}).
"""
from enum import Enum
import datetime as dt

from app.db import Scheme, Rule, ConversationState
from app.rules.engine import check_scheme_eligibility
from app.rag.retriever import build_grounded_prompt
from app.graph import claim_graph
from app.i18n.translate import LANG_CODES, t

MAX_FOLLOWUP_TURNS = 5
EXIT_PHRASES_HUMAN = {"human", "agent", "talk to a person", "3", "csc"}


class State(Enum):
    LANG_SELECT = "lang_select"
    OCCUPATION = "occupation"
    AGE_GENDER = "age_gender"
    INCOME = "income"
    LAND_HOUSE = "land_house"
    SPECIAL = "special"
    RESULTS = "results"
    FOLLOWUP = "followup"
    SATISFACTION_CHECK = "satisfaction_check"
    DONE = "done"


OCCUPATION_MAP = {"1": "farmer", "2": "laborer", "3": "self_employed", "4": "unemployed", "5": "other"}
INCOME_MAP = {"1": 80000, "2": 180000, "3": 350000}


class Session:
    def __init__(self, phone):
        self.phone = phone
        self.state = State.LANG_SELECT
        self.lang = "eng_Latn"
        self.user = {"special_categories": []}
        self.turn_count = 0
        self.satisfied = False
        self.handed_off = False

    @classmethod
    def from_db(cls, row):
        s = cls(row.phone)
        if row.state:
            s.state = State(row.state)
        s.lang = row.lang or "eng_Latn"
        s.user = row.user_json or {"special_categories": []}
        s.turn_count = row.turn_count or 0
        s.satisfied = bool(row.satisfied)
        s.handed_off = bool(row.handed_off)
        return s

    def to_db(self, row):
        row.state = self.state.value
        row.lang = self.lang
        row.user_json = self.user
        row.turn_count = self.turn_count
        row.satisfied = self.satisfied
        row.handed_off = self.handed_off
        row.updated_at = dt.datetime.utcnow()


def _is_yes(token):
    t_ = token.lower().strip()
    return t_ in ("yes", "y", "1", "हाँ", "ஆம்") or "yes" in t_


# ---------------------------------------------------------------------------
# Eligibility shortlist using ALL synced schemes + hybrid rules engine
# ---------------------------------------------------------------------------
def compute_shortlist(db, user):
    """
    Returns list of dicts: {scheme, eligible, reasons, rule_ids} for every
    scheme currently in the DB (post-purge, so only non-expired schemes).
    """
    schemes = db.query(Scheme).all()
    results = []
    for scheme in schemes:
        rules = db.query(Rule).filter(Rule.scheme_id == scheme.id).all()
        eligible, reasons, rule_ids = check_scheme_eligibility(rules, user)
        results.append({
            "scheme": scheme,
            "eligible": eligible,
            "reasons": reasons,
            "rule_ids": rule_ids,
        })
    return results


def build_results_message(db, session):
    results = compute_shortlist(db, session.user)
    eligible = [r for r in results if r["eligible"]]
    lang = session.lang
    conv_id = session.phone

    if not eligible:
        claim_graph.log_ungrounded_claim(
            db, conv_id, session.turn_count,
            "No schemes matched user profile (or rules engine could not assert eligibility for any scheme)."
        )
        return t("no_match", lang)

    lines = [t("shortlist_header", lang)]
    for r in eligible:
        scheme = r["scheme"]
        lines.append(f"\n- {scheme.name}\n{scheme.benefits}\n(Source: {scheme.source_url})")

        for rule_id in r["rule_ids"]:
            rule = db.query(Rule).get(rule_id) if rule_id else None
            claim_graph.log_claim(
                db, conv_id, session.turn_count,
                claim_text=f"User is eligible for '{scheme.name}'",
                scheme_id=scheme.id,
                rule_id=rule_id,
                source_url=(rule.source_url if rule else scheme.source_url),
                source_scraped_at=scheme.scraped_at,
            )

    lines.append("\n\n" + t("checklist_header", lang))
    seen_docs = set()
    for r in eligible:
        scheme = r["scheme"]
        for d in (scheme.documents_required or []):
            if d not in seen_docs:
                lines.append(f"[ ] {d}")
                seen_docs.add(d)
                claim_graph.log_claim(
                    db, conv_id, session.turn_count,
                    claim_text=f"Document required: '{d}' (for {scheme.name})",
                    scheme_id=scheme.id,
                    source_url=scheme.source_url,
                    source_scraped_at=scheme.scraped_at,
                )

    lines.append("\n\nAsk me any question about these schemes, or type 'done' to finish.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 1: structured intake (same shape as v1)
# ---------------------------------------------------------------------------
def handle_intake(db, session, text):
    text = text.strip()

    if session.state == State.LANG_SELECT:
        session.lang = LANG_CODES.get(text.lower(), "eng_Latn")
        session.state = State.OCCUPATION
        return t("ask_occupation", session.lang)

    elif session.state == State.OCCUPATION:
        choice = text.split()[0] if text else "5"
        session.user["occupation"] = OCCUPATION_MAP.get(choice, "other")
        session.state = State.AGE_GENDER
        return t("ask_age_gender", session.lang)

    elif session.state == State.AGE_GENDER:
        parts = text.lower().split()
        age = next((int(p) for p in parts if p.isdigit()), 30)
        gender = "female" if any(g in text.lower() for g in ["female", "महिला", "பெண்"]) else "male"
        session.user["age"] = age
        session.user["gender"] = gender
        session.state = State.INCOME
        return t("ask_income", session.lang)

    elif session.state == State.INCOME:
        choice = text.split()[0] if text else "1"
        session.user["annual_income"] = INCOME_MAP.get(choice, 80000)
        session.state = State.LAND_HOUSE
        return t("ask_land_house", session.lang)

    elif session.state == State.LAND_HOUSE:
        parts = [p.strip() for p in text.split(",")]
        session.user["land_owner"] = _is_yes(parts[0]) if len(parts) > 0 else False
        session.user["has_pucca_house"] = _is_yes(parts[1]) if len(parts) > 1 else False
        session.user["residence"] = "rural"
        session.state = State.SPECIAL
        return t("ask_special", session.lang)

    elif session.state == State.SPECIAL:
        selections = [s.strip() for s in text.replace(",", " ").split()]
        cats = []
        for s in selections:
            if s == "1":
                cats.append("BPL")
            elif s == "2":
                session.user["is_widow"] = True
            elif s == "3":
                session.user["has_girl_child_under10"] = True
            elif s == "4":
                session.user["is_street_vendor"] = True
        session.user["special_categories"] = cats
        session.state = State.RESULTS
        return build_results_message(db, session)

    return t("welcome", session.lang)


# ---------------------------------------------------------------------------
# Phase 2: open follow-up loop with satisfaction check
# ---------------------------------------------------------------------------
def handle_followup(db, session, text, call_llm):
    lang = session.lang
    conv_id = session.phone

    if session.state == State.RESULTS:
        if text.strip().lower() in ("done", "बस", "முடிந்தது"):
            session.state = State.DONE
            session.satisfied = True
            return t("goodbye", lang)
        session.state = State.FOLLOWUP
        # fall through to FOLLOWUP handling for this same message

    if session.state == State.FOLLOWUP:
        if text.strip().lower() in ("done", "बस", "முடிந்தது"):
            session.state = State.DONE
            session.satisfied = True
            return t("goodbye", lang)

        prompt, retrieved_chunks = build_grounded_prompt(db, text)
        answer = call_llm(prompt)

        if retrieved_chunks:
            for score, chunk in retrieved_chunks:
                claim_graph.log_claim(
                    db, conv_id, session.turn_count,
                    claim_text=answer[:200],
                    scheme_id=chunk.scheme_id,
                    chunk_id=chunk.id,
                    source_url=chunk.source_url,
                    source_scraped_at=chunk.scraped_at,
                    confidence=float(score),
                )
        else:
            claim_graph.log_ungrounded_claim(db, conv_id, session.turn_count, answer[:200])

        session.turn_count += 1
        session.state = State.SATISFACTION_CHECK
        return answer + "\n\n" + t("satisfaction_check", lang)

    if session.state == State.SATISFACTION_CHECK:
        lowered = text.strip().lower()

        if lowered in EXIT_PHRASES_HUMAN or "person" in lowered or "csc" in lowered:
            session.state = State.DONE
            session.handed_off = True
            return t("handoff", lang)

        if _is_yes(lowered):
            session.state = State.DONE
            session.satisfied = True
            return t("goodbye", lang)

        if session.turn_count >= MAX_FOLLOWUP_TURNS:
            session.state = State.DONE
            session.handed_off = True
            return t("max_turns_handoff", lang)

        session.state = State.FOLLOWUP
        return t("ask_again", lang)

    return t("goodbye", lang)


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------
def handle_message(db, session, text, call_llm):
    if session.state in (
        State.LANG_SELECT, State.OCCUPATION, State.AGE_GENDER,
        State.INCOME, State.LAND_HOUSE, State.SPECIAL,
    ):
        return handle_intake(db, session, text)

    return handle_followup(db, session, text, call_llm)
