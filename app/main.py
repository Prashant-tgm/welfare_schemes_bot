"""
FastAPI App (v2)
================
Endpoints:
- POST /webhook/whatsapp    — Twilio WhatsApp webhook
- POST /test/chat           — local testing (no Twilio)
- GET  /admin/graph/{phone} — claim graph (JSON) for a conversation
- GET  /admin/graph/{phone}/mermaid — claim graph as Mermaid diagram source
- GET  /admin/schemes       — list all synced schemes + expiry info
- GET  /health
"""
from fastapi import FastAPI, Request, Response, Depends
from sqlalchemy.orm import Session as DBSession

from app.db import init_db, get_session, ConversationState, Scheme
from app.conversation import Session, State, handle_message
from app.graph import claim_graph
from app.i18n.translate import t

app = FastAPI(title="Welfare Scheme Chatbot v2")


@app.on_event("startup")
def startup():
    init_db()


def call_llm(prompt: str) -> str:
    """
    Stub: replace with Groq/Llama-3 call.

        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        return resp.choices[0].message.content
    """
    return "[LLM not configured — set GROQ_API_KEY and implement call_llm() in app/main.py]"


def _load_session(db: DBSession, phone: str) -> Session:
    row = db.query(ConversationState).get(phone)
    if row:
        return Session.from_db(row)
    return Session(phone)


def _save_session(db: DBSession, session: Session):
    row = db.query(ConversationState).get(session.phone)
    if not row:
        row = ConversationState(phone=session.phone)
        db.add(row)
    session.to_db(row)
    db.commit()


def _process(db: DBSession, phone: str, body: str) -> str:
    if body.strip().lower() in ("hi", "hello", "start", "नमस्ते", "வணக்கம்"):
        session = Session(phone)
        _save_session(db, session)
        return (
            t("welcome", "eng_Latn") + "\n\n" +
            t("welcome", "hin_Deva") + "\n\n" +
            t("welcome", "tam_Taml")
        )

    session = _load_session(db, phone)

    if session.state == State.DONE:
        session.state = State.FOLLOWUP

    reply = handle_message(db, session, body, call_llm)
    _save_session(db, session)
    return reply


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request, db: DBSession = Depends(get_session)):
    form = await request.form()
    phone = form.get("From", "unknown")
    body = form.get("Body", "").strip()

    reply = _process(db, phone, body)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response><Message>{reply}</Message></Response>"""
    return Response(content=xml, media_type="application/xml")


@app.post("/test/chat")
async def test_chat(request: Request, db: DBSession = Depends(get_session)):
    data = await request.json()
    phone = data.get("phone", "test_user")
    body = data.get("message", "")

    reply = _process(db, phone, body)

    row = db.query(ConversationState).get(phone)
    return {
        "reply": reply,
        "state": row.state if row else None,
        "user": row.user_json if row else None,
        "satisfied": row.satisfied if row else None,
        "handed_off": row.handed_off if row else None,
    }


@app.get("/admin/graph/{phone}")
def get_claim_graph(phone: str, db: DBSession = Depends(get_session)):
    edges = claim_graph.get_conversation_graph(db, phone)
    return {
        "graph": claim_graph.to_json(edges),
        "score": claim_graph.conversation_grounding_score(edges),
    }


@app.get("/admin/graph/{phone}/mermaid")
def get_claim_graph_mermaid(phone: str, db: DBSession = Depends(get_session)):
    edges = claim_graph.get_conversation_graph(db, phone)
    return Response(content=claim_graph.to_mermaid(edges), media_type="text/plain")


@app.get("/admin/schemes")
def list_schemes(db: DBSession = Depends(get_session)):
    schemes = db.query(Scheme).all()
    return [s.to_dict() for s in schemes]


@app.get("/health")
def health():
    return {"status": "ok"}
