"""
RAG Retriever (DB-backed)
==========================
Retrieves SchemeChunk rows by cosine similarity against stored embeddings.
Brute-force similarity search — fine for hundreds/low-thousands of chunks
(a few hundred schemes x ~5 chunks each). For larger corpora, swap in
sqlite-vec, FAISS, or a Postgres+pgvector index behind the same interface.
"""
from app.db import SchemeChunk
from app.rag.embeddings import embed_text, cosine_similarity


def retrieve(db, query: str, k: int = 5, scheme_id: str = None):
    """Returns top-k SchemeChunk rows by similarity to query, with scores."""
    q_emb = embed_text(query)

    qry = db.query(SchemeChunk)
    if scheme_id:
        qry = qry.filter(SchemeChunk.scheme_id == scheme_id)

    scored = []
    for chunk in qry.all():
        if not chunk.embedding:
            continue
        score = cosine_similarity(q_emb, chunk.embedding)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


SYSTEM_PROMPT = """You are a helpful assistant for India's welfare scheme information, \
answering for rural users in their own language. STRICT RULES:
1. ONLY use information from the CONTEXT provided below. Never use outside knowledge \
about eligibility, benefit amounts, or documents.
2. ALWAYS mention the scheme name for every claim you make.
3. If the CONTEXT does not contain the answer, say clearly: \
"I don't have verified information on this — please check with your local Common \
Service Centre (CSC) or the official portal." Do NOT guess or invent eligibility \
rules, amounts, or document lists.
4. Keep answers SHORT (under 60 words) — this is for WhatsApp/SMS on a low-end phone.
5. Respond in the same language as the user's question.

CONTEXT:
{context}

USER QUESTION: {question}
"""


def build_grounded_prompt(db, query: str, k: int = 5, scheme_id: str = None):
    results = retrieve(db, query, k=k, scheme_id=scheme_id)
    context = "\n\n---\n\n".join(c.text for _, c in results)
    prompt = SYSTEM_PROMPT.format(context=context, question=query)
    return prompt, results  # results returned for claim-graph logging
