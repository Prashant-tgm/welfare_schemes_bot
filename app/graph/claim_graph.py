"""
Claim Graph
===========
Every factual claim the bot makes (eligibility verdict, benefit amount,
document requirement, etc.) is recorded as a ClaimGraphEdge linking:

    claim_text --> scheme_id --> (rule_id | chunk_id) --> source_url + scraped_at

This produces an auditable graph: for any bot statement, you can trace
exactly which DB row, which rule or retrieved chunk, and which scraped
source URL/timestamp justify it. If a claim has no edge, it's flagged as
UNGROUNDED — this is the mechanism that proves (or disproves) "the AI isn't
hallucinating" for a given conversation.

Two output formats:
- to_mermaid(): Mermaid flowchart syntax for visual rendering
- to_json(): structured graph (nodes/edges) for programmatic audit/UI
"""
import datetime as dt
from app.db import ClaimGraphEdge


def log_claim(db, conversation_id, turn_index, claim_text,
               scheme_id=None, chunk_id=None, rule_id=None,
               source_url=None, source_scraped_at=None, confidence=1.0):
    edge = ClaimGraphEdge(
        conversation_id=conversation_id,
        turn_index=turn_index,
        claim_text=claim_text,
        scheme_id=scheme_id,
        chunk_id=chunk_id,
        rule_id=rule_id,
        source_url=source_url,
        source_scraped_at=source_scraped_at,
        confidence=confidence,
    )
    db.add(edge)
    db.commit()
    return edge


def log_ungrounded_claim(db, conversation_id, turn_index, claim_text):
    """Explicitly log a claim with NO supporting source — for audit visibility."""
    return log_claim(db, conversation_id, turn_index, claim_text, confidence=0.0)


def get_conversation_graph(db, conversation_id):
    return db.query(ClaimGraphEdge).filter(
        ClaimGraphEdge.conversation_id == conversation_id
    ).order_by(ClaimGraphEdge.turn_index, ClaimGraphEdge.id).all()


def to_json(edges):
    """Structured node/edge graph for programmatic audit or front-end rendering."""
    nodes = {}
    links = []

    def add_node(node_id, label, kind):
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "label": label, "kind": kind}

    for e in edges:
        claim_id = f"claim_{e.id}"
        add_node(claim_id, e.claim_text[:80], "claim")

        if e.confidence == 0.0:
            add_node("UNGROUNDED", "No source (ungrounded)", "warning")
            links.append({"from": claim_id, "to": "UNGROUNDED", "type": "grounding"})
            continue

        if e.scheme_id:
            scheme_node = f"scheme_{e.scheme_id}"
            add_node(scheme_node, e.scheme_id, "scheme")
            links.append({"from": claim_id, "to": scheme_node, "type": "about"})

        if e.rule_id:
            rule_node = f"rule_{e.rule_id}"
            add_node(rule_node, f"Rule #{e.rule_id}", "rule")
            links.append({"from": claim_id, "to": rule_node, "type": "based_on"})
            if e.scheme_id:
                links.append({"from": rule_node, "to": f"scheme_{e.scheme_id}", "type": "rule_of"})

        if e.chunk_id:
            chunk_node = f"chunk_{e.chunk_id}"
            add_node(chunk_node, f"Chunk #{e.chunk_id}", "chunk")
            links.append({"from": claim_id, "to": chunk_node, "type": "retrieved_from"})

        if e.source_url:
            source_node = f"source_{abs(hash(e.source_url)) % 100000}"
            ts = e.source_scraped_at.isoformat() if e.source_scraped_at else "unknown"
            add_node(source_node, f"{e.source_url} (scraped {ts})", "source")
            anchor = f"rule_{e.rule_id}" if e.rule_id else (f"chunk_{e.chunk_id}" if e.chunk_id else claim_id)
            links.append({"from": anchor, "to": source_node, "type": "sourced_from"})

    return {"nodes": list(nodes.values()), "links": links}


def to_mermaid(edges):
    """Render the claim graph as a Mermaid flowchart for inline visualization."""
    graph = to_json(edges)
    lines = ["flowchart LR"]

    style_map = {
        "claim": "fill:#e0f2fe,stroke:#0284c7",
        "scheme": "fill:#dcfce7,stroke:#16a34a",
        "rule": "fill:#fef9c3,stroke:#ca8a04",
        "chunk": "fill:#f3e8ff,stroke:#9333ea",
        "source": "fill:#f1f5f9,stroke:#64748b",
        "warning": "fill:#fee2e2,stroke:#dc2626",
    }

    for node in graph["nodes"]:
        safe_label = node["label"].replace('"', "'").replace("\n", " ")
        lines.append(f'    {node["id"]}["{safe_label}"]')
        lines.append(f'    style {node["id"]} {style_map.get(node["kind"], "")}')

    for link in graph["links"]:
        if link["to"] == "UNGROUNDED":
            arrow = "-.->|UNGROUNDED|"
        else:
            arrow = f"-->|{link['type']}|"
        lines.append(f'    {link["from"]} {arrow} {link["to"]}')

    return "\n".join(lines)


def conversation_grounding_score(edges):
    """Summary metric: what fraction of claims in this conversation are grounded?"""
    total = len(edges)
    ungrounded = sum(1 for e in edges if e.confidence == 0.0)
    return {
        "total_claims": total,
        "grounded_claims": total - ungrounded,
        "ungrounded_claims": ungrounded,
        "grounding_rate": (total - ungrounded) / total if total else 1.0,
    }
