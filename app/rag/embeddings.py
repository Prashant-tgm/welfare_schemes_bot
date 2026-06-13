"""
Embeddings
==========
Primary: multilingual sentence-transformers (good for Hindi/Tamil/English).
Fallback: a simple hashing-based bag-of-words vector if sentence-transformers
isn't installed — keeps the pipeline runnable for testing without heavy deps,
though retrieval quality will be much lower. Always prefer installing
requirements-rag.txt for production.
"""
import hashlib
import re
import math

_model = None
EMBED_DIM_FALLBACK = 256


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except ImportError:
            _model = "fallback"
    return _model


def embed_text(text: str) -> list:
    model = _get_model()
    if model == "fallback":
        return _fallback_embed(text)
    return model.encode(text, normalize_embeddings=True).tolist()


def _fallback_embed(text: str, dim: int = EMBED_DIM_FALLBACK) -> list:
    """
    Deterministic hashing-based embedding (feature hashing of word tokens).
    Not semantically rich, but allows cosine-similarity retrieval to function
    for testing/demo without sentence-transformers/torch installed.
    """
    vec = [0.0] * dim
    tokens = re.findall(r"\w+", text.lower())
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)
