"""Chroma vector database: the knowledge base of news stories and filings.

Embedding backends, best first:
1. nomic-embed-text via Ollama (local, semantic)
2. all-MiniLM-L6-v2 via Chroma's built-in ONNX model (downloads once)
3. a hashing vectorizer (offline, keyword-level) so the demo never breaks
The backend in use is stored on the collection and shown in the UI.
"""
import hashlib
import re

import chromadb
import numpy as np
import requests

from . import config, llm

COLLECTION = "news_filings"
_backend: str | None = None
_onnx = None


def client():
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def _hash_embed(texts: list[str], dim: int = 512) -> list[list[float]]:
    out = []
    for t in texts:
        v = np.zeros(dim)
        words = re.findall(r"[a-z0-9$%]+", t.lower())
        feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % dim] += 1 if (h >> 9) % 2 else -1
        n = np.linalg.norm(v)
        out.append((v / n if n else v).tolist())
    return out


def backend() -> str:
    global _backend, _onnx
    if _backend:
        return _backend
    if llm.status().get("ollama_embed"):
        _backend = f"ollama:{config.OLLAMA_EMBED_MODEL}"
        return _backend
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        _onnx = DefaultEmbeddingFunction()
        _onnx(["warm up"])
        _backend = "onnx:all-MiniLM-L6-v2"
    except Exception:  # noqa: BLE001 - offline: model download blocked
        _backend = "hashing-512"
    return _backend


def embed(texts: list[str]) -> list[list[float]]:
    b = backend()
    if b.startswith("ollama"):
        vecs = []
        for i in range(0, len(texts), 64):
            r = requests.post(f"{config.OLLAMA_URL}/api/embed",
                              json={"model": config.OLLAMA_EMBED_MODEL, "input": texts[i:i + 64]}, timeout=120)
            r.raise_for_status()
            vecs += r.json()["embeddings"]
        return vecs
    if b.startswith("onnx"):
        return [list(map(float, v)) for v in _onnx(texts)]
    return _hash_embed(texts)


def dedup_threshold() -> float:
    """Cosine similarity above which two items are the same story."""
    return 0.72 if backend() == "hashing-512" else 0.86


def fresh_collection():
    c = client()
    try:
        c.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001
        pass
    return c.create_collection(COLLECTION, metadata={"hnsw:space": "cosine", "embedding_backend": backend()})


def collection():
    return client().get_collection(COLLECTION)


def date_num(d: str) -> int:
    return int(d.replace("-", ""))
