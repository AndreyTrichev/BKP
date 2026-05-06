"""Lazy синглтоны для baseline. Изолированы от ml.singletons —
если оба pipeline работают одновременно, у каждого свои инстансы.
"""
from typing import Optional

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import (
    CROSS_ENCODER_MODEL,
    EMBEDDING_MODEL,
    QDRANT_URL,
)

_embedder: Optional[SentenceTransformer] = None
_cross_encoder: Optional[CrossEncoder] = None
_qdrant: Optional[QdrantClient] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=2048)
    return _cross_encoder


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        if QDRANT_URL == ":memory:":
            _qdrant = QdrantClient(location=":memory:")
        else:
            _qdrant = QdrantClient(url=QDRANT_URL)
    return _qdrant
