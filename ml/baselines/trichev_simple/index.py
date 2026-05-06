"""Создание Qdrant-коллекции baseline и заполнение её эмбеддингами.

Использует свой knowledge_base (vendored из baseline-репо) — у baseline
своя 16-subdept схема preambles, отличная от продвинутой 18-subdept.
"""
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from .config import COLLECTION_NAME
from .knowledge_base import get_all_cards
from .reranker import compact_text
from .singletons import get_embedder, get_qdrant


def create_collection(recreate: bool = False) -> None:
    """Идемпотентное создание коллекции `citizen_complaints_baseline`."""
    client = get_qdrant()

    if not recreate and client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        if info.points_count and info.points_count > 0:
            return

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    cards = get_all_cards()
    embedder = get_embedder()
    texts = ["passage: " + card["text"] for card in cards]
    vectors = embedder.encode(texts, normalize_embeddings=True)

    points = []
    for idx, (card, vec) in enumerate(zip(cards, vectors)):
        meta = card["metadata"]
        points.append(
            PointStruct(
                id=idx,
                vector=vec.tolist(),
                payload={
                    "text": card["text"],
                    "ce_text": compact_text(card["text"]),
                    "level": meta["level"],
                    "department": meta["department"],
                    "subdepartment": meta.get("subdepartment"),
                },
            )
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)
