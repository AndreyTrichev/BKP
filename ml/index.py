from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from ml.chunker import split_to_chunks
from ml.config import COLLECTION_NAME
from ml.preambles_loader import load_l1, load_l2
from ml.reranker import compact_text
from ml.singletons import get_embedder, get_qdrant


def _build_cards() -> list[dict]:
    cards: list[dict] = []

                                              
    cards.extend(load_l1())

                                  
    chunk_idx = 0
    for entry in load_l2():
        meta = entry["metadata"]
        text = entry["text"]
        for chunk in split_to_chunks(text):
            cards.append({
                "text": chunk,
                "metadata": {
                    "level": meta["level"],
                    "department": meta["department"],
                    "subdepartment": meta["subdepartment"],
                    "chunk_idx": chunk_idx,
                },
            })
            chunk_idx += 1

    return cards


def create_collection(recreate: bool = False) -> None:
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

    cards = _build_cards()
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
                    "chunk_idx": meta.get("chunk_idx"),
                },
            )
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)
