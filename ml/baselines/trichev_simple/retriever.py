"""ANN-prefilter L2 + запрос L1 для агрегации.

В отличие от продвинутого retriever'а (`ml/retriever.py`):
- НЕТ beam search — все 4 dept в скоупе сразу
- НЕТ adaptive prune по REL_PRUNE
- НЕТ path scoring (W_L1×L1 + W_L2×L2)
- L2-кандидаты идут в CE напрямую, без diversify per subdept
"""
from qdrant_client.models import FieldCondition, Filter, MatchValue

from .config import CE_CANDIDATES_LIMIT, COLLECTION_NAME, L2_PREFILTER_LIMIT
from .singletons import get_embedder, get_qdrant


def encode_query(text: str) -> list:
    """E5: префикс query: + нормализация."""
    embedder = get_embedder()
    return embedder.encode("query: " + text, normalize_embeddings=True).tolist()


def ann_l2(query_vector: list) -> list:
    """Простой ANN top-K по L2 (без фильтра по dept)."""
    client = get_qdrant()
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="level", match=MatchValue(value=2))]
        ),
        limit=max(L2_PREFILTER_LIMIT, CE_CANDIDATES_LIMIT),
    )
    return result.points


def query_l1(query_vector: list) -> dict:
    """ANN по L1 — для дополнения dept-агрегации L1-хитом."""
    client = get_qdrant()
    l1_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="level", match=MatchValue(value=1))]
        ),
        limit=4,
    )
    return {hit.payload["department"]: hit for hit in l1_result.points}


def filter_l2_by_dept(query_vector: list, department: str, limit: int = 10) -> list:
    """L2-чанки внутри конкретного dept (используется в get_routing_context)."""
    client = get_qdrant()
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="level", match=MatchValue(value=2)),
                FieldCondition(key="department", match=MatchValue(value=department)),
            ]
        ),
        limit=limit,
    )
    return result.points
