from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

from ml.config import (
    BEAM_L2_LIMIT,
    BEAM_MIN,
    COLLECTION_NAME,
    REL_PRUNE,
    W_L1,
    W_L2,
)
from ml.singletons import get_embedder, get_qdrant


def beam_retrieve(text: str):
    client = get_qdrant()
    embedder = get_embedder()

    query_vector = embedder.encode(
        "query: " + text, normalize_embeddings=True
    ).tolist()

                                                                      
    l1_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="level", match=MatchValue(value=1))]
        ),
        limit=4,
    )

    l1_by_dept = {hit.payload["department"]: hit for hit in l1_result.points}
    if not l1_by_dept:
        return {}, [], []

                                                                      
    l1_scores = {d: h.score for d, h in l1_by_dept.items()}
    best_l1 = max(l1_scores.values())
    threshold = REL_PRUNE * best_l1

    beam_depts = sorted(
        [d for d, s in l1_scores.items() if s >= threshold],
        key=lambda d: l1_scores[d],
        reverse=True,
    )
    if len(beam_depts) < BEAM_MIN:
        sorted_d = sorted(l1_scores.items(), key=lambda x: x[1], reverse=True)
        beam_depts = [d for d, _ in sorted_d[:BEAM_MIN]]

                                                                      
    if len(beam_depts) >= len(l1_by_dept):
                                             
        l2_filter = Filter(
            must=[FieldCondition(key="level", match=MatchValue(value=2))]
        )
    else:
        l2_filter = Filter(
            must=[
                FieldCondition(key="level", match=MatchValue(value=2)),
                FieldCondition(key="department", match=MatchAny(any=beam_depts)),
            ]
        )

    l2_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=l2_filter,
        limit=BEAM_L2_LIMIT,
    )

                                                                      
    l2_with_path = []
    for hit in l2_result.points:
        dept = hit.payload["department"]
        l1_score = l1_scores.get(dept, 0.0)
        path_score = W_L1 * l1_score + W_L2 * hit.score
        l2_with_path.append((path_score, hit))

    l2_with_path.sort(key=lambda x: x[0], reverse=True)
    return l1_by_dept, beam_depts, l2_with_path
