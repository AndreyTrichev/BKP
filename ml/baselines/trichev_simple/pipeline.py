"""Оркестратор baseline: текст → ml.schemas.RouteResult.

Алгоритм (проще чем продвинутый):
  1. encode_query
  2. ANN L2 top-K (без beam, без фильтра по dept)
  3. CE rerank top-CE_CANDIDATES_LIMIT (с diversify per dept)
  4. MaxP-per-dept + softmax
  5. confidence по 2 порогам (без RAW_FLOOR/GAP_MIN)
  6. subdept = top из winning dept (softmax внутри dept)

Возвращает ml.schemas.RouteResult напрямую — совместим с продвинутым
pipeline'ом, не нужен адаптер.
"""
from ml.schemas import RouteResult

from .confidence import determine_confidence
from .reranker import ce_rerank, softmax
from .retriever import (
    ann_l2,
    encode_query,
    filter_l2_by_dept,
    query_l1,
)
from .singletons import get_cross_encoder, get_qdrant
from .config import COLLECTION_NAME


_EMPTY_RESULT = RouteResult(
    department="unknown",
    subdepartment=None,
    confidence="low",
    score_dept=0.0,
    score_subdept=0.0,
    fallback=True,
    raw_score=0.0,
)


def _aggregate_departments(l2_scored: list, l1_by_dept: dict):
    """MaxP-per-dept: для каждого dept берём максимальный CE-скор,
    softmax по dept-уровню. Возвращает [(softmax, l1_hit, raw), ...]."""
    dept_best_raw: dict = {}
    for raw, hit in l2_scored:
        dept = hit.payload["department"]
        if dept not in dept_best_raw or raw > dept_best_raw[dept]:
            dept_best_raw[dept] = float(raw)

    depts_sorted = sorted(dept_best_raw.items(), key=lambda x: x[1], reverse=True)
    dept_names = [d for d, _ in depts_sorted]
    dept_raws = [s for _, s in depts_sorted]
    if not dept_raws:
        return []

    dept_softmax = softmax(dept_raws).tolist()

    ranked = []
    for dept, score in zip(dept_names, dept_softmax):
        if dept in l1_by_dept:
            ranked.append(
                (float(score), l1_by_dept[dept], float(dept_best_raw[dept]))
            )
    return ranked


def _pick_subdepartment(l2_scored: list, department: str):
    """В winning dept softmax по raw_scores → top subdept."""
    dept_hits = [
        (raw, hit) for raw, hit in l2_scored
        if hit.payload["department"] == department
    ]
    if not dept_hits:
        return []

    raws = [r for r, _ in dept_hits]
    hits = [h for _, h in dept_hits]
    norm_scores = softmax(raws)

    return sorted(zip(norm_scores, hits), key=lambda x: x[0], reverse=True)


def route(text: str) -> RouteResult:
    """Маршрутизация одного обращения. Возвращает ml.schemas.RouteResult."""
    client = get_qdrant()
    if not client.collection_exists(COLLECTION_NAME):
        from .index import create_collection
        create_collection()

    # 1. Embed + ANN L2
    query_vector = encode_query(text)
    l2_hits = ann_l2(query_vector)
    if not l2_hits:
        return _EMPTY_RESULT

    # 2. CE rerank top-N (diversify per dept)
    l2_scored = ce_rerank(text, l2_hits)
    if not l2_scored:
        return _EMPTY_RESULT

    # 3. MaxP-per-dept + L1 lookup для richer payload
    l1_by_dept = query_l1(query_vector)
    dept_ranked = _aggregate_departments(l2_scored, l1_by_dept)
    if not dept_ranked:
        return _EMPTY_RESULT

    # 4. Confidence
    scores_desc = [score for score, _, _ in dept_ranked]
    raw_scores_desc = [raw for _, _, raw in dept_ranked]
    confidence, fallback = determine_confidence(scores_desc, raw_scores_desc)

    winner_score, winner_hit, winner_raw = dept_ranked[0]
    department = winner_hit.payload["department"]

    # 5. Best subdept внутри winning dept
    sub_ranked = _pick_subdepartment(l2_scored, department)
    if sub_ranked:
        sub_score, sub_hit = sub_ranked[0]
        subdepartment = sub_hit.payload["subdepartment"]
    else:
        sub_score = 0.0
        subdepartment = None

    return RouteResult(
        department=department,
        subdepartment=subdepartment,
        confidence=confidence,
        score_dept=float(winner_score),
        score_subdept=float(sub_score),
        fallback=fallback,
        raw_score=float(winner_raw),
    )


def get_routing_context(text: str, top_k: int = 2) -> dict:
    """Возвращает top-K dept с их subdept-чанками.

    Делает дополнительный CE-прогон для subdept (отличие от route()).
    """
    client = get_qdrant()
    if not client.collection_exists(COLLECTION_NAME):
        from .index import create_collection
        create_collection()

    query_vector = encode_query(text)
    l2_hits = ann_l2(query_vector)
    if not l2_hits:
        return {"ranked_departments": []}

    l2_scored = ce_rerank(text, l2_hits)
    if not l2_scored:
        return {"ranked_departments": []}

    l1_by_dept = query_l1(query_vector)
    dept_ranked = _aggregate_departments(l2_scored, l1_by_dept)
    if not dept_ranked:
        return {"ranked_departments": []}

    # Для каждого top-K dept делаем доп. CE-rerank по 10 кандидатам внутри dept
    ce = get_cross_encoder()
    ranked_departments = []
    for score, hit, _ in dept_ranked[:top_k]:
        dept = hit.payload["department"]
        dept_l2_hits = filter_l2_by_dept(query_vector, dept, limit=10)

        if dept_l2_hits:
            pairs = [(text, sh.payload["text"]) for sh in dept_l2_hits]
            raw_scores = ce.predict(pairs)
            norm = softmax(raw_scores)
            sub_ranked = sorted(
                zip(norm, dept_l2_hits), key=lambda x: x[0], reverse=True
            )
            subdepartments = [
                {
                    "subdepartment": sh.payload["subdepartment"],
                    "text": sh.payload["text"],
                    "score": float(ss),
                }
                for ss, sh in sub_ranked
            ]
        else:
            subdepartments = []

        ranked_departments.append({
            "department": dept,
            "score": float(score),
            "text": hit.payload["text"],
            "subdepartments": subdepartments,
        })

    return {"ranked_departments": ranked_departments}
