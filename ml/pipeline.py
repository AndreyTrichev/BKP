from ml.config import COLLECTION_NAME, MAXP_PER_DEPT
from ml.confidence import determine_confidence
from ml.reranker import ce_rerank, softmax
from ml.retriever import beam_retrieve
from ml.schemas import RouteResult
from ml.singletons import get_qdrant


_EMPTY_RESULT = RouteResult(
    department="unknown",
    subdepartment=None,
    confidence="low",
    score_dept=0.0,
    score_subdept=0.0,
    fallback=True,
    raw_score=0.0,
    fallback_reason="no_candidates",
)


def route(text: str) -> RouteResult:
    client = get_qdrant()
    if not client.collection_exists(COLLECTION_NAME):
        from ml.index import create_collection
        create_collection()

                                         
    l1_by_dept, beam_depts, l2_with_path = beam_retrieve(text)
    if not l2_with_path:
        return _EMPTY_RESULT

                                         
    ce_scored = ce_rerank(text, l2_with_path)
    if not ce_scored:
        return _EMPTY_RESULT

    if MAXP_PER_DEPT:
                                                                          
                                                        
        dept_best_ce: dict = {}
        for ce_score, hit in ce_scored:
            dept = hit.payload["department"]
            if dept not in dept_best_ce or ce_score > dept_best_ce[dept]:
                dept_best_ce[dept] = ce_score

        dept_list = sorted(dept_best_ce.items(), key=lambda x: x[1], reverse=True)
        dept_names = [d for d, _ in dept_list]
        dept_raws = [s for _, s in dept_list]
        dept_softmax = softmax(dept_raws).tolist()

                                         
        confidence, fallback, fallback_reason = determine_confidence(dept_softmax, dept_raws)
        department = dept_names[0]
        score_dept = dept_softmax[0]

                                              
        dept_ce_hits = [
            (s, h) for s, h in ce_scored if h.payload["department"] == department
        ]
        if dept_ce_hits:
            subdepartment = dept_ce_hits[0][1].payload["subdepartment"]
            if len(dept_ce_hits) > 1:
                subdept_scores = [s for s, _ in dept_ce_hits]
                subdept_softmax = softmax(subdept_scores).tolist()
                score_subdept = subdept_softmax[0]
            else:
                score_subdept = 1.0
        else:
            subdepartment = None
            score_subdept = 0.0

        top_raw = dept_raws[0]
    else:
                                                                          
                                                
        flat_raws = [s for s, _ in ce_scored]
        flat_softmax = softmax(flat_raws).tolist()

                                                         
        confidence, fallback, fallback_reason = determine_confidence(flat_softmax, flat_raws)

                                                       
        winner_score, winner_hit = ce_scored[0]
        department = winner_hit.payload["department"]
        subdepartment = winner_hit.payload["subdepartment"]
        score_dept = flat_softmax[0]
        score_subdept = flat_softmax[0]                             
        top_raw = winner_score

    return RouteResult(
        department=department,
        subdepartment=subdepartment,
        confidence=confidence,
        score_dept=float(score_dept),
        score_subdept=float(score_subdept),
        fallback=fallback,
        raw_score=float(top_raw),
        fallback_reason=fallback_reason,
    )


def route_debug(text: str) -> dict:
    client = get_qdrant()
    if not client.collection_exists(COLLECTION_NAME):
        from ml.index import create_collection
        create_collection()

    l1_by_dept, beam_depts, l2_with_path = beam_retrieve(text)
    if not l2_with_path:
        return {
            "result": _EMPTY_RESULT,
            "dept_scores": [], "raw_scores": [],
            "gap": 0.0, "top_raw": 0.0,
        }

    ce_scored = ce_rerank(text, l2_with_path)
    if not ce_scored:
        return {
            "result": _EMPTY_RESULT,
            "dept_scores": [], "raw_scores": [],
            "gap": 0.0, "top_raw": 0.0,
        }

    dept_best_ce: dict = {}
    for ce_score, hit in ce_scored:
        dept = hit.payload["department"]
        if dept not in dept_best_ce or ce_score > dept_best_ce[dept]:
            dept_best_ce[dept] = ce_score

    dept_list = sorted(dept_best_ce.items(), key=lambda x: x[1], reverse=True)
    dept_names = [d for d, _ in dept_list]
    dept_raws = [s for _, s in dept_list]
    dept_softmax = softmax(dept_raws).tolist()

    confidence, fallback, _reason = determine_confidence(dept_softmax, dept_raws)

    department = dept_names[0]
    dept_ce_hits = [
        (s, h) for s, h in ce_scored if h.payload["department"] == department
    ]
    if dept_ce_hits:
        subdepartment = dept_ce_hits[0][1].payload["subdepartment"]
        if len(dept_ce_hits) > 1:
            subdept_softmax = softmax([s for s, _ in dept_ce_hits]).tolist()
            score_subdept = subdept_softmax[0]
        else:
            score_subdept = 1.0
    else:
        subdepartment = None
        score_subdept = 0.0

    gap = (dept_softmax[0] - dept_softmax[1]) if len(dept_softmax) > 1 else 1.0

    result = RouteResult(
        department=department,
        subdepartment=subdepartment,
        confidence=confidence,
        score_dept=float(dept_softmax[0]),
        score_subdept=float(score_subdept),
        fallback=fallback,
        raw_score=float(dept_raws[0]),
    )

    dept_scores = [
        (dept, float(sm), float(raw))
        for dept, sm, raw in zip(dept_names, dept_softmax, dept_raws)
    ]

    return {
        "result": result,
        "dept_scores": dept_scores,
        "raw_scores": dept_raws,
        "gap": gap,
        "top_raw": dept_raws[0],
        "beam_depts": beam_depts,
        "ce_candidates": len(ce_scored),
    }


def get_routing_context(text: str, top_k: int = 2) -> dict:
    client = get_qdrant()
    if not client.collection_exists(COLLECTION_NAME):
        from ml.index import create_collection
        create_collection()

    l1_by_dept, beam_depts, l2_with_path = beam_retrieve(text)
    if not l2_with_path:
        return {"ranked_departments": []}

    ce_scored = ce_rerank(text, l2_with_path)
    if not ce_scored:
        return {"ranked_departments": []}

    dept_best_ce: dict = {}
    dept_ce_all: dict = {}
    for ce_score, hit in ce_scored:
        dept = hit.payload["department"]
        if dept not in dept_best_ce or ce_score > dept_best_ce[dept]:
            dept_best_ce[dept] = ce_score
        if dept not in dept_ce_all:
            dept_ce_all[dept] = []
        dept_ce_all[dept].append((ce_score, hit))

    dept_list = sorted(dept_best_ce.items(), key=lambda x: x[1], reverse=True)

    ranked_departments = []
    for dept, score in dept_list[:top_k]:
        dept_hits = dept_ce_all.get(dept, [])
        subdepartments = []
        seen_sub = set()
        for ce_s, hit in sorted(dept_hits, key=lambda x: x[0], reverse=True):
            sub = hit.payload["subdepartment"]
            if sub not in seen_sub:
                subdepartments.append({
                    "subdepartment": sub,
                    "text": hit.payload["text"],
                    "score": float(ce_s),
                })
                seen_sub.add(sub)

        l1_hit = l1_by_dept.get(dept)
        l1_text = l1_hit.payload["text"] if l1_hit else ""

        ranked_departments.append({
            "department": dept,
            "score": float(score),
            "text": l1_text,
            "subdepartments": subdepartments,
        })

    return {"ranked_departments": ranked_departments}
