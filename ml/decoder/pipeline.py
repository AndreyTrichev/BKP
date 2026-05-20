from ml.config import (
    BEAM_L2_LIMIT,
    BEAM_MIN,
    COLLECTION_NAME,
    CE_FINAL_LIMIT,
    MAXP_PER_DEPT,
    REL_PRUNE,
    RERANK_ALPHA,
    RERANK_MODE,
    RRF_K,
    W_L1,
    W_L2,
)
from ml.confidence import determine_confidence
from ml.decoder.reranker import ce_rerank, softmax
from ml.decoder.singletons import get_qdrant
from ml.retriever import beam_retrieve
from ml.schemas import RouteResult


def _rerank_within_dept(dept_ce_hits):
    if not dept_ce_hits or RERANK_MODE == "none":
        return dept_ce_hits

    if RERANK_MODE == "hybrid":
        ce_scores = [s for s, _ in dept_ce_hits]
        l2_scores = [h.score for _, h in dept_ce_hits]
        ce_min, ce_max = min(ce_scores), max(ce_scores)
        l2_min, l2_max = min(l2_scores), max(l2_scores)
        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
        l2_range = l2_max - l2_min if l2_max > l2_min else 1.0
        scored = []
        for ce_s, hit in dept_ce_hits:
            norm_ce = (ce_s - ce_min) / ce_range
            norm_l2 = (hit.score - l2_min) / l2_range
            combined = RERANK_ALPHA * norm_ce + (1 - RERANK_ALPHA) * norm_l2
            scored.append((combined, ce_s, hit))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(ce_s, hit) for _, ce_s, hit in scored]

    if RERANK_MODE == "rrf":
        ce_sorted = sorted(dept_ce_hits, key=lambda x: -x[0])
        l2_sorted = sorted(dept_ce_hits, key=lambda x: -x[1].score)
        ce_rank = {id(item[1]): i for i, item in enumerate(ce_sorted)}
        l2_rank = {id(item[1]): i for i, item in enumerate(l2_sorted)}
        scored = []
        for ce_s, hit in dept_ce_hits:
            rrf = 1.0 / (RRF_K + ce_rank[id(hit)]) + 1.0 / (RRF_K + l2_rank[id(hit)])
            scored.append((rrf, ce_s, hit))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(ce_s, hit) for _, ce_s, hit in scored]

    return dept_ce_hits


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
        dept_ce_hits = _rerank_within_dept(dept_ce_hits)
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
    empty = {"result": _EMPTY_RESULT, "dept_scores": [], "raw_scores": [], "gap": 0.0, "top_raw": 0.0}
    if not l2_with_path:
        return empty
    ce_scored = ce_rerank(text, l2_with_path)
    if not ce_scored:
        return empty

    dept_best_ce: dict = {}
    for ce_score, hit in ce_scored:
        dept = hit.payload["department"]
        if dept not in dept_best_ce or ce_score > dept_best_ce[dept]:
            dept_best_ce[dept] = ce_score
    dept_list = sorted(dept_best_ce.items(), key=lambda x: x[1], reverse=True)
    dept_names = [d for d, _ in dept_list]
    dept_raws = [s for _, s in dept_list]
    dept_softmax = softmax(dept_raws).tolist()
    confidence, fallback, _ = determine_confidence(dept_softmax, dept_raws)

    department = dept_names[0]
    dept_ce_hits = [(s, h) for s, h in ce_scored if h.payload["department"] == department]
    dept_ce_hits = _rerank_within_dept(dept_ce_hits)
    if dept_ce_hits:
        subdepartment = dept_ce_hits[0][1].payload["subdepartment"]
        if len(dept_ce_hits) > 1:
            score_subdept = softmax([s for s, _ in dept_ce_hits]).tolist()[0]
        else:
            score_subdept = 1.0
    else:
        subdepartment = None
        score_subdept = 0.0

    gap = (dept_softmax[0] - dept_softmax[1]) if len(dept_softmax) > 1 else 1.0

    result = RouteResult(
        department=department, subdepartment=subdepartment, confidence=confidence,
        score_dept=float(dept_softmax[0]), score_subdept=float(score_subdept),
        fallback=fallback, raw_score=float(dept_raws[0]),
    )
    dept_scores = [(d, float(sm), float(raw)) for d, sm, raw in zip(dept_names, dept_softmax, dept_raws)]
    return {
        "result": result, "dept_scores": dept_scores, "raw_scores": dept_raws,
        "gap": gap, "top_raw": dept_raws[0],
        "beam_depts": beam_depts, "ce_candidates": len(ce_scored),
    }


def route_debug_full(text: str, gt_dept: str | None = None, gt_sub: str | None = None) -> dict:
    client = get_qdrant()
    if not client.collection_exists(COLLECTION_NAME):
        from ml.index import create_collection
        create_collection()

    l1_by_dept, beam_depts, l2_with_path = beam_retrieve(text)

    l1_stage = []
    if l1_by_dept:
        sorted_l1 = sorted(l1_by_dept.items(), key=lambda x: x[1].score, reverse=True)
        best_l1 = sorted_l1[0][1].score
        threshold = REL_PRUNE * best_l1
        for rank, (d, hit) in enumerate(sorted_l1):
            l1_stage.append({
                "rank": rank, "department": d, "score": float(hit.score),
                "in_beam": d in beam_depts, "above_threshold": hit.score >= threshold,
            })

    l2_stage = []
    for rank, (path_score, hit) in enumerate(l2_with_path):
        dept = hit.payload["department"]
        sub = hit.payload["subdepartment"]
        l1_score = l1_by_dept[dept].score if dept in l1_by_dept else 0.0
        l2_stage.append({
            "rank": rank, "department": dept, "subdepartment": sub,
            "l1_score": float(l1_score), "l2_score": float(hit.score),
            "path_score": float(path_score),
        })

    config_dict = {
        "W_L1": W_L1, "W_L2": W_L2, "REL_PRUNE": REL_PRUNE,
        "BEAM_L2_LIMIT": BEAM_L2_LIMIT, "BEAM_MIN": BEAM_MIN,
        "CE_FINAL_LIMIT": CE_FINAL_LIMIT, "MAXP_PER_DEPT": MAXP_PER_DEPT,
        "RERANK_MODE": RERANK_MODE, "RERANK_ALPHA": RERANK_ALPHA, "RRF_K": RRF_K,
        "BACKEND": "decoder",
    }

    if not l2_with_path:
        return {
            "config": config_dict, "gt_dept": gt_dept, "gt_sub": gt_sub,
            "l1_stage": l1_stage, "beam_depts": list(beam_depts),
            "l2_stage": [], "ce_stage": [], "aggregation": None, "result": None,
            "diagnostics": {"empty_l2": True},
        }

    ce_scored = ce_rerank(text, l2_with_path)
    ce_stage = []
    for rank, (ce_score, hit) in enumerate(ce_scored):
        dept = hit.payload["department"]
        sub = hit.payload["subdepartment"]
        l1_score = l1_by_dept[dept].score if dept in l1_by_dept else 0.0
        l2_match = next((p for p in l2_stage if p["department"] == dept and p["subdepartment"] == sub), None)
        ce_stage.append({
            "rank": rank, "department": dept, "subdepartment": sub,
            "ce_score": float(ce_score),
            "l1_score": float(l1_score),
            "l2_score": float(l2_match["l2_score"]) if l2_match else None,
            "path_score": float(l2_match["path_score"]) if l2_match else None,
        })

    aggregation = {}
    if MAXP_PER_DEPT:
        dept_best_ce: dict = {}
        for ce_score, hit in ce_scored:
            d = hit.payload["department"]
            if d not in dept_best_ce or ce_score > dept_best_ce[d]:
                dept_best_ce[d] = ce_score
        dept_list = sorted(dept_best_ce.items(), key=lambda x: x[1], reverse=True)
        dept_names = [d for d, _ in dept_list]
        dept_raws = [s for _, s in dept_list]
        dept_softmax = softmax(dept_raws).tolist()
        aggregation["mode"] = "maxp_per_dept"
        aggregation["dept_max_ce"] = {d: float(s) for d, s in dept_best_ce.items()}
        aggregation["dept_softmax"] = {d: float(p) for d, p in zip(dept_names, dept_softmax)}
        aggregation["winner_dept"] = dept_names[0]
        aggregation["gap_dept"] = float(dept_softmax[0] - dept_softmax[1]) if len(dept_softmax) > 1 else 1.0
        winner = dept_names[0]
        winner_ce_hits = [(s, h) for s, h in ce_scored if h.payload["department"] == winner]
        winner_ce_hits = _rerank_within_dept(winner_ce_hits)
        if winner_ce_hits:
            sub_scores = [s for s, _ in winner_ce_hits]
            sub_softmax_in_dept = softmax(sub_scores).tolist() if len(sub_scores) > 1 else [1.0]
            aggregation["winner_sub"] = winner_ce_hits[0][1].payload["subdepartment"]
            aggregation["sub_softmax_in_dept"] = float(sub_softmax_in_dept[0])
        else:
            aggregation["winner_sub"] = None
        confidence, fallback, reason = determine_confidence(dept_softmax, dept_raws)
    else:
        flat_raws = [s for s, _ in ce_scored]
        flat_softmax = softmax(flat_raws).tolist()
        winner_score, winner_hit = ce_scored[0]
        aggregation["mode"] = "flat"
        aggregation["winner_dept"] = winner_hit.payload["department"]
        aggregation["winner_sub"] = winner_hit.payload["subdepartment"]
        aggregation["flat_softmax_top"] = float(flat_softmax[0])
        aggregation["gap_flat"] = float(flat_softmax[0] - flat_softmax[1]) if len(flat_softmax) > 1 else 1.0
        confidence, fallback, reason = determine_confidence(flat_softmax, flat_raws)

    result = {
        "department": aggregation["winner_dept"],
        "subdepartment": aggregation.get("winner_sub"),
        "confidence": confidence,
        "fallback": fallback,
        "fallback_reason": reason,
        "raw_score": float(ce_scored[0][0]),
    }

    diagnostics = {}
    if gt_dept is not None:
        diagnostics["gt_dept"] = gt_dept
        diagnostics["gt_in_beam"] = gt_dept in beam_depts
        diagnostics["gt_l1_score"] = next((s["score"] for s in l1_stage if s["department"] == gt_dept), None)
        diagnostics["gt_l1_rank"] = next((s["rank"] for s in l1_stage if s["department"] == gt_dept), None)
    if gt_sub is not None:
        l2_match = next(((s["rank"], s) for s in l2_stage if s["subdepartment"] == gt_sub and s["department"] == gt_dept), None)
        diagnostics["gt_in_l2_top"] = l2_match is not None
        diagnostics["gt_l2_rank"] = l2_match[0] if l2_match else None
        diagnostics["gt_l2_path_score"] = l2_match[1]["path_score"] if l2_match else None
        ce_match = next(((s["rank"], s) for s in ce_stage if s["subdepartment"] == gt_sub and s["department"] == gt_dept), None)
        diagnostics["gt_in_ce_top"] = ce_match is not None
        diagnostics["gt_ce_rank"] = ce_match[0] if ce_match else None
        diagnostics["gt_ce_score"] = ce_match[1]["ce_score"] if ce_match else None
        diagnostics["winner_ce_score"] = ce_stage[0]["ce_score"] if ce_stage else None
        if ce_match:
            diagnostics["ce_score_gap_to_winner"] = ce_stage[0]["ce_score"] - ce_match[1]["ce_score"]
        diagnostics["dept_correct"] = result["department"] == gt_dept
        diagnostics["sub_correct"] = (
            result["department"] == gt_dept
            and (result["subdepartment"] or "unknown") == (gt_sub or "unknown")
        )

    return {
        "config": config_dict, "gt_dept": gt_dept, "gt_sub": gt_sub,
        "l1_stage": l1_stage, "beam_depts": list(beam_depts),
        "l2_stage": l2_stage, "ce_stage": ce_stage,
        "aggregation": aggregation, "result": result, "diagnostics": diagnostics,
    }
