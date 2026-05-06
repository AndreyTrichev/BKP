from ml.config import GAP_HIGH, GAP_MIN, RAW_FLOOR, SCORE_HIGH


def determine_confidence(
    softmax_scores: list[float],
    raw_scores: list[float],
) -> tuple[str, bool, str | None]:
    if not softmax_scores:
        return "low", True, "no_candidates"

    top = softmax_scores[0]
    top_raw = raw_scores[0] if raw_scores else top
    gap = (top - softmax_scores[1]) if len(softmax_scores) > 1 else 1.0

                                                                  
    if top_raw < RAW_FLOOR:
        return "low", True, "raw_floor"

                                                        
    if gap < GAP_MIN:
        return "low", True, "gap_min"

                            
    if top >= SCORE_HIGH and gap >= GAP_HIGH:
        return "high", False, None
    return "medium", False, None
