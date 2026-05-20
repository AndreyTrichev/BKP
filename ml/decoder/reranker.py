import numpy as np

from ml.config import (
    CE_FINAL_LIMIT,
    CE_QUERY_MAX_CHARS,
    CE_TEXT_MAX_CHARS,
    DEPT_NAMES,
    SUBDEPT_NAMES,
)
from ml.decoder.singletons import get_cross_encoder


def softmax(scores) -> np.ndarray:
    s = np.array(scores, dtype=np.float64)
    e = np.exp(s - s.max())
    return e / e.sum()


def compact_text(text: str, max_chars: int = CE_TEXT_MAX_CHARS) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars]


def ce_rerank(text: str, l2_with_path: list):
    if not l2_with_path:
        return []

    best_per_subdept = {}
    for path_score, hit in l2_with_path:
        sub = hit.payload["subdepartment"]
        if sub not in best_per_subdept:
            best_per_subdept[sub] = (path_score, hit)

    candidates = sorted(
        best_per_subdept.values(), key=lambda x: x[0], reverse=True,
    )[:CE_FINAL_LIMIT]

    ce = get_cross_encoder()
    ce_query = compact_text(text, max_chars=CE_QUERY_MAX_CHARS)

    pairs, hits = [], []
    for _, hit in candidates:
        dept = hit.payload["department"]
        sub = hit.payload["subdepartment"]
        dept_name = DEPT_NAMES.get(dept, dept)
        sub_name = SUBDEPT_NAMES.get(sub, sub or "")
        raw_text = compact_text(hit.payload.get("ce_text") or hit.payload["text"])
        ce_text = f"{dept_name}. {sub_name}. {raw_text}"
        pairs.append((ce_query, ce_text))
        hits.append(hit)

    raw_scores = ce.predict(pairs)
    scored = list(zip(raw_scores.tolist(), hits))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
