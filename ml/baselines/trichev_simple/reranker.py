"""CE rerank + helpers (softmax, compact_text).

В отличие от продвинутого reranker'а (`ml/reranker.py`):
- Diversify — по dept, а не по subdept (грубее, теряет subdept-информацию)
- НЕТ context injection в CE-пары: подаётся сырой chunk, без префикса
  '{Dept}. {Sub}.'
- CE_CANDIDATES_LIMIT=4 (vs CE_FINAL_LIMIT=10 в продвинутом)
"""
import numpy as np

from .config import (
    CE_CANDIDATES_LIMIT,
    CE_QUERY_MAX_CHARS,
    CE_TEXT_MAX_CHARS,
)
from .singletons import get_cross_encoder


def softmax(scores) -> np.ndarray:
    """Численно стабильный softmax."""
    s = np.array(scores, dtype=np.float64)
    e = np.exp(s - s.max())
    return e / e.sum()


def compact_text(text: str, max_chars: int = CE_TEXT_MAX_CHARS) -> str:
    """Удаляет лишние пробелы, обрезает до max_chars."""
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars]


def select_l2_for_ce(l2_hits: list) -> list:
    """Diversify candidates per dept (1 на dept), потом дополняем по ANN-скору."""
    if len(l2_hits) <= CE_CANDIDATES_LIMIT:
        return l2_hits

    selected = []
    seen_ids = set()
    seen_departments = set()

    for hit in l2_hits:
        hid = str(hit.id)
        dept = hit.payload["department"]
        if dept in seen_departments or hid in seen_ids:
            continue
        selected.append(hit)
        seen_ids.add(hid)
        seen_departments.add(dept)
        if len(selected) >= CE_CANDIDATES_LIMIT:
            return selected

    # Дополняем оставшимися хитами по ANN-скору
    for hit in l2_hits:
        hid = str(hit.id)
        if hid in seen_ids:
            continue
        selected.append(hit)
        seen_ids.add(hid)
        if len(selected) >= CE_CANDIDATES_LIMIT:
            break

    return selected


def ce_rerank(text: str, l2_hits: list) -> list:
    """Diversify per dept → CE.predict → [(raw_score, hit), ...].

    NB: подаём в CE сырой chunk без context injection — это одно из
    архитектурных отличий от продвинутого pipeline.
    """
    if not l2_hits:
        return []

    ce_hits = select_l2_for_ce(l2_hits)
    ce = get_cross_encoder()
    ce_query = compact_text(text, max_chars=CE_QUERY_MAX_CHARS)

    pairs = [
        (ce_query, compact_text(hit.payload.get("ce_text") or hit.payload["text"]))
        for hit in ce_hits
    ]
    raw_scores = ce.predict(pairs)
    return list(zip(raw_scores.tolist(), ce_hits))
