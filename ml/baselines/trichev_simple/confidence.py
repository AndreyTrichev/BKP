"""Упрощённая confidence-логика baseline.

В отличие от продвинутого (`ml/confidence.py`):
- НЕТ слоя RAW_FLOOR (отбраковка нерелевантных по сырому CE-логиту)
- НЕТ слоя GAP_MIN (отбраковка многотемных по малому gap)
- Только два порога: SCORE_HIGH+GAP_HIGH → high, SCORE_MEDIUM → medium,
                     иначе → low+fallback
"""
from .config import GAP_HIGH, SCORE_HIGH, SCORE_MEDIUM


def determine_confidence(
    scores_desc: list[float],
    raw_scores_desc: list[float],
) -> tuple[str, bool]:
    """Возвращает (confidence, fallback). Baseline-версия — 2 порога без причины."""
    if not scores_desc:
        return "low", True

    top = scores_desc[0]
    gap = (top - scores_desc[1]) if len(scores_desc) > 1 else 1.0

    if top >= SCORE_HIGH and gap >= GAP_HIGH:
        return "high", False
    elif top >= SCORE_MEDIUM:
        return "medium", False
    else:
        return "low", True
