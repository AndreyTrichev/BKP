"""Параметры baseline pipeline. Vendored из github.com/AndreyTrichev/rag.

Симметричен `ml/config.py`, но:
- меньше констант (нет beam-параметров, нет path-весов, нет RAW_FLOOR/GAP_MIN)
- свой COLLECTION_NAME для изоляции от продвинутого pipeline
"""
import os


# ── Модели (те же что в продвинутом) ────────────────────────────────────────

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"


# ── Qdrant — изолированная коллекция ────────────────────────────────────────

QDRANT_URL = os.environ.get("QDRANT_URL", ":memory:")
COLLECTION_NAME = "citizen_complaints_baseline"


# ── ANN-prefilter и CE параметры ────────────────────────────────────────────

L2_PREFILTER_LIMIT = max(1, int(os.environ.get("L2_PREFILTER_LIMIT", "8")))
CE_CANDIDATES_LIMIT = max(1, int(os.environ.get("CE_CANDIDATES_LIMIT", "4")))
CE_TEXT_MAX_CHARS = max(200, int(os.environ.get("CE_TEXT_MAX_CHARS", "900")))
CE_QUERY_MAX_CHARS = max(120, int(os.environ.get("CE_QUERY_MAX_CHARS", "500")))


# ── Confidence (упрощённая логика: 1 порог high + 1 порог medium) ───────────

SCORE_HIGH = 0.347
GAP_HIGH = 0.004
SCORE_MEDIUM = 0.250
