import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.pipeline import route

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      description="Текст обращения гражданина")


class ClassifyResponse(BaseModel):
    department: str
    subdepartment: Optional[str]
    confidence: str                             
    score_dept: float
    score_subdept: float
    raw_score: float
    fallback: bool
    fallback_reason: Optional[str] = None                                              
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    version: str
    qdrant_collection: str
    qdrant_points: int


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPEALS_LOG_PATH = PROJECT_ROOT / "data" / "appeals.jsonl"


def _log_appeal(text: str, response: ClassifyResponse) -> None:
    APPEALS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "text": text,
        "department": response.department,
        "subdepartment": response.subdepartment,
        "confidence": response.confidence,
        "score_dept": response.score_dept,
        "raw_score": response.raw_score,
        "fallback": response.fallback,
        "fallback_reason": response.fallback_reason,
        "latency_ms": response.latency_ms,
    }
    with APPEALS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


app = FastAPI(
    title="RAG Feodosia — маршрутизация обращений граждан",
    description="ML-сервис для классификации обращений в Администрацию г. Феодосия",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    logger.info("=== ML service starting ===")
    from ml.singletons import get_embedder, get_cross_encoder, get_qdrant
    from ml.index import create_collection
    from ml.config import COLLECTION_NAME

    logger.info("Loading embedder...")
    get_embedder()
    logger.info("Loading cross-encoder...")
    get_cross_encoder()

    logger.info("Building Qdrant collection from preambles/...")
    create_collection(recreate=False)

    info = get_qdrant().get_collection(COLLECTION_NAME)
    logger.info(f"Qdrant collection '{COLLECTION_NAME}': {info.points_count} points")
    logger.info("=== ML service ready ===")


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    t0 = time.time()
    try:
        result = route(request.text)
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = int((time.time() - t0) * 1000)
    response = ClassifyResponse(
        department=result.department,
        subdepartment=result.subdepartment,
        confidence=result.confidence,
        score_dept=result.score_dept,
        score_subdept=result.score_subdept,
        raw_score=result.raw_score,
        fallback=result.fallback,
        fallback_reason=result.fallback_reason,
        latency_ms=latency_ms,
    )

    if not os.environ.get("ML_SKIP_APPEAL_LOG"):
        _log_appeal(request.text, response)

    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    from ml.singletons import get_qdrant
    from ml.config import COLLECTION_NAME

    client = get_qdrant()
    if not client.collection_exists(COLLECTION_NAME):
        raise HTTPException(status_code=503, detail="Qdrant collection not ready")

    info = client.get_collection(COLLECTION_NAME)
    return HealthResponse(
        status="ok",
        version=app.version,
        qdrant_collection=COLLECTION_NAME,
        qdrant_points=info.points_count or 0,
    )


@app.get("/")
def root():
    return {
        "service": "rag-feodosia-ml",
        "version": app.version,
        "endpoints": ["POST /classify", "GET /health"],
    }
