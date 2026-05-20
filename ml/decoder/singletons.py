from typing import Optional

from ml.config import CE_MAX_LENGTH, CROSS_ENCODER_MODEL
from ml.decoder.decoder_ce import DecoderCrossEncoder
from ml.singletons import get_embedder, get_qdrant

_cross_encoder: Optional[DecoderCrossEncoder] = None


def get_cross_encoder() -> DecoderCrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = DecoderCrossEncoder(CROSS_ENCODER_MODEL, max_length=CE_MAX_LENGTH)
    return _cross_encoder


__all__ = ["get_cross_encoder", "get_embedder", "get_qdrant"]
