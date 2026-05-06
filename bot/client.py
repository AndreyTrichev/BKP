import logging
from typing import Optional

import httpx

from bot.config import ML_API_URL, ML_TIMEOUT_SEC

logger = logging.getLogger(__name__)


class MLClient:

    def __init__(self, base_url: str = ML_API_URL):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=ML_TIMEOUT_SEC)

    async def classify(self, text: str) -> dict:
        resp = await self._client.post(
            f"{self.base_url}/classify",
            json={"text": text},
        )
        resp.raise_for_status()
        return resp.json()

    async def health(self) -> Optional[dict]:
        try:
            resp = await self._client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("ml health check failed: %s", e)
            return None

    async def close(self):
        await self._client.aclose()
