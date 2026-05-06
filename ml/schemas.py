from dataclasses import dataclass
from typing import Optional


@dataclass
class RouteResult:
    department: str
    subdepartment: Optional[str]
    confidence: str
    score_dept: float
    score_subdept: float
    fallback: bool
    raw_score: float = 0.0
    fallback_reason: Optional[str] = None

    def get(self, key, default=None):
        return getattr(self, key, default)
