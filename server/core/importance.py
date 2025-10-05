"""Importance and token utility helpers."""
from datetime import datetime
from typing import Optional


def approx_tokens(text: Optional[str]) -> int:
    words = len(text.split()) if text else 0
    return int(words / 0.75)


def compute_recency_bonus(created_at: Optional[str]) -> float:
    """Compute a recency bonus given an ISO timestamp string or None."""
    if not created_at:
        return 1.0
    try:
        created_dt = datetime.fromisoformat(created_at.replace("Z", ""))
        age_days = max((datetime.utcnow() - created_dt).days, 0)
        return max(0.1, 1.0 - (age_days / 365.0))
    except Exception:
        return 1.0


def compute_importance(times_recalled: int, created_at: Optional[str], llm_weight: float, pinned: int) -> float:
    recency = compute_recency_bonus(created_at)
    importance = (times_recalled * 0.4) + (recency * 0.3) + (llm_weight * 0.2) + (pinned * 0.1)
    return importance
