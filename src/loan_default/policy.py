"""Approval policy: approve the lowest-risk share of a batch of scored loans."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_TIE_SEED = 42


def approve_safest(scores: ArrayLike, rate: float, seed: int = DEFAULT_TIE_SEED) -> NDArray[np.bool_]:
    """Mark the lowest-scoring `rate` share of loans as approved.

    Equal scores are ordered at random with `seed`, so a model with few distinct
    scores (FICO alone) still approves exactly the requested share.
    """
    if not 0 <= rate <= 1:
        raise ValueError(f"rate must be between 0 and 1, got {rate}")
    scores = np.asarray(scores, dtype=float)
    tie_break = np.random.default_rng(seed).random(len(scores))
    order = np.lexsort((tie_break, scores))
    approved = np.zeros(len(scores), dtype=bool)
    approved[order[: int(round(rate * len(scores)))]] = True
    return approved


def policy_results(y: ArrayLike, approved: NDArray[np.bool_]) -> dict[str, float]:
    """Approval rate, number approved, and default rates among approved and rejected loans."""
    y = np.asarray(y)
    rejected = ~approved
    return {
        "approval_rate": float(approved.mean()),
        "approved": int(approved.sum()),
        "default_rate_approved": float(y[approved].mean()) if approved.any() else float("nan"),
        "default_rate_rejected": float(y[rejected].mean()) if rejected.any() else float("nan"),
    }


def cutoff_at(scores: ArrayLike, rate: float, seed: int = DEFAULT_TIE_SEED) -> float:
    """Highest score among the approved loans: approve a new loan if its score is at or below this."""
    scores = np.asarray(scores, dtype=float)
    return float(scores[approve_safest(scores, rate, seed)].max())
