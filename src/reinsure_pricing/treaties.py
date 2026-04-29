import numpy as np


class ExcessOfLoss:
    """
    Per-occurrence Excess-of-Loss treaty.
    Covers individual losses between retention and retention + limit.
    """

    def __init__(self, retention: float, limit: float):
        if retention < 0:
            raise ValueError("retention must be non-negative")
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.retention = retention
        self.limit = limit

    def apply(self, losses: np.ndarray) -> np.ndarray:
        """Apply treaty to an array of individual losses."""
        return np.minimum(np.maximum(losses - self.retention, 0), self.limit)

    def upper_limit(self) -> float:
        return self.retention + self.limit


class StopLoss:
    """
    Stop-Loss treaty applied to aggregate annual losses.
    Covers aggregate losses exceeding attachment, subject to a cap.
    """

    def __init__(self, attachment: float, cap: float):
        if attachment < 0:
            raise ValueError("attachment must be non-negative")
        if cap <= 0:
            raise ValueError("cap must be positive")
        self.attachment = attachment
        self.cap = cap

    def apply(self, aggregate_loss: float) -> float:
        """Apply treaty to a single aggregate annual loss."""
        return min(max(aggregate_loss - self.attachment, 0), self.cap)