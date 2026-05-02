"""
treaties.py
-----------
Reinsurance treaty structures for applying to simulated losses.

Each class implements a specific treaty type and exposes an apply()
method that takes simulated losses and returns the ceded amount.
The apply() method is designed to work with numpy arrays for
efficient vectorised computation across large numbers of simulated
losses.

Supported treaty types:
    - ExcessOfLoss : per-occurrence XL treaty
    - StopLoss     : aggregate stop-loss treaty

Note on aggregate limits (AAL):
    The current implementation does not include aggregate annual limits
    or annual aggregate deductibles. These are planned for Phase 4.
    When implemented, the AAL will track ceded losses claim by claim
    and partially cede the claim that straddles the AAL threshold,
    rather than simply clipping the annual total. This allows precise
    identification of which individual claims consumed the AAL.
"""

import numpy as np


class ExcessOfLoss:
    """
    Per-occurrence Excess-of-Loss (XL) treaty.

    Covers individual losses between the retention and the exhaustion
    point (retention + limit) on a per-occurrence basis. The reinsurer
    pays the portion of each individual loss that exceeds the retention,
    subject to the per-occurrence limit.

    For a loss X, the ceded amount per occurrence is:
        C_i = min(max(X_i - R, 0), L)

    where R is the retention and L is the limit.

    The treaty is applied to each individual claim before aggregation.
    The total annual ceded loss is the sum of ceded amounts across
    all claims in the accident year.

    Parameters
    ----------
    retention : float
        The cedant's per-occurrence retention (also called deductible).
        The reinsurer pays nothing below this threshold. Must be
        non-negative.
    limit : float
        The reinsurer's maximum liability per occurrence. The reinsurer
        pays nothing above retention + limit. Must be positive.
    """

    def __init__(self, retention: float, limit: float):
        if retention < 0:
            raise ValueError("retention must be non-negative")
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.retention = retention
        self.limit = limit

    def apply(self, losses: np.ndarray) -> np.ndarray:
        """
        Apply the XL treaty to an array of individual losses.

        Vectorised operation — processes all claims in a single year
        simultaneously using numpy without a Python loop.

        The three steps are:
            1. losses - retention  : subtract the retention from each loss
            2. max(..., 0)         : clip negatives to zero (below retention = no cession)
            3. min(..., limit)     : cap at the limit (above exhaustion = no extra cession)

        Parameters
        ----------
        losses : np.ndarray
            Array of individual gross loss amounts for one accident year.

        Returns
        -------
        np.ndarray
            Array of ceded loss amounts, one per claim. Same shape as losses.
            Values are guaranteed to be in [0, limit].
        """
        return np.minimum(np.maximum(losses - self.retention, 0), self.limit)

    def upper_limit(self) -> float:
        """
        Exhaustion point of the treaty (retention + limit).

        Any individual loss above this amount results in the same
        maximum ceded loss — the reinsurer's liability is capped here.
        """
        return self.retention + self.limit


class StopLoss:
    """
    Stop-Loss treaty applied to aggregate annual losses.

    Unlike the per-occurrence XL, the Stop-Loss treaty operates on the
    total aggregate loss for the entire accident year. The reinsurer
    pays the portion of the aggregate annual loss that exceeds the
    attachment point, subject to a cap.

    For an aggregate annual loss S, the ceded amount is:
        C = min(max(S - A, 0), M)

    where A is the attachment point and M is the cap.

    Parameters
    ----------
    attachment : float
        The aggregate loss level above which the reinsurer begins paying.
        The cedant retains all losses below this threshold. Must be
        non-negative.
    cap : float
        The reinsurer's maximum aggregate liability for the year.
        Must be positive.
    """

    def __init__(self, attachment: float, cap: float):
        if attachment < 0:
            raise ValueError("attachment must be non-negative")
        if cap <= 0:
            raise ValueError("cap must be positive")
        self.attachment = attachment
        self.cap = cap

    def apply(self, aggregate_loss: float) -> float:
        """
        Apply the Stop-Loss treaty to a single aggregate annual loss.

        Parameters
        ----------
        aggregate_loss : float
            Total gross aggregate loss for one simulated accident year.

        Returns
        -------
        float
            Ceded aggregate loss for the year.
            Guaranteed to be in [0, cap].
        """
        return min(max(aggregate_loss - self.attachment, 0), self.cap)