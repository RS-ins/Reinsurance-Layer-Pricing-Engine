"""
treaties.py
-----------
Reinsurance treaty structures for applying to simulated losses.

Each class implements a specific treaty type and exposes an apply()
method that takes simulated losses and returns the ceded amount.

Supported treaty types:
    - ExcessOfLoss : per-occurrence XL treaty with optional AAL and AAD
    - StopLoss     : aggregate stop-loss treaty

Phase 4 additions:
    - AAL (Aggregate Annual Limit) with per-claim tracking
    - AAD (Annual Aggregate Deductible)
    - ReinstatementProvision for reinstatement premium calculation

AAL design note:
    The AAL is tracked claim by claim. The claim that straddles the
    AAL threshold is partially ceded — not simply clipped at the
    annual total. This allows precise identification of which
    individual claims consumed the AAL and is actuarially correct
    for reinstatement premium calculations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class XLYearResult:
    """
    Detailed result of applying an XL treaty to one accident year.

    Returned by ExcessOfLoss.apply_detailed() when AAL or AAD
    parameters are active. Provides per-claim ceded amounts alongside
    the annual aggregate, enabling reinstatement premium calculations
    and AAL/AAD analysis.

    Attributes
    ----------
    ceded_per_claim : np.ndarray
        Ceded amount for each individual claim after applying the
        per-occurrence limit, AAL cap, and AAD threshold.
        Shape matches the input losses array.
    annual_ceded : float
        Sum of ceded_per_claim — the total annual ceded loss.
    aal_exhausted : bool
        True if the AAL was fully consumed during the year.
    aad_remaining : float
        Unused AAD at end of year. Zero if AAD was fully absorbed.
    n_reinstatements_used : int
        Number of full limit exhaustions during the year.
        Used by ReinstatementProvision to compute reinstatement premiums.
    """
    ceded_per_claim:      np.ndarray
    annual_ceded:         float
    aal_exhausted:        bool
    aad_remaining:        float
    n_reinstatements_used: int


class ExcessOfLoss:
    """
    Per-occurrence Excess-of-Loss (XL) treaty with optional AAL and AAD.

    Covers individual losses between the retention and the exhaustion
    point (retention + limit) on a per-occurrence basis.

    For a loss X, the ceded amount per occurrence is:
        C_i = min(max(X_i - R, 0), L)

    With AAL: cumulative annual ceded losses are capped at the
    aggregate annual limit. The claim straddling the AAL is partially
    ceded — not simply zeroed — preserving actuarial correctness.

    With AAD: an annual aggregate deductible is applied after the
    per-occurrence treaty. The reinsurer pays nothing until cumulative
    ceded losses exceed the AAD for the year.

    Parameters
    ----------
    retention : float
        Per-occurrence retention. Must be non-negative.
    limit : float
        Per-occurrence limit. Must be positive.
    aggregate_limit : float, optional
        Maximum total annual ceded loss across all occurrences (AAL).
        If None, no annual aggregate cap is applied.
    aggregate_deductible : float, optional
        Minimum annual ceded loss before the reinsurer pays (AAD).
        If None, no annual aggregate deductible is applied.
    """

    def __init__(self, retention: float, limit: float,
                 aggregate_limit: Optional[float] = None,
                 aggregate_deductible: Optional[float] = None):
        if retention < 0:
            raise ValueError("retention must be non-negative")
        if limit <= 0:
            raise ValueError("limit must be positive")
        if aggregate_limit is not None and aggregate_limit <= 0:
            raise ValueError("aggregate_limit must be positive")
        if aggregate_deductible is not None and aggregate_deductible < 0:
            raise ValueError("aggregate_deductible must be non-negative")

        self.retention            = retention
        self.limit                = limit
        self.aggregate_limit      = aggregate_limit
        self.aggregate_deductible = aggregate_deductible

    def apply(self, losses: np.ndarray) -> np.ndarray:
        """
        Apply the XL treaty to an array of individual losses.

        When no AAL or AAD is set, this is a simple vectorised
        operation applying the per-occurrence formula to all claims
        simultaneously.

        When AAL or AAD is active, delegates to apply_detailed()
        and returns only the per-claim ceded amounts for compatibility
        with the simulation engine.

        Parameters
        ----------
        losses : np.ndarray
            Array of individual gross loss amounts for one accident year.

        Returns
        -------
        np.ndarray
            Array of ceded loss amounts, one per claim.
            Values are in [0, limit] per claim, subject to AAL/AAD.
        """
        if self.aggregate_limit is None and self.aggregate_deductible is None:
            # Fast vectorised path — no annual aggregate features
            return np.minimum(np.maximum(losses - self.retention, 0), self.limit)

        return self.apply_detailed(losses).ceded_per_claim

    def apply_detailed(self, losses: np.ndarray) -> XLYearResult:
        """
        Apply the XL treaty with full per-claim tracking.

        Uses vectorised numpy operations throughout — no Python loops.
        This is significantly faster than the original claim-by-claim
        approach and produces identical results.

        The three steps are:

        Step 1 — Per-occurrence ceded amounts:
            Apply the per-occurrence formula to every claim simultaneously:
            C_i = min(max(X_i - retention, 0), limit)

        Step 2 — Apply AAL via cumulative sum:
            Compute the cumulative ceded loss before each claim using
            np.cumsum. The remaining AAL before claim i is:
                remaining_i = max(AAL - cumsum[i-1], 0)
            Each claim cedes only min(C_i, remaining_i), so the claim
            that straddles the AAL threshold is partially ceded rather
            than simply zeroed out. This is actuarially correct and
            allows precise identification of which claims consumed the AAL.

        Step 3 — Apply AAD via cumulative deduction:
            If the total annual ceded loss is below the AAD, the reinsurer
            pays nothing. Otherwise, the AAD is subtracted from the first
            claims in order using a cumulative deduction approach, leaving
            only the excess over the AAD.

        Parameters
        ----------
        losses : np.ndarray
            Array of individual gross loss amounts for one accident year.

        Returns
        -------
        XLYearResult
            Dataclass containing:
            - ceded_per_claim : per-claim ceded amounts after AAL and AAD
            - annual_ceded    : sum of ceded_per_claim
            - aal_exhausted   : True if the AAL was fully consumed
            - aad_remaining   : unused AAD at end of year (0 if AAD absorbed)
            - n_reinstatements_used : number of full limit exhaustions,
            used by ReinstatementProvision to compute reinstatement premiums

        Notes
        -----
        The partial cession of the AAL-straddling claim is implemented via:
            remaining = max(AAL - cumsum_before_claim, 0)
            ceded_i   = min(per_occurrence_i, remaining)

        This ensures the total annual ceded loss never exceeds the AAL
        while correctly attributing partial payment to the straddling claim.
        """
        per_occurrence = np.minimum(
        np.maximum(losses - self.retention, 0), self.limit
        )

        if self.aggregate_limit is not None:
            cumsum    = np.cumsum(per_occurrence)
            remaining = np.maximum(
                self.aggregate_limit - np.concatenate([[0], cumsum[:-1]]), 0
            )
            ceded         = np.minimum(per_occurrence, remaining)
            aal_exhausted = bool(cumsum[-1] >= self.aggregate_limit) if len(cumsum) > 0 else False
            n_reinst      = int((ceded >= self.limit).sum())
        else:
            ceded         = per_occurrence.copy()
            aal_exhausted = False
            n_reinst      = int((per_occurrence >= self.limit).sum())

        aad_remaining = 0.0
        if self.aggregate_deductible is not None:
            total = ceded.sum()
            if total <= self.aggregate_deductible:
                aad_remaining = float(self.aggregate_deductible - total)
                ceded         = np.zeros(len(losses))
            else:
                cumsum_c  = np.cumsum(ceded)
                aad       = self.aggregate_deductible
                deducted  = np.minimum(ceded, np.maximum(aad - np.concatenate([[0], cumsum_c[:-1]]), 0))
                ceded     = ceded - deducted

        return XLYearResult(
            ceded_per_claim=ceded,
            annual_ceded=float(ceded.sum()),
            aal_exhausted=aal_exhausted,
            aad_remaining=aad_remaining,
            n_reinstatements_used=n_reinst,
        )

    def upper_limit(self) -> float:
        """Exhaustion point of the treaty (retention + limit)."""
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

    Parameters
    ----------
    attachment : float
        The aggregate loss level above which the reinsurer begins paying.
        Must be non-negative.
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
            Ceded aggregate loss for the year. Guaranteed to be in [0, cap].
        """
        return min(max(aggregate_loss - self.attachment, 0), self.cap)


@dataclass
class ReinstatementProvision:
    """
    Reinstatement premium provision for XL treaties.

    After a loss event exhausts the per-occurrence limit, the cedant
    can reinstate the limit by paying an additional premium. This is
    standard market practice for XL treaties.

    The standard structure is:
        - 1 free reinstatement (no additional premium)
        - 1 paid reinstatement at 100% pro rata as to time

    Pro rata as to time means the reinstatement premium is scaled by
    the fraction of the accident year remaining at the time of the
    reinstatement:
        reinstatement_premium = original_premium * pro_rata_fraction

    Parameters
    ----------
    n_free : int
        Number of free reinstatements. Default is 1.
    n_paid : int
        Number of paid reinstatements at 100% pro rata as to time.
        Default is 1.
    original_premium : float
        The technical premium for the layer — used as the base for
        reinstatement premium calculations.

    Notes
    -----
    In the Monte Carlo simulation, the timing of reinstatements within
    the accident year is approximated by assuming losses occur uniformly
    throughout the year. The k-th reinstatement is assumed to occur at
    time k / (n_reinstatements + 1) within the year, giving a pro rata
    fraction of 1 - k / (n_reinstatements + 1).
    """

    n_free:           int   = 1
    n_paid:           int   = 1
    original_premium: float = 0.0

    def __post_init__(self):
        if self.n_free < 0:
            raise ValueError("n_free must be non-negative")
        if self.n_paid < 0:
            raise ValueError("n_paid must be non-negative")
        if self.original_premium < 0:
            raise ValueError("original_premium must be non-negative")

    @property
    def max_reinstatements(self) -> int:
        """Maximum number of reinstatements allowed (free + paid)."""
        return self.n_free + self.n_paid

    def reinstatement_premium(self, n_reinstatements: int) -> float:
        """
        Compute the total reinstatement premium for a given number
        of limit exhaustions in a single accident year.

        The first n_free exhaustions trigger no premium. Subsequent
        exhaustions trigger a pro rata premium based on when in the
        year the reinstatement occurs.

        Parameters
        ----------
        n_reinstatements : int
            Number of times the per-occurrence limit was exhausted
            during the accident year.

        Returns
        -------
        float
            Total reinstatement premium payable by the cedant for
            this accident year.
        """
        if n_reinstatements <= 0:
            return 0.0

        n_eff = min(n_reinstatements, self.max_reinstatements)

        total_premium = 0.0
        for k in range(1, n_eff + 1):
            if k <= self.n_free:
                continue
            # Use n_eff (not n_reinstatements) in denominator
            # so pro rata is consistent regardless of total exhaustions
            pro_rata = 1.0 - k / (n_eff + 1)
            total_premium += self.original_premium * pro_rata

        return total_premium

    def net_cost(self, ceded_loss: float, n_reinstatements: int) -> float:
        """
        Net cost to the cedant for a single accident year.

        Net cost = ceded_loss - reinstatement_premium_received_by_reinsurer
        From the cedant's perspective:
            net_recovery = ceded_loss - reinstatement_premium

        Parameters
        ----------
        ceded_loss : float
            Total ceded loss for the accident year.
        n_reinstatements : int
            Number of limit exhaustions during the year.

        Returns
        -------
        float
            Net recovery to the cedant after paying reinstatement premiums.
        """
        return ceded_loss - self.reinstatement_premium(n_reinstatements)