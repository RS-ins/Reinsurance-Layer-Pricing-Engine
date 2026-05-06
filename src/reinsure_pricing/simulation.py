"""
simulation.py
-------------
Monte Carlo simulation engine for reinsurance layer pricing.

The MonteCarloEngine orchestrates the full frequency-severity simulation:
    1. Draw claim counts from the frequency distribution (one per year)
    2. Draw individual loss amounts from the severity distribution
    3. Apply the treaty structure to obtain ceded losses
    4. Return a SimulationResults object containing the full
       distribution of annual ceded losses

The SimulationResults class provides risk measures and summary
statistics computed directly from the simulated ceded loss distribution.
"""

import numpy as np
from dataclasses import dataclass
from reinsure_pricing.frequency import PoissonFrequency, NegativeBinomialFrequency
from reinsure_pricing.severity import LognormalSeverity, GammaSeverity, ParetoSeverity
from reinsure_pricing.treaties import ExcessOfLoss, StopLoss


@dataclass
class SimulationResults:
    """
    Container for the output of a Monte Carlo simulation run.

    Parameters
    ----------
    ceded_losses : np.ndarray
        Array of shape (n_simulations,) containing the total ceded
        loss for each simulated accident year.
    reinstatement_premiums : np.ndarray, optional
        Array of shape (n_simulations,) containing the reinstatement
        premium payable by the cedant for each accident year.
        None if no ReinstatementProvision was passed to run().
    """
    ceded_losses:            np.ndarray
    reinstatement_premiums:  np.ndarray | None = None

    @property
    def expected_ceded_loss(self) -> float:
        """Expected Ceded Loss (ECL) — the pure premium."""
        return float(self.ceded_losses.mean())

    @property
    def std_ceded_loss(self) -> float:
        """Standard deviation of the simulated ceded loss distribution."""
        return float(self.ceded_losses.std())

    def var(self, alpha: float = 0.99) -> float:
        """Value at Risk at confidence level alpha."""
        return float(np.quantile(self.ceded_losses, alpha))

    def tvar(self, alpha: float = 0.99) -> float:
        """Tail Value at Risk at confidence level alpha."""
        var  = self.var(alpha)
        tail = self.ceded_losses[self.ceded_losses > var]
        return float(tail.mean()) if len(tail) > 0 else var

    @property
    def tvar_99(self) -> float:
        """Convenience property for TVaR at 99%."""
        return self.tvar(0.99)

    def prob_attachment(self) -> float:
        """Fraction of years the treaty was triggered."""
        return float((self.ceded_losses > 0).mean())

    def prob_exhaustion(self, treaty_limit: float) -> float:
        """Fraction of years the full treaty limit was consumed."""
        return float((self.ceded_losses >= treaty_limit).mean())

    @property
    def expected_reinstatement_premium(self) -> float:
        """
        Expected annual reinstatement premium payable by the cedant.

        Returns 0.0 if no ReinstatementProvision was used.
        The net expected cost to the cedant is:
            ECL - expected_reinstatement_premium
        because the reinstatement premium flows back to the reinsurer.
        """
        if self.reinstatement_premiums is None:
            return 0.0
        return float(self.reinstatement_premiums.mean())

    @property
    def net_expected_recovery(self) -> float:
        """
        Net expected recovery to the cedant after reinstatement premiums.

        net_recovery = ECL - expected_reinstatement_premium
        """
        return self.expected_ceded_loss - self.expected_reinstatement_premium

    def summary(self) -> str:
        """Return a formatted string of key simulation statistics."""
        lines = [
            f"Expected Ceded Loss : {self.expected_ceded_loss:>15,.0f}",
            f"Std Deviation       : {self.std_ceded_loss:>15,.0f}",
            f"VaR 99%             : {self.var(0.99):>15,.0f}",
            f"TVaR 99%            : {self.tvar_99:>15,.0f}",
            f"Prob of Attachment  : {self.prob_attachment():>15.1%}",
        ]
        if self.reinstatement_premiums is not None:
            lines += [
                f"Exp Reinst Premium  : {self.expected_reinstatement_premium:>15,.0f}",
                f"Net Expected Recov  : {self.net_expected_recovery:>15,.0f}",
            ]
        return "\n".join(lines)

class MonteCarloEngine:
    """
    Frequency-severity Monte Carlo simulation engine.

    Orchestrates the full simulation pipeline:
        1. For each of n_simulations accident years, draw a claim count
           from the frequency distribution
        2. For each claim, draw an individual loss from the severity
           distribution
        3. Apply the treaty to obtain the ceded loss for the year
        4. Collect all annual ceded losses into a SimulationResults object

    The engine uses a single numpy random generator seeded at
    initialisation, ensuring all random draws across frequency,
    severity, and any future modules are reproducible and statistically
    independent.

    Parameters
    ----------
    frequency : PoissonFrequency or NegativeBinomialFrequency
        Claim count distribution for the portfolio.
    severity : LognormalSeverity, GammaSeverity, or ParetoSeverity
        Individual loss severity distribution.
    treaty : ExcessOfLoss or StopLoss
        Reinsurance treaty structure to apply to simulated losses.
    n_simulations : int
        Number of accident years to simulate. Higher values produce
        more stable risk measures but take longer to run. Recommended
        minimum is 50,000 for VaR/TVaR estimation. Default is 100,000.
    random_state : int
        Seed for the numpy random generator. Setting this ensures
        identical results across runs and machines. Default is 42.
    """

    def __init__(self, frequency, severity, treaty,
                 n_simulations: int = 100_000,
                 random_state: int = 42):
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        self.frequency = frequency
        self.severity = severity
        self.treaty = treaty
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_state)

    def run(self, reinstatement: "ReinstatementProvision | None" = None) -> SimulationResults:
        """
        Execute the Monte Carlo simulation and return results.

        When a ReinstatementProvision is passed, the simulation also
        computes reinstatement premiums for each accident year and
        returns them in SimulationResults alongside the ceded losses.

        For ExcessOfLoss with AAL or AAD active, uses apply_detailed()
        to get per-claim ceded amounts and reinstatement counts.

        For ExcessOfLoss without AAL/AAD, uses the fast vectorised
        apply() path unless reinstatement tracking is needed.

        For StopLoss, uses the vectorised aggregate approach.

        Parameters
        ----------
        reinstatement : ReinstatementProvision, optional
            If provided, computes reinstatement premiums for each year.
            Only applicable to ExcessOfLoss treaties.

        Returns
        -------
        SimulationResults
            Container with ceded losses and optional reinstatement premiums.
        """
        from reinsure_pricing.treaties import ExcessOfLoss, StopLoss, ReinstatementProvision

        claim_counts   = self.frequency.sample(self.n_simulations, self.rng)
        ceded_losses   = np.zeros(self.n_simulations)
        reinst_premiums = np.zeros(self.n_simulations) if reinstatement else None

        needs_detailed = (
            isinstance(self.treaty, ExcessOfLoss) and (
                self.treaty.aggregate_limit is not None or
                self.treaty.aggregate_deductible is not None or
                reinstatement is not None
            )
        )

        if isinstance(self.treaty, ExcessOfLoss):
            for i, n in enumerate(claim_counts):
                if n == 0:
                    continue

                losses = self.severity.sample(int(n), self.rng)

                if needs_detailed:
                    # Detailed path — track per-claim ceded amounts
                    result = self.treaty.apply_detailed(losses)
                    ceded_losses[i] = result.annual_ceded

                    if reinstatement is not None:
                        # Compute reinstatement premium for this year
                        reinst_premiums[i] = reinstatement.reinstatement_premium(
                            result.n_reinstatements_used
                        )
                else:
                    # Fast vectorised path
                    ceded_losses[i] = self.treaty.apply(losses).sum()

        elif isinstance(self.treaty, StopLoss):
            total_claims = int(claim_counts.sum())
            if total_claims > 0:
                all_losses = self.severity.sample(total_claims, self.rng)
                splits = np.split(all_losses, np.cumsum(claim_counts[:-1]))

                for i, year_losses in enumerate(splits):
                    if len(year_losses) == 0:
                        continue
                    aggregate = year_losses.sum()
                    ceded_losses[i] = self.treaty.apply(aggregate)

        return SimulationResults(
            ceded_losses=ceded_losses,
            reinstatement_premiums=reinst_premiums,
        )