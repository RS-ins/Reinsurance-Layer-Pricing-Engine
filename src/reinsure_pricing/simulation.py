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

    Stores the full array of simulated annual ceded losses and provides
    methods for computing risk measures directly from the empirical
    distribution.

    Parameters
    ----------
    ceded_losses : np.ndarray
        Array of shape (n_simulations,) containing the total ceded loss
        for each simulated accident year. Years with no ceded loss are
        stored as 0.0.
    """

    ceded_losses: np.ndarray

    @property
    def expected_ceded_loss(self) -> float:
        """
        Expected Ceded Loss (ECL) — the pure premium.

        Computed as the arithmetic mean of all simulated annual ceded
        losses, including years with zero cession. This is the amount
        the reinsurer expects to pay on average per accident year.
        """
        return float(self.ceded_losses.mean())

    @property
    def std_ceded_loss(self) -> float:
        """
        Standard deviation of the simulated ceded loss distribution.

        Measures the spread of annual ceded losses around the ECL.
        A high standard deviation relative to the ECL indicates a
        volatile layer where annual results are unpredictable.
        """
        return float(self.ceded_losses.std())

    def var(self, alpha: float = 0.99) -> float:
        """
        Value at Risk (VaR) at confidence level alpha.

        VaR answers: 'what is the ceded loss level that will NOT be
        exceeded in alpha% of accident years?'

        For example, VaR 99% = 2.5M means that in 99 out of 100
        simulated years, ceded losses were below 2.5M. Only 1 in 100
        years produced a ceded loss above this threshold.

        Parameters
        ----------
        alpha : float
            Confidence level, between 0 and 1. Common values are
            0.95, 0.99, and 0.995. Default is 0.99.

        Returns
        -------
        float
            The alpha-quantile of the simulated ceded loss distribution.
        """
        return float(np.quantile(self.ceded_losses, alpha))

    def tvar(self, alpha: float = 0.99) -> float:
        """
        Tail Value at Risk (TVaR) at confidence level alpha.

        Also known as Conditional Tail Expectation (CTE) or Expected
        Shortfall (ES). TVaR answers: 'given that we are already in
        the worst (1-alpha)% of years, what is the average ceded loss?'

        TVaR is always >= VaR at the same confidence level. The gap
        between TVaR and VaR indicates how severe losses are once the
        VaR threshold is breached — a large gap signals a heavy tail.

        TVaR is used in the pricing formula as the capital proxy:
            capital_load = cost_of_capital × max(TVaR99 - ECL, 0)

        Parameters
        ----------
        alpha : float
            Confidence level, between 0 and 1. Default is 0.99.

        Returns
        -------
        float
            Mean of all simulated ceded losses that exceed VaR(alpha).
            Returns VaR(alpha) if no losses exceed the threshold.
        """
        var = self.var(alpha)
        tail = self.ceded_losses[self.ceded_losses > var]
        return float(tail.mean()) if len(tail) > 0 else var

    @property
    def tvar_99(self) -> float:
        """Convenience property for TVaR at 99% — used directly by TechnicalPricer."""
        return self.tvar(0.99)

    def prob_attachment(self) -> float:
        """
        Probability of attachment — fraction of years the treaty is triggered.

        A year attaches when at least one individual loss (XL) or the
        aggregate loss (Stop-Loss) exceeds the retention or attachment
        point, resulting in a positive ceded loss.

        In commercial terms: how often will the reinsurer receive a
        claim from the cedant in a given accident year?
        """
        return float((self.ceded_losses > 0).mean())

    def prob_exhaustion(self, treaty_limit: float) -> float:
        """
        Probability of exhaustion — fraction of years the full limit is used.

        A year exhausts when the total annual ceded loss equals or
        exceeds the treaty limit. High exhaustion probability signals
        that the layer is very exposed and should command a higher premium.

        In commercial terms: how often will the reinsurer pay the
        maximum possible amount in a given accident year?

        Parameters
        ----------
        treaty_limit : float
            The maximum annual ceded loss — the treaty limit for XL
            or the cap for Stop-Loss.
        """
        return float((self.ceded_losses >= treaty_limit).mean())

    def summary(self) -> str:
        """Return a formatted string of key simulation statistics."""
        lines = [
            f"Expected Ceded Loss : {self.expected_ceded_loss:>15,.0f}",
            f"Std Deviation       : {self.std_ceded_loss:>15,.0f}",
            f"VaR 99%             : {self.var(0.99):>15,.0f}",
            f"TVaR 99%            : {self.tvar_99:>15,.0f}",
            f"Prob of Attachment  : {self.prob_attachment():>15.1%}",
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

    def run(self) -> SimulationResults:
        """
        Execute the Monte Carlo simulation and return results.

        The simulation is structured differently depending on treaty type:

        For ExcessOfLoss:
            Loops year by year. For each year, draws individual losses
            and applies the treaty per occurrence. The annual ceded loss
            is the sum of per-occurrence ceded amounts. Cannot be fully
            vectorised because the treaty must be applied to each claim
            individually before summing.

        For StopLoss:
            Fully vectorised. All individual losses across all years are
            drawn in a single numpy call, then split into per-year chunks
            using numpy.split(). The treaty is then applied to each year's
            aggregate. This is significantly faster than looping year by
            year for large simulations.

        In both cases, years with zero claims are skipped and assigned
        a ceded loss of 0.0, avoiding unnecessary severity sampling.

        Returns
        -------
        SimulationResults
            Container with the full array of simulated annual ceded
            losses, ready for risk measure computation and pricing.

        Notes
        -----
        The vectorised Stop-Loss approach draws all losses at once using
        np.split() with cumulative claim counts as split indices. This
        avoids 100,000 individual Python loop iterations and reduces
        runtime from minutes to seconds for large simulations.
        """

        # Draw all claim counts for all years at once — vectorised
        claim_counts = self.frequency.sample(self.n_simulations, self.rng)

        # Pre-allocate output array — years with no claims stay at 0
        ceded_losses = np.zeros(self.n_simulations)

        if isinstance(self.treaty, ExcessOfLoss):
            for i, n in enumerate(claim_counts):
                if n == 0:
                    # no claims this year — reinsurer pays nothing
                    continue
                # draw individual losses, apply treaty per occurrence, sum
                losses = self.severity.sample(int(n), self.rng)
                ceded_losses[i] = self.treaty.apply(losses).sum()

        elif isinstance(self.treaty, StopLoss):
            total_claims = int(claim_counts.sum())
            if total_claims > 0:
                # Draw all individual losses across all years in one call.
                # Far faster than sampling year by year in a Python loop.
                all_losses = self.severity.sample(total_claims, self.rng)

                # Split the flat loss array into per-year chunks.
                # np.cumsum(claim_counts[:-1]) gives the split indices —
                # e.g. counts [3, 2, 4] → splits at [3, 5] →
                # chunks [losses[0:3], losses[3:5], losses[5:9]]
                splits = np.split(all_losses, np.cumsum(claim_counts[:-1]))

                for i, year_losses in enumerate(splits):
                    if len(year_losses) == 0:
                        # year had zero claims — ceded loss stays at 0
                        continue
                    # aggregate all losses for the year, then apply treaty
                    aggregate = year_losses.sum()
                    ceded_losses[i] = self.treaty.apply(aggregate)

        return SimulationResults(ceded_losses=ceded_losses)