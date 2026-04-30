import numpy as np
from dataclasses import dataclass
from reinsure_pricing.frequency import PoissonFrequency, NegativeBinomialFrequency
from reinsure_pricing.severity import LognormalSeverity, GammaSeverity, ParetoSeverity
from reinsure_pricing.treaties import ExcessOfLoss, StopLoss


@dataclass
class SimulationResults:
    ceded_losses: np.ndarray  # array of ceded losses, one per simulated year

    @property #let the defined function be called without parenthesis
    def expected_ceded_loss(self) -> float:
        return float(self.ceded_losses.mean())

    @property
    def std_ceded_loss(self) -> float:
        return float(self.ceded_losses.std())

    def var(self, alpha: float = 0.99) -> float: #Value at Risk computation
        return float(np.quantile(self.ceded_losses, alpha))

    def tvar(self, alpha: float = 0.99) -> float: #Tail Value at Risk (What's the average loss in the worst 1% cases, when alpha = 99%)
        var = self.var(alpha)
        tail = self.ceded_losses[self.ceded_losses > var]
        return float(tail.mean()) if len(tail) > 0 else var

    @property
    def tvar_99(self) -> float:
        return self.tvar(0.99)

    def prob_attachment(self) -> float: #Attatchment means how often the treaty gets hit
        return float((self.ceded_losses > 0).mean())

    def prob_exhaustion(self, treaty_limit: float) -> float: #Exhaustion means how often the treaty gets maxed out
        return float((self.ceded_losses >= treaty_limit).mean())

    def summary(self) -> str:
        lines = [
            f"Expected Ceded Loss : {self.expected_ceded_loss:>15,.0f}",
            f"Std Deviation       : {self.std_ceded_loss:>15,.0f}",
            f"VaR 99%             : {self.var(0.99):>15,.0f}",
            f"TVaR 99%            : {self.tvar_99:>15,.0f}",
            f"Prob of Attachment  : {self.prob_attachment():>15.1%}", 
        ]
        return "\n".join(lines)


class MonteCarloEngine:

    def __init__(self, frequency, severity, treaty, n_simulations: int = 100_000,
                 random_state: int = 42):
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        self.frequency = frequency
        self.severity = severity
        self.treaty = treaty
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_state)

    def run(self) -> SimulationResults:
        claim_counts = self.frequency.sample(self.n_simulations, self.rng)
        ceded_losses = np.zeros(self.n_simulations)

        if isinstance(self.treaty, ExcessOfLoss):
            for i, n in enumerate(claim_counts):
                if n == 0:
                    continue
                losses = self.severity.sample(int(n), self.rng)
                ceded_losses[i] = self.treaty.apply(losses).sum()

        elif isinstance(self.treaty, StopLoss):
            for i, n in enumerate(claim_counts):
                if n == 0:
                    continue
                losses = self.severity.sample(int(n), self.rng)
                aggregate = losses.sum()
                ceded_losses[i] = self.treaty.apply(aggregate)

        return SimulationResults(ceded_losses=ceded_losses)