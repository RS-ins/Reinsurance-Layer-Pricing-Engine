import numpy as np
from scipy import stats


class PoissonFrequency: #Generating claims from a Poisson distribution
    """Poisson claim count distribution."""

    def __init__(self, lambda_: float):
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")
        self.lambda_ = lambda_

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n claim counts."""
        return rng.poisson(lam=self.lambda_, size=n)

    def mean(self) -> float:
        return self.lambda_

    def variance(self) -> float:
        return self.lambda_


class NegativeBinomialFrequency: 
    """
    Negative Binomial claim count distribution.
    Parameterised by mean (mu) and overdispersion (phi),
    where variance = mu + phi * mu^2.
    """

    def __init__(self, mu: float, phi: float):
        if mu <= 0:
            raise ValueError("mu must be positive")
        if phi <= 0:
            raise ValueError("phi must be positive")
        self.mu = mu
        self.phi = phi

    def _scipy_params(self):
        # scipy uses (n, p) parameterisation
        n = 1.0 / self.phi
        p = n / (n + self.mu)
        return n, p

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n claim counts."""
        n_param, p_param = self._scipy_params()
        return rng.negative_binomial(n=n_param, p=p_param, size=n)

    def mean(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.mu + self.phi * self.mu ** 2