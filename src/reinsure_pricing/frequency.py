"""
frequency.py
------------
Claim count (frequency) distributions for the Monte Carlo simulation.

Each class wraps a scipy/numpy distribution and exposes a consistent
interface: sample(n, rng) for drawing claim counts and analytical
mean() / variance() methods for quick sanity checks.
"""

import numpy as np


class PoissonFrequency:
    """
    Poisson claim count distribution.

    The Poisson distribution is the standard starting point for claim
    frequency modelling. It assumes that claims arrive independently
    at a constant rate, and that the mean and variance are equal
    (equidispersion). If the data shows overdispersion (variance > mean),
    use NegativeBinomialFrequency instead.

    Parameters
    ----------
    lambda_ : float
        Expected number of claims per accident year, must be positive.
    """

    def __init__(self, lambda_: float):
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")
        self.lambda_ = lambda_

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw n independent claim counts from the Poisson distribution.

        Parameters
        ----------
        n : int
            Number of accident years to simulate.
        rng : np.random.Generator
            Numpy random generator — pass the engine's shared generator
            to ensure reproducibility across all modules.

        Returns
        -------
        np.ndarray
            Integer array of shape (n,) containing simulated claim counts.
        """
        return rng.poisson(lam=self.lambda_, size=n)

    def mean(self) -> float:
        """Analytical mean of the distribution. Equal to lambda_."""
        return self.lambda_

    def variance(self) -> float:
        """Analytical variance of the distribution. Equal to lambda_ (equidispersion)."""
        return self.lambda_


class NegativeBinomialFrequency:
    """
    Negative Binomial claim count distribution.

    Extends the Poisson by allowing overdispersion — variance exceeds
    the mean. This is common in real portfolios where heterogeneity
    across insureds inflates the variance beyond what a Poisson
    would predict.

    Parameterised by the actuarial (mu, phi) convention:
        mean     = mu
        variance = mu + phi * mu²

    where phi is the overdispersion parameter. As phi → 0 the
    distribution converges to a Poisson(mu).

    Parameters
    ----------
    mu : float
        Expected number of claims per accident year. Must be positive.
    phi : float
        Overdispersion parameter. Must be positive. Higher values
        produce heavier tails in the claim count distribution.
    """

    def __init__(self, mu: float, phi: float):
        if mu <= 0:
            raise ValueError("mu must be positive")
        if phi <= 0:
            raise ValueError("phi must be positive")
        self.mu = mu
        self.phi = phi

    def _scipy_params(self):
        """
        Convert from actuarial (mu, phi) to numpy's (n, p) parameterisation.

        Numpy's negative binomial uses:
            n = number of successes (shape)
            p = probability of success

        The mapping is:
            n = 1 / phi
            p = n / (n + mu)
        """
        n = 1.0 / self.phi
        p = n / (n + self.mu)
        return n, p

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw n independent claim counts from the Negative Binomial distribution.

        Parameters
        ----------
        n : int
            Number of accident years to simulate.
        rng : np.random.Generator
            Numpy random generator for reproducibility.

        Returns
        -------
        np.ndarray
            Integer array of shape (n,) containing simulated claim counts.
        """
        n_param, p_param = self._scipy_params()
        return rng.negative_binomial(n=n_param, p=p_param, size=n)

    def mean(self) -> float:
        """Analytical mean of the distribution. Equal to mu."""
        return self.mu

    def variance(self) -> float:
        """
        Analytical variance of the distribution.
        Always greater than the mean due to overdispersion.
        """
        return self.mu + self.phi * self.mu ** 2