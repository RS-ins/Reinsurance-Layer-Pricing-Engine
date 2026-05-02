"""
severity.py
-----------
Individual loss severity distributions for the Monte Carlo simulation.

Each class wraps a numpy distribution and exposes a consistent
interface: sample(n, rng) for drawing individual loss amounts and
an analytical mean() method for sanity checks.

All distributions are parameterised using actuarial conventions
(mean, cv) rather than the raw shape/scale parameters used internally
by numpy, so users can specify inputs they can estimate directly
from loss data.
"""

import math
import numpy as np


class LognormalSeverity:
    """
    Lognormal individual loss severity.

    The lognormal is the most widely used severity distribution in
    non-life insurance. It arises naturally when losses are the product
    of many independent multiplicative factors, and it produces a
    right-skewed distribution consistent with observed loss data.

    Parameterised on the log scale:
        log(X) ~ Normal(mu, sigma²)

    so mu and sigma are the mean and standard deviation of the
    underlying normal distribution, not of the losses themselves.

    Parameters
    ----------
    mu : float
        Mean of the underlying normal distribution (log-scale).
        A typical value for large commercial losses might be 10-12.
    sigma : float
        Standard deviation of the underlying normal distribution.
        Controls the heaviness of the tail. Must be positive.
        Higher sigma → heavier tail → more extreme losses.
    """

    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw n independent individual losses from the Lognormal distribution.

        Parameters
        ----------
        n : int
            Number of individual claims to simulate.
        rng : np.random.Generator
            Numpy random generator for reproducibility.

        Returns
        -------
        np.ndarray
            Array of shape (n,) containing simulated loss amounts.
        """
        return rng.lognormal(mean=self.mu, sigma=self.sigma, size=n)

    def mean(self) -> float:
        """
        Analytical mean of the lognormal distribution.

        E[X] = exp(mu + sigma²/2)

        Note this is the mean of the losses themselves, not of log(losses).
        """
        return math.exp(self.mu + 0.5 * self.sigma ** 2)


class GammaSeverity:
    """
    Gamma individual loss severity.

    The Gamma distribution is a flexible, right-skewed severity model.
    Unlike the lognormal it has a lighter tail, making it more appropriate
    for lines of business where extreme losses are less likely.

    Parameterised using actuarial conventions (mean, cv) rather than
    the raw shape/scale parameters used by numpy internally:
        shape (k) = 1 / cv²
        scale (θ) = mean × cv²

    so that:
        E[X]   = mean
        Var[X] = (cv × mean)²
        CV     = std / mean = cv

    Parameters
    ----------
    mean : float
        Expected individual loss amount. Must be positive.
    cv : float
        Coefficient of variation (std / mean). Must be positive.
        Higher cv → more dispersed losses → heavier tail.
    """

    def __init__(self, mean: float, cv: float):
        if mean <= 0:
            raise ValueError("mean must be positive")
        if cv <= 0:
            raise ValueError("cv must be positive")
        self.mean = mean
        self.cv = cv

    def _shape_scale(self):
        """
        Convert from actuarial (mean, cv) to numpy's (shape, scale).

        Derivation:
            mean  = shape × scale      → scale = mean / shape
            var   = shape × scale²     → shape = mean² / var = 1 / cv²
            scale = mean × cv²
        """
        shape = 1.0 / self.cv ** 2
        scale = self.mean * self.cv ** 2
        return shape, scale

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw n independent individual losses from the Gamma distribution.

        Parameters
        ----------
        n : int
            Number of individual claims to simulate.
        rng : np.random.Generator
            Numpy random generator for reproducibility.

        Returns
        -------
        np.ndarray
            Array of shape (n,) containing simulated loss amounts.
        """
        shape, scale = self._shape_scale()
        return rng.gamma(shape=shape, scale=scale, size=n)


class ParetoSeverity:
    """
    Single-parameter Pareto individual loss severity.

    The Pareto distribution is a heavy-tailed severity model used for
    catastrophic or very large loss modelling. Unlike the lognormal and
    gamma, it has a power-law tail, meaning extreme losses are
    significantly more likely.

    The tail index alpha controls how heavy the tail is:
        - alpha > 1  : finite mean (required)
        - alpha > 2  : finite variance
        - lower alpha → heavier tail → more extreme losses

    Parameterised by:
        alpha : tail index (shape)
        x_m   : minimum possible loss (scale) — all losses are >= x_m

    Sampling uses the inverse CDF method:
        X = x_m / U^(1/alpha)   where U ~ Uniform(0, 1)

    Parameters
    ----------
    alpha : float
        Tail index. Must be strictly greater than 1 for finite mean.
    x_m : float
        Minimum loss (scale parameter). Must be positive.
        Represents the smallest loss that enters the severity model,
        typically the policy deductible or a modelling threshold.
    """

    def __init__(self, alpha: float, x_m: float):
        if alpha <= 1:
            raise ValueError("alpha must be > 1 for finite mean")
        if x_m <= 0:
            raise ValueError("x_m must be positive")
        self.alpha = alpha
        self.x_m = x_m

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw n independent individual losses from the Pareto distribution.

        Uses the inverse CDF method:
            U ~ Uniform(0, 1)
            X = x_m / U^(1/alpha)

        This guarantees all sampled losses are >= x_m by construction.

        Parameters
        ----------
        n : int
            Number of individual claims to simulate.
        rng : np.random.Generator
            Numpy random generator for reproducibility.

        Returns
        -------
        np.ndarray
            Array of shape (n,) containing simulated loss amounts,
            all guaranteed to be >= x_m.
        """
        u = rng.uniform(size=n)
        return self.x_m / u ** (1.0 / self.alpha)

    def mean(self) -> float:
        """
        Analytical mean of the Pareto distribution.

        E[X] = alpha × x_m / (alpha - 1)

        Only finite when alpha > 1, which is enforced in __init__.
        """
        return self.alpha * self.x_m / (self.alpha - 1)