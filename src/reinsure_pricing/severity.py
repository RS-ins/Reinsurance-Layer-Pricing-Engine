import numpy as np


class LognormalSeverity:
    """
    Lognormal individual loss severity.
    mu and sigma are the mean and std of the underlying normal (log-scale).
    """

    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(mean=self.mu, sigma=self.sigma, size=n)

    def mean(self) -> float:
        import math
        return math.exp(self.mu + 0.5 * self.sigma ** 2)


class GammaSeverity:
    """
    Gamma individual loss severity.
    Parameterised by mean and coefficient of variation (cv).
    """

    def __init__(self, mean: float, cv: float):
        if mean <= 0:
            raise ValueError("mean must be positive")
        if cv <= 0:
            raise ValueError("cv must be positive")
        self.mean = mean
        self.cv = cv

    def _shape_scale(self):
        shape = 1.0 / self.cv ** 2
        scale = self.mean * self.cv ** 2
        return shape, scale

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        shape, scale = self._shape_scale()
        return rng.gamma(shape=shape, scale=scale, size=n)


class ParetoSeverity:
    """
    Single-parameter Pareto severity (heavy-tailed).
    alpha: shape (tail index), x_m: minimum loss (scale).
    """

    def __init__(self, alpha: float, x_m: float):
        if alpha <= 1:
            raise ValueError("alpha must be > 1 for finite mean")
        if x_m <= 0:
            raise ValueError("x_m must be positive")
        self.alpha = alpha
        self.x_m = x_m

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # Inverse CDF method: X = x_m / U^(1/alpha)
        u = rng.uniform(size=n)
        return self.x_m / u ** (1.0 / self.alpha)

    def mean(self) -> float:
        return self.alpha * self.x_m / (self.alpha - 1)