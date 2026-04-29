import numpy as np
import pytest
from reinsure_pricing.frequency import PoissonFrequency, NegativeBinomialFrequency

RNG = np.random.default_rng(42)


def test_poisson_mean():
    freq = PoissonFrequency(lambda_=100)
    samples = freq.sample(100_000, RNG)
    assert abs(samples.mean() - 100) < 1  # should be very close


def test_poisson_rejects_bad_params():
    with pytest.raises(ValueError):
        PoissonFrequency(lambda_=-1)


def test_nb_mean():
    freq = NegativeBinomialFrequency(mu=50, phi=0.1)
    samples = freq.sample(100_000, RNG)
    assert abs(samples.mean() - 50) < 1


def test_nb_overdispersion():
    # NB variance should exceed mean (unlike Poisson)
    freq = NegativeBinomialFrequency(mu=50, phi=0.1)
    assert freq.variance() > freq.mean()