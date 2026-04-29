import numpy as np
import pytest
from reinsure_pricing.severity import LognormalSeverity, GammaSeverity, ParetoSeverity

RNG = np.random.default_rng(42)


def test_lognormal_mean():
    sev = LognormalSeverity(mu=10.5, sigma=1.2)
    samples = sev.sample(200_000, RNG)
    assert abs(samples.mean() - sev.mean()) / sev.mean() < 0.02  # within 2%


def test_lognormal_rejects_bad_sigma():
    with pytest.raises(ValueError):
        LognormalSeverity(mu=10.5, sigma=-1)


def test_gamma_mean():
    sev = GammaSeverity(mean=500_000, cv=1.5)
    samples = sev.sample(200_000, RNG)
    assert abs(samples.mean() - 500_000) / 500_000 < 0.02


def test_pareto_mean():
    sev = ParetoSeverity(alpha=3.0, x_m=100_000)
    samples = sev.sample(500_000, RNG)
    assert abs(samples.mean() - sev.mean()) / sev.mean() < 0.02


def test_pareto_rejects_alpha_leq_1():
    with pytest.raises(ValueError):
        ParetoSeverity(alpha=0.8, x_m=100_000)


def test_pareto_all_above_xm():
    sev = ParetoSeverity(alpha=2.0, x_m=50_000)
    samples = sev.sample(10_000, RNG)
    assert (samples >= 50_000).all()