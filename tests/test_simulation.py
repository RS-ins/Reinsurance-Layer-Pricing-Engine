import numpy as np
import pytest
from reinsure_pricing.frequency import PoissonFrequency
from reinsure_pricing.severity import LognormalSeverity
from reinsure_pricing.treaties import ExcessOfLoss, StopLoss
from reinsure_pricing.simulation import MonteCarloEngine, SimulationResults


FREQ = PoissonFrequency(lambda_=120)
SEV = LognormalSeverity(mu=10.5, sigma=1.2)
XL = ExcessOfLoss(retention=1_000_000, limit=5_000_000)
SL = StopLoss(attachment=50_000_000, cap=20_000_000)


def test_xl_returns_results():
    engine = MonteCarloEngine(FREQ, SEV, XL, n_simulations=10_000)
    results = engine.run()
    assert isinstance(results, SimulationResults)
    assert len(results.ceded_losses) == 10_000


def test_ceded_losses_non_negative():
    engine = MonteCarloEngine(FREQ, SEV, XL, n_simulations=10_000)
    results = engine.run()
    assert (results.ceded_losses >= 0).all()


def test_single_claim_capped_at_limit():
    import numpy as np
    
    # Create a 5M xs 1M treaty
    # reinsurer pays losses between 1M and 6M, capped at 5M per occurrence
    treaty = ExcessOfLoss(retention=1_000_000, limit=5_000_000)
    
    # Apply the treaty to a single massive claim (~1 billion)
    # this is way above the exhaustion point (1M + 5M = 6M)
    # so the reinsurer should pay exactly the limit: 5M
    result = treaty.apply(np.array([999_999_999.0]))
    
    # result is an array, so we take the first (and only) element
    # and verify it equals exactly 5M — not more
    assert result[0] == 5_000_000.0


def test_stop_loss_runs():
    engine = MonteCarloEngine(FREQ, SEV, SL, n_simulations=10_000)
    results = engine.run()
    assert len(results.ceded_losses) == 10_000


def test_tvar_geq_var():
    engine = MonteCarloEngine(FREQ, SEV, XL, n_simulations=50_000)
    results = engine.run()
    assert results.tvar_99 >= results.var(0.99)


def test_prob_attachment_between_0_and_1():
    engine = MonteCarloEngine(FREQ, SEV, XL, n_simulations=10_000)
    results = engine.run()
    assert 0 <= results.prob_attachment() <= 1


def test_rejects_zero_simulations():
    with pytest.raises(ValueError):
        MonteCarloEngine(FREQ, SEV, XL, n_simulations=0)