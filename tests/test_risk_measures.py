import pytest
from reinsure_pricing.frequency import PoissonFrequency
from reinsure_pricing.severity import LognormalSeverity
from reinsure_pricing.treaties import ExcessOfLoss
from reinsure_pricing.simulation import MonteCarloEngine
from reinsure_pricing.risk_measures import compute_risk_measures


def get_results():
    freq   = PoissonFrequency(lambda_=120)
    sev    = LognormalSeverity(mu=10.5, sigma=1.2)
    treaty = ExcessOfLoss(retention=1_000_000, limit=5_000_000)
    engine = MonteCarloEngine(freq, sev, treaty, n_simulations=50_000)
    return engine.run(), treaty


def test_var_ordering():
    results, treaty = get_results()
    rm = compute_risk_measures(results, treaty.limit)
    assert rm.var_95 <= rm.var_99 <= rm.var_995


def test_tvar_ordering():
    results, treaty = get_results()
    rm = compute_risk_measures(results, treaty.limit)
    assert rm.tvar_95 <= rm.tvar_99 <= rm.tvar_995


def test_tvar_geq_var():
    results, treaty = get_results()
    rm = compute_risk_measures(results, treaty.limit)
    assert rm.tvar_99 >= rm.var_99


def test_probabilities_between_0_and_1():
    results, treaty = get_results()
    rm = compute_risk_measures(results, treaty.limit)
    assert 0 <= rm.prob_attachment <= 1
    assert 0 <= rm.prob_exhaustion <= 1


def test_exhaustion_leq_attachment():
    # can't exhaust without attaching first
    results, treaty = get_results()
    rm = compute_risk_measures(results, treaty.limit)
    assert rm.prob_exhaustion <= rm.prob_attachment


def test_cv_positive():
    results, treaty = get_results()
    rm = compute_risk_measures(results, treaty.limit)
    assert rm.coefficient_of_variation > 0