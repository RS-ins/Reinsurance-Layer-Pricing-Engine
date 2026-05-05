import pytest
from reinsure_pricing.frequency import PoissonFrequency
from reinsure_pricing.severity import LognormalSeverity
from reinsure_pricing.treaties import ExcessOfLoss
from reinsure_pricing.simulation import MonteCarloEngine
from reinsure_pricing.risk_measures import compute_risk_measures
from reinsure_pricing.bootstrap import bootstrap_risk_measures, BootstrappedRiskMeasures


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

def test_bootstrap_returns_correct_type():
    results, treaty = get_results()
    boot = bootstrap_risk_measures(results, treaty_limit=treaty.limit,
                                   n_bootstrap=100)
    assert isinstance(boot, BootstrappedRiskMeasures)

def test_bootstrap_ci_contains_point_estimate():
    results, treaty = get_results()
    boot = bootstrap_risk_measures(results, treaty_limit=treaty.limit,
                                   n_bootstrap=200)
    for bi in [boot.ecl, boot.var_99, boot.tvar_99]:
        assert bi.ci_lower <= bi.point_estimate <= bi.ci_upper

def test_bootstrap_ci_ordering():
    results, treaty = get_results()
    boot = bootstrap_risk_measures(results, treaty_limit=treaty.limit,
                                   n_bootstrap=200)
    for bi in [boot.ecl, boot.var_95, boot.var_99, boot.var_995,
               boot.tvar_95, boot.tvar_99, boot.tvar_995,
               boot.prob_attachment, boot.prob_exhaustion]:
        assert bi.ci_lower <= bi.ci_upper

def test_bootstrap_var_ordering():
    results, treaty = get_results()
    boot = bootstrap_risk_measures(results, treaty_limit=treaty.limit,
                                   n_bootstrap=200)
    assert boot.var_95.point_estimate <= boot.var_99.point_estimate <= boot.var_995.point_estimate