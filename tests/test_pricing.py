import pytest
from reinsure_pricing.frequency import PoissonFrequency
from reinsure_pricing.severity import LognormalSeverity
from reinsure_pricing.treaties import ExcessOfLoss
from reinsure_pricing.simulation import MonteCarloEngine
from reinsure_pricing.pricing import TechnicalPricer


def get_results():
    freq = PoissonFrequency(lambda_=120)
    sev = LognormalSeverity(mu=10.5, sigma=1.2)
    treaty = ExcessOfLoss(retention=1_000_000, limit=5_000_000)
    engine = MonteCarloEngine(freq, sev, treaty, n_simulations=50_000)
    return engine.run(), treaty


def test_premium_above_ecl():
    results, treaty = get_results()
    pricer = TechnicalPricer()
    pricing = pricer.price(results, treaty_limit=treaty.limit)
    assert pricing.technical_premium > pricing.expected_ceded_loss


def test_rate_on_line_between_0_and_1():
    results, treaty = get_results()
    pricer = TechnicalPricer()
    pricing = pricer.price(results, treaty_limit=treaty.limit)
    assert 0 < pricing.rate_on_line < 1


def test_loss_ratio_between_0_and_1():
    results, treaty = get_results()
    pricer = TechnicalPricer()
    pricing = pricer.price(results, treaty_limit=treaty.limit)
    assert 0 < pricing.loss_ratio_at_technical < 1


def test_premium_components_add_up():
    results, treaty = get_results()
    pricer = TechnicalPricer()
    p = pricer.price(results, treaty_limit=treaty.limit)
    total = p.expected_ceded_loss + p.risk_margin + p.expense_loading
    assert abs(total - p.technical_premium) < 1  # within €1 rounding


def test_rejects_bad_expense_ratio():
    with pytest.raises(ValueError):
        TechnicalPricer(expense_ratio=1.5)


def test_higher_tvar_multiplier_increases_premium():
    results, treaty = get_results()
    p_low = TechnicalPricer(tvar_multiplier=0.1).price(results, treaty.limit)
    p_high = TechnicalPricer(tvar_multiplier=0.9).price(results, treaty.limit)
    assert p_high.technical_premium > p_low.technical_premium