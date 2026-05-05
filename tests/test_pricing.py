import pytest
from reinsure_pricing.frequency import PoissonFrequency
from reinsure_pricing.severity import LognormalSeverity
from reinsure_pricing.treaties import ExcessOfLoss
from reinsure_pricing.simulation import MonteCarloEngine
from reinsure_pricing.pricing import TechnicalPricer
from reinsure_pricing.treaties import ExcessOfLoss, ReinstatementProvision


def get_pricer():
    freq    = PoissonFrequency(lambda_=120)
    sev     = LognormalSeverity(mu=10.5, sigma=1.2)
    treaty  = ExcessOfLoss(retention=1_000_000, limit=5_000_000)
    engine  = MonteCarloEngine(freq, sev, treaty, n_simulations=50_000)
    results = engine.run()
    return TechnicalPricer(
        expected_ceded_loss=results.expected_ceded_loss,
        tvar_99=results.tvar_99,
        expense_load=0.05,
        profit_load=0.08,
        cost_of_capital=0.10
    )


def test_premium_above_ecl():
    pricer = get_pricer()
    assert pricer.technical_premium() > pricer.expected_ceded_loss


def test_rate_on_line_between_0_and_1():
    pricer = get_pricer()
    assert 0 < pricer.rate_on_line(treaty_limit=5_000_000) < 1


def test_premium_components_add_up():
    pricer = get_pricer()
    p = pricer.price(treaty_limit=5_000_000)
    total = p.expected_ceded_loss + p.expense_loading + p.profit_loading + p.capital_load
    assert abs(total - p.technical_premium) < 1


def test_rejects_bad_expense_load():
    with pytest.raises(ValueError):
        TechnicalPricer(expected_ceded_loss=1_000_000, tvar_99=2_000_000,
                        expense_load=1.5)


def test_higher_cost_of_capital_increases_premium():
    freq    = PoissonFrequency(lambda_=120)
    sev     = LognormalSeverity(mu=10.5, sigma=1.2)
    treaty  = ExcessOfLoss(retention=1_000_000, limit=5_000_000)
    engine  = MonteCarloEngine(freq, sev, treaty, n_simulations=50_000)
    results = engine.run()

    p_low  = TechnicalPricer(results.expected_ceded_loss, results.tvar_99,
                              cost_of_capital=0.05).technical_premium()
    p_high = TechnicalPricer(results.expected_ceded_loss, results.tvar_99,
                              cost_of_capital=0.20).technical_premium()
    assert p_high > p_low

def test_reinstatement_reduces_premium():
    freq    = PoissonFrequency(lambda_=200)
    sev     = LognormalSeverity(mu=12.0, sigma=2.0) 
    treaty  = ExcessOfLoss(retention=500_000, limit=2_000_000)
    engine  = MonteCarloEngine(freq, sev, treaty, n_simulations=50_000)
    rp      = ReinstatementProvision(n_free=1, n_paid=1,
                                     original_premium=500_000)
    results = engine.run(reinstatement=rp)

    # verify reinstatements actually occurred
    assert results.expected_reinstatement_premium > 0, \
        "No reinstatements fired — adjust parameters"

    pricer_gross = TechnicalPricer(
        expected_ceded_loss=results.expected_ceded_loss,
        tvar_99=results.tvar_99,
    )
    pricer_net = TechnicalPricer(
        expected_ceded_loss=results.expected_ceded_loss,
        tvar_99=results.tvar_99,
        expected_reinstatement_premium=results.expected_reinstatement_premium,
    )
    assert pricer_net.technical_premium() < pricer_gross.technical_premium()


def test_net_ecl_never_negative():
    # even if reinstatement premium exceeds ECL, net ECL should be 0
    pricer = TechnicalPricer(
        expected_ceded_loss=100_000,
        tvar_99=500_000,
        expected_reinstatement_premium=200_000,
    )
    assert pricer.net_ecl >= 0


def test_pricing_result_components_add_up_with_reinstatement():
    pricer = TechnicalPricer(
        expected_ceded_loss=500_000,
        tvar_99=2_000_000,
        expense_load=0.05,
        profit_load=0.08,
        cost_of_capital=0.10,
        expected_reinstatement_premium=50_000,
    )
    p = pricer.price(treaty_limit=5_000_000)
    total = p.net_expected_ceded_loss + p.expense_loading + \
            p.profit_loading + p.capital_load
    assert abs(total - p.technical_premium) < 1