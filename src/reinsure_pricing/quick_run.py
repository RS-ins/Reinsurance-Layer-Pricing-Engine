from reinsure_pricing.frequency import PoissonFrequency
from reinsure_pricing.severity import LognormalSeverity
from reinsure_pricing.treaties import ExcessOfLoss
from reinsure_pricing.simulation import MonteCarloEngine
from reinsure_pricing.pricing import TechnicalPricer

# Define frequency and severity distributions
frequency = PoissonFrequency(lambda_=120)
severity  = LognormalSeverity(mu=10.5, sigma=1.2)

# Define the reinsurance treaty: 5M xs 1M per occurrence
treaty = ExcessOfLoss(retention=1_000_000, limit=5_000_000)

# Run the Monte Carlo simulation
engine = MonteCarloEngine(
    frequency=frequency,
    severity=severity,
    treaty=treaty,
    n_simulations=100_000,
    random_state=42
)
results = engine.run()

# Compute the technical premium
pricer = TechnicalPricer(
    expected_ceded_loss=results.expected_ceded_loss,
    tvar_99=results.tvar_99,
    expense_load=0.05,
    profit_load=0.08,
    cost_of_capital=0.10
)

premium = pricer.technical_premium()

print(results.summary())
print(f"Expected Ceded Loss : {pricer.expected_ceded_loss:,.0f}")
print(f"Technical Premium   : {premium:,.0f}")
print(f"Rate on Line        : {pricer.rate_on_line(treaty_limit=5_000_000):.2%}")