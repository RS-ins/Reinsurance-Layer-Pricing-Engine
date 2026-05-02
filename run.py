from reinsure_pricing.frequency import PoissonFrequency
from reinsure_pricing.severity import LognormalSeverity
from reinsure_pricing.treaties import ExcessOfLoss
from reinsure_pricing.simulation import MonteCarloEngine
from reinsure_pricing.pricing import TechnicalPricer
from reinsure_pricing.risk_measures import compute_risk_measures
from reinsure_pricing.plots import plot_ceded_loss_distribution, plot_sensitivity
import matplotlib.pyplot as plt

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

# Compute full risk measures
rm = compute_risk_measures(results, treaty_limit=treaty.limit)
print(rm.summary())

# Compute the technical premium
pricer = TechnicalPricer(
    expected_ceded_loss=results.expected_ceded_loss,
    tvar_99=results.tvar_99,
    expense_load=0.05,
    profit_load=0.08,
    cost_of_capital=0.10
)
print(f"Technical Premium : {pricer.technical_premium():,.0f}")
print(f"Rate on Line      : {pricer.rate_on_line(treaty_limit=treaty.limit):.2%}")

# Plot 1 — ceded loss distribution
plot_ceded_loss_distribution(
    results=results,
    risk_measures=rm,
    treaty_limit=treaty.limit,
    title="5M xs 1M XL Layer — Ceded Loss Distribution",
)

# Plot 2 — sensitivity: vary retention
retentions = [500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000]
premiums_list = []
ecls_list = []

for ret in retentions:
    t = ExcessOfLoss(retention=ret, limit=5_000_000)
    e = MonteCarloEngine(frequency, severity, t,
                         n_simulations=50_000, random_state=42)
    r = e.run()
    p = TechnicalPricer(
        expected_ceded_loss=r.expected_ceded_loss,
        tvar_99=r.tvar_99,
        expense_load=0.05,
        profit_load=0.08,
        cost_of_capital=0.10,
    )
    premiums_list.append(p.technical_premium())
    ecls_list.append(r.expected_ceded_loss)

plot_sensitivity(
    parameter_values=[r/1e6 for r in retentions],
    premiums=premiums_list,
    ecls=ecls_list,
    parameter_name="Retention (M€)",
)
plt.show()