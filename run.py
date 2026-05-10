"""
run.py
------
Quick start script demonstrating the full Reinsurance Layer Pricing Engine
pipeline including Phase 4 features: AAL, AAD, and reinstatement premiums.

Run with:
    python run.py
"""

from reinsure_pricing.frequency import PoissonFrequency
from reinsure_pricing.severity import LognormalSeverity
from reinsure_pricing.treaties import ExcessOfLoss, StopLoss, ReinstatementProvision
from reinsure_pricing.simulation import MonteCarloEngine
from reinsure_pricing.risk_measures import compute_risk_measures
from reinsure_pricing.pricing import TechnicalPricer
from reinsure_pricing.plots import plot_ceded_loss_distribution, plot_sensitivity
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. Basic XL — no advanced features
# ─────────────────────────────────────────────
print("=" * 55)
print("EXAMPLE 1 — Basic 5M xs 1M XL Layer")
print("=" * 55)

frequency = PoissonFrequency(lambda_=120)
severity  = LognormalSeverity(mu=10.5, sigma=1.2)
treaty    = ExcessOfLoss(retention=1_000_000, limit=5_000_000)

engine  = MonteCarloEngine(frequency, severity, treaty,
                           n_simulations=100_000, random_state=42)
results = engine.run()
rm      = compute_risk_measures(results, treaty_limit=treaty.limit)
print(rm.summary())

pricer = TechnicalPricer(
    expected_ceded_loss=results.expected_ceded_loss,
    tvar_99=results.tvar_99,
    expense_load=0.05,
    profit_load=0.08,
    cost_of_capital=0.10,
)
pricing = pricer.price(treaty_limit=treaty.limit)
print(pricing.summary())

# ─────────────────────────────────────────────
# 2. XL with AAL
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("EXAMPLE 2 — 5M xs 1M XL with 15M Aggregate Annual Limit")
print("=" * 55)

treaty_aal = ExcessOfLoss(
    retention=1_000_000,
    limit=5_000_000,
    aggregate_limit=15_000_000,
)
engine_aal  = MonteCarloEngine(frequency, severity, treaty_aal,
                               n_simulations=100_000, random_state=42)
results_aal = engine_aal.run()
rm_aal      = compute_risk_measures(results_aal, treaty_limit=treaty_aal.limit)
print(rm_aal.summary())

pricer_aal = TechnicalPricer(
    expected_ceded_loss=results_aal.expected_ceded_loss,
    tvar_99=results_aal.tvar_99,
    expense_load=0.05,
    profit_load=0.08,
    cost_of_capital=0.10,
)
print(pricer_aal.price(treaty_limit=treaty_aal.limit).summary())

# ─────────────────────────────────────────────
# 3. XL with AAD
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("EXAMPLE 3 — 5M xs 1M XL with 500K Annual Aggregate Deductible")
print("=" * 55)

treaty_aad = ExcessOfLoss(
    retention=1_000_000,
    limit=5_000_000,
    aggregate_deductible=500_000,
)
engine_aad  = MonteCarloEngine(frequency, severity, treaty_aad,
                               n_simulations=100_000, random_state=42)
results_aad = engine_aad.run()
rm_aad      = compute_risk_measures(results_aad, treaty_limit=treaty_aad.limit)
print(rm_aad.summary())

pricer_aad = TechnicalPricer(
    expected_ceded_loss=results_aad.expected_ceded_loss,
    tvar_99=results_aad.tvar_99,
    expense_load=0.05,
    profit_load=0.08,
    cost_of_capital=0.10,
)
print(pricer_aad.price(treaty_limit=treaty_aad.limit).summary())

# ─────────────────────────────────────────────
# 4. XL with Reinstatement Premiums
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("EXAMPLE 4 — 5M xs 1M XL with 1 Free + 1 Paid Reinstatement")
print("=" * 55)

freq_heavy = PoissonFrequency(lambda_=120)
sev_heavy  = LognormalSeverity(mu=11.0, sigma=1.2)
treaty_ri  = ExcessOfLoss(retention=1_000_000, limit=5_000_000)

rp = ReinstatementProvision(
    n_free=1,
    n_paid=1,
    original_premium=300_000,
)

engine_ri  = MonteCarloEngine(freq_heavy, sev_heavy, treaty_ri,
                              n_simulations=100_000, random_state=42)
results_ri = engine_ri.run(reinstatement=rp)
rm_ri      = compute_risk_measures(results_ri, treaty_limit=treaty_ri.limit)
print(rm_ri.summary())

pricer_ri = TechnicalPricer(
    expected_ceded_loss=results_ri.expected_ceded_loss,
    tvar_99=results_ri.tvar_99,
    expense_load=0.05,
    profit_load=0.08,
    cost_of_capital=0.10,
    expected_reinstatement_premium=results_ri.expected_reinstatement_premium,
)
print(pricer_ri.price(treaty_limit=treaty_ri.limit).summary())
print(f"\nGross ECL            : {results_ri.expected_ceded_loss:>15,.0f}")
print(f"Exp Reinst Premium   : {results_ri.expected_reinstatement_premium:>15,.0f}")
print(f"Net Expected Recovery: {results_ri.net_expected_recovery:>15,.0f}")

# ─────────────────────────────────────────────
# 5. Stop-Loss
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("EXAMPLE 5 — 20M xs 10M Stop-Loss")
print("=" * 55)

treaty_sl  = StopLoss(attachment=10_000_000, cap=20_000_000)
engine_sl  = MonteCarloEngine(frequency, severity, treaty_sl,
                              n_simulations=100_000, random_state=42)
results_sl = engine_sl.run()
rm_sl      = compute_risk_measures(results_sl, treaty_limit=treaty_sl.cap)
print(rm_sl.summary())

pricer_sl = TechnicalPricer(
    expected_ceded_loss=results_sl.expected_ceded_loss,
    tvar_99=results_sl.tvar_99,
    expense_load=0.05,
    profit_load=0.08,
    cost_of_capital=0.10,
)
print(pricer_sl.price(treaty_limit=treaty_sl.cap).summary())

# ─────────────────────────────────────────────
# 6. Bootstrapped Confidence Intervals
#    Applied to Example 1 (basic 5M xs 1M XL)
#    to show how stable the risk measure
#    estimates are given 100,000 simulations.
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("EXAMPLE 6 — Bootstrapped Confidence Intervals")
print("Applied to: 5M xs 1M XL | Poisson(λ=120) | Lognormal(μ=10.5, σ=1.2)")
print("Question: how stable are our risk measures with 100,000 simulations?")
print("=" * 55)

from reinsure_pricing.bootstrap import bootstrap_risk_measures

boot = bootstrap_risk_measures(
    results,                       # from Example 1 — basic 5M xs 1M XL
    treaty_limit=treaty.limit,     # 5,000,000
    n_bootstrap=1_000,
    confidence_level=0.95,
)
print(boot.summary())
print("\nInterpretation: Rel Width < 5% = stable estimate.")
print("If Rel Width > 10%, consider increasing n_simulations.")

# ─────────────────────────────────────────────
# 7. Plots
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("Generating plots...")
print("=" * 55)

# Ceded loss distribution — basic XL
plot_ceded_loss_distribution(
    results=results,
    risk_measures=rm,
    treaty_limit=treaty.limit,
    title="5M xs 1M XL Layer — Ceded Loss Distribution",
)
plt.show()

# Sensitivity — vary retention
retentions = [500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000]
premiums_list, ecls_list = [], []

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

# ─────────────────────────────────────────────
# 7. Export report to Excel
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("EXAMPLE 7 — Export to Excel")
print("=" * 55)

from reinsure_pricing.io import export_report

export_report(
    results=results,
    rm=rm,
    pricing=pricing,
    treaty=treaty,
    frequency=frequency,
    severity=severity,
    bootstrap=boot,
)

# ─────────────────────────────────────────────
# 8. Distribution Fitting
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("EXAMPLE 8 — Distribution Fitting")
print("=" * 55)

from reinsure_pricing.fitting import fit_frequency, fit_severity
import numpy as np

# Simulate some historical data to fit
rng_hist       = np.random.default_rng(99)
hist_counts    = rng_hist.poisson(lam=115, size=15).tolist()
hist_losses    = rng_hist.lognormal(mean=10.8, sigma=1.1, size=300).tolist()

print("\nFitting frequency distribution to 15 years of claim counts...")
freq_comp = fit_frequency(hist_counts)
print(freq_comp.summary())

print("\nFitting severity distribution to 300 individual losses...")
sev_comp = fit_severity(hist_losses)
print(sev_comp.summary())

print("\nUsing fitted distributions to price the treaty...")
engine_fitted = MonteCarloEngine(
    frequency=freq_comp.best.distribution,
    severity=sev_comp.best.distribution,
    treaty=treaty,
    n_simulations=100_000,
    random_state=42,
)
results_fitted = engine_fitted.run()
rm_fitted      = compute_risk_measures(results_fitted, treaty_limit=treaty.limit)
pricer_fitted  = TechnicalPricer(
    expected_ceded_loss=results_fitted.expected_ceded_loss,
    tvar_99=results_fitted.tvar_99,
)
print(f"\nFitted model technical premium : "
      f"{pricer_fitted.technical_premium():,.0f}")
print(f"Rate on Line                   : "
      f"{pricer_fitted.rate_on_line(treaty.limit):.2%}")