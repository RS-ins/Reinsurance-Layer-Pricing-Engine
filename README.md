# Reinsurance Layer Pricing Engine

> A Python Monte Carlo engine for pricing Excess-of-Loss and Stop-Loss reinsurance layers, estimating ceded loss distributions, risk measures, and capital-loaded technical premiums.

[![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Educational](https://img.shields.io/badge/purpose-educational-green.svg)]()

---

## Live Dashboard (v0.2.0)

An interactive dashboard with 11250 pre-computed scenarios is available online — no installation required:

**[→ Open Dashboard](https://rs-ins.github.io/Reinsurance-Layer-Pricing-Engine/)**

Move the sliders to explore how treaty structure, frequency distribution, severity tail, and cost of capital affect the technical premium. Results update instantly.

---

## Overview

`reinsurance-layer-pricing-engine` is an open-source Python library for actuarial researchers and quantitative analysts working in reinsurance pricing. It implements a frequency-severity Monte Carlo simulation framework to model aggregate annual loss distributions, apply reinsurance treaty structures, and derive technically sound premium estimates.

The engine supports Excess-of-Loss (XL) and Stop-Loss treaty types. Given user-specified frequency and severity distributions, it simulates a large number of accident years, applies the selected treaty to each simulated year's losses, and computes a full suite of risk measures on the resulting ceded loss distribution. A technical premium is derived using a cost-of-capital loading approach consistent with Solvency II principles.

The project is intended for educational and research purposes. It provides a transparent, modular codebase that can serve as a foundation for more sophisticated pricing models or as a reference implementation for actuarial students and practitioners.

---

## Key Features

- **Frequency-severity Monte Carlo simulation** over configurable numbers of accident-year scenarios
- **Modular distribution library** — Poisson and Negative Binomial frequency; Lognormal, Gamma, and Pareto severity
- **Excess-of-Loss layer pricing** with configurable retention and limit
- **Stop-Loss treaty pricing** with configurable attachment point and cap
- **Aggregate Annual Limit (AAL)** with per-claim tracking and partial cession of the straddling claim
- **Annual Aggregate Deductible (AAD)** — reinsurer pays only the excess over the annual threshold
- **Reinstatement premiums** — standard 1 free + 1 paid at 100% pro rata as to time
- **Full risk measure suite** — ECL, VaR, TVaR, CV, skewness, attachment and exhaustion probabilities
- **Technical premium** with explicit expense, profit, and capital loads via a cost-of-capital approach
- **Net ECL pricing** — technical premium adjusted for expected reinstatement premium income
- **Rate on Line (ROL)** calculation
- **Interactive dashboard** — 11250 pre-computed scenarios, available online or deployable as a static HTML file with no server required
- **Jupyter notebooks** — step-by-step walkthroughs of XL pricing, Stop-Loss pricing, and sensitivity analysis
- **Reproducible results** via configurable random seed
- **79% test coverage** across 49 unit tests

---

## Methodology

The engine follows a standard frequency-severity collective risk model. For a full mathematical treatment see [`docs/methodology.md`](docs/methodology.md).

**Step 1 — Frequency.** Draw claim count $N$ from the chosen frequency distribution (Poisson or Negative Binomial).

**Step 2 — Severity.** Draw $N$ individual losses $X_i$ independently from the chosen severity distribution (Lognormal, Gamma, or Pareto).

**Step 3 — Treaty.** Apply the reinsurance structure to each $X_i$ (XL) or to the aggregate $S = \sum X_i$ (Stop-Loss) to obtain the annual ceded loss $C$.

**Step 4 — Risk measures.** Compute ECL, VaR, TVaR, and other metrics from the empirical distribution of $\{C_1, \ldots, C_n\}$.

**Step 5 — Pricing.** Derive the technical premium using a cost-of-capital loading approach.

---

## Supported Treaty Types

| Treaty Type | Applied to | Ceded Amount |
|---|---|---|
| Excess-of-Loss (XL) | Each individual loss $X_i$ | $C_i = \min(\max(X_i - R, 0), L)$ |
| Stop-Loss | Aggregate annual loss $S$ | $C = \min(\max(S - A, 0), M)$ |

---

## Technical Pricing Formula

$$\text{Technical Premium} = \text{ECL} + e \cdot \text{ECL} + p \cdot \text{ECL} + r_c \cdot \max(\text{TVaR}_{99} - \text{ECL},\ 0)$$

| Parameter | Default | Description |
|---|---|---|
| `expected_ceded_loss` | — | Mean simulated ceded loss (pure premium) |
| `tvar_99` | — | TVaR at 99% confidence from simulation |
| `expense_load` | 0.05 | Proportional load for acquisition and admin costs |
| `profit_load` | 0.08 | Target profit margin as a fraction of ECL |
| `cost_of_capital` | 0.10 | Required return on risk capital |

The capital load is applied only to the unexpected loss — the portion of the tail that exceeds the ECL and requires capital support, consistent with Solvency II cost-of-capital principles.

---

## Project Structure

```
reinsurance-layer-pricing-engine/
├── README.md
├── SECURITY.md
├── pyproject.toml
├── run.py                                  # Quick start script (5 examples)
├── src/
│   └── reinsure_pricing/
│       ├── __init__.py
│       ├── frequency.py                    # ✅ Poisson, Negative Binomial
│       ├── severity.py                     # ✅ Lognormal, Gamma, Pareto
│       ├── treaties.py                     # ✅ ExcessOfLoss (AAL, AAD), StopLoss, ReinstatementProvision
│       ├── simulation.py                   # ✅ MonteCarloEngine, SimulationResults
│       ├── pricing.py                      # ✅ TechnicalPricer, PricingResult
│       ├── risk_measures.py                # ✅ compute_risk_measures, RiskMeasures
│       └── plots.py                        # ✅ plot_ceded_loss_distribution, plot_sensitivity
├── notebooks/
│   ├── 01_xol_pricing.ipynb               # ✅ Full XL pricing walkthrough
│   ├── 02_stop_loss_pricing.ipynb         # ✅ Stop-Loss pricing walkthrough
│   └── 03_sensitivity_analysis.ipynb      # ✅ Parameter sensitivity analysis
├── app/
│   ├── generate_scenarios.py              # ✅ Pre-computes 3,900+ scenarios
│   ├── dashboard.html                     # ✅ Static interactive dashboard
│   └── scenarios.json                     # ✅ Pre-computed scenario data
├── tests/
│   ├── test_frequency.py
│   ├── test_severity.py
│   ├── test_treaties.py                   # includes AAL, AAD, reinstatement tests
│   ├── test_simulation.py                 # includes AAL, AAD, reinstatement tests
│   ├── test_pricing.py                    # includes net ECL pricing tests
│   └── test_risk_measures.py
├── examples/
│   ├── negative_binomial_comparison.py    # 🔜 Poisson vs NB frequency
│   └── pareto_severity.py                 # 🔜 Heavy-tailed severity example
└── docs/
    └── methodology.md                     # ✅ Mathematical foundations
```

---

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/RS-ins/Reinsurance-Layer-Pricing-Engine.git
cd Reinsurance-Layer-Pricing-Engine
pip install -e ".[dev]"
```

Dependencies managed via `pyproject.toml`: `numpy`, `scipy`, `pandas`, `matplotlib`, `plotly`.

---

## Quick Start

```python
from reinsure_pricing.frequency import PoissonFrequency
from reinsure_pricing.severity import LognormalSeverity
from reinsure_pricing.treaties import ExcessOfLoss, ReinstatementProvision
from reinsure_pricing.simulation import MonteCarloEngine
from reinsure_pricing.risk_measures import compute_risk_measures
from reinsure_pricing.pricing import TechnicalPricer

# Define distributions and treaty
frequency = PoissonFrequency(lambda_=120)
severity  = LognormalSeverity(mu=10.5, sigma=1.2)
treaty    = ExcessOfLoss(
    retention=1_000_000,
    limit=5_000_000,
    aggregate_limit=15_000_000,      # AAL — optional
    aggregate_deductible=500_000,    # AAD — optional
)

# Reinstatement provision — 1 free + 1 paid at 100% pro rata
rp = ReinstatementProvision(n_free=1, n_paid=1, original_premium=300_000)

# Run Monte Carlo simulation
engine  = MonteCarloEngine(frequency, severity, treaty,
                           n_simulations=100_000, random_state=42)
results = engine.run(reinstatement=rp)

# Risk measures
rm = compute_risk_measures(results, treaty_limit=treaty.limit)
print(rm.summary())

# Technical premium — adjusted for reinstatement premium income
pricer = TechnicalPricer(
    expected_ceded_loss=results.expected_ceded_loss,
    tvar_99=results.tvar_99,
    expense_load=0.05,
    profit_load=0.08,
    cost_of_capital=0.10,
    expected_reinstatement_premium=results.expected_reinstatement_premium,
)
print(pricer.price(treaty_limit=treaty.limit).summary())
```

For five complete examples including basic XL, AAL, AAD, reinstatements, and Stop-Loss run `python run.py` or open the notebooks in `notebooks/`.

---

## Expected Output

Running `python run.py` produces the following output and two matplotlib plots.

```
─────────────────────────────────────────────
              RISK MEASURES
─────────────────────────────────────────────
Expected Ceded Loss   :         184,454
Std Deviation         :         515,188
Coeff of Variation    :           2.793
Skewness              :           4.218
─────────────────────────────────────────────
VaR  95%              :       1,243,721
VaR  99%              :       2,542,759
VaR  99.5%            :       3,198,442
─────────────────────────────────────────────
TVaR 95%              :       2,187,334
TVaR 99%              :       3,675,235
TVaR 99.5%            :       4,102,918
─────────────────────────────────────────────
Prob of Attachment    :           29.1%
Prob of Exhaustion    :            2.3%
─────────────────────────────────────────────
Technical Premium     :         557,511
Rate on Line          :          11.15%
```

Two plots follow sequentially — close the first to see the second:

- **Plot 1** — Ceded loss distribution histogram with VaR, TVaR, and treaty limit annotations
- **Plot 2** — Sensitivity chart showing premium and ECL as the retention varies from 500K to 3M

---

## Notebooks

Install Jupyter with `pip install jupyter ipykernel` then launch with `jupyter notebook` from the repo root.

| Notebook | Description |
|---|---|
| `01_xol_pricing.ipynb` | Full pricing walkthrough for a 5M xs 1M per-occurrence XL layer |
| `02_stop_loss_pricing.ipynb` | Pricing walkthrough for a Stop-Loss treaty with attachment sensitivity analysis |
| `03_sensitivity_analysis.ipynb` | One-at-a-time sensitivity analysis varying λ, σ, retention, cost of capital, and frequency model |

---

## Interactive Dashboard

A static HTML dashboard is deployed at:

**[https://rs-ins.github.io/Reinsurance-Layer-Pricing-Engine/](https://rs-ins.github.io/Reinsurance-Layer-Pricing-Engine/)**

It covers pre-computed scenarios across:

| Parameter | Values |
|---|---|
| Retention / Attachment | 5 levels |
| Limit / Cap | 5 levels (XL), 3 levels (SL) |
| Aggregate Annual Limit (AAL) | None, 2× limit, 4× limit |
| Annual Aggregate Deductible (AAD) | None, €200K, €500K |
| Severity tail σ | 5 levels (0.8 → 1.6) |
| Mean claim count λ | 5 levels (60 → 180) |
| Cost of capital | 2 levels (5% → 15%) |

Each XL scenario includes reinstatement premium data (1 free + 1 paid at 100% pro rata as to time).

To regenerate scenarios locally:

---

## Running the Dashboard Locally

The dashboard is a static HTML file that reads `scenarios.json` from the same folder. Because browsers block local file access for security reasons, you need a simple local server to run it — you cannot just double-click `dashboard.html`.

**Step 1 — Generate scenarios** (only needed once, or when parameters change):

```bash
python app/generate_scenarios.py
```

**Step 2 — Start a local server** from the repo root:

```bash
cd app
python -m http.server 8080
```

**Step 3 — Open in browser:**

```
http://127.0.0.1:8080/dashboard.html
```

That's it. The server runs until you press `Ctrl+C` in the terminal.

> **Note:** If you change any parameters in `generate_scenarios.py` (retentions, limits, sigmas, etc.) you must regenerate `scenarios.json` and update the matching arrays at the top of `dashboard.html` to keep them in sync. The parameter grids in the JavaScript must exactly match the values used during generation.

---

## Running Tests

```bash
python -m pytest
```

Output includes a coverage report for all source modules. Current coverage: 88% across 34 tests.

---

## Roadmap

| Phase | Status | Features |
|---|---|---|
| Phase 1 — Core Engine | ✅ Complete | Frequency, severity, treaties, simulation, risk measures, pricing, plots |
| Phase 2 — Notebooks | ✅ Complete | XL pricing, Stop-Loss pricing, sensitivity analysis |
| Phase 3 — Dashboard | ✅ Complete | Static HTML dashboard, GitHub Pages deployment |
| Phase 4 — Advanced Features | ✅ Complete | AAL with per-claim tracking, AAD, reinstatement premiums |
| Phase 5 — Production Hardening | 🔜 Planned | Distribution fitting, bootstrapped CIs, export to CSV/Excel |

---

## Limitations

- **Independence assumption** — frequency and severity are modelled as independent
- **No parameter uncertainty** — distribution parameters are treated as known
- **Reinstatement timing** — reinstatement pro rata fraction uses a uniform timing approximation
- **No distribution fitting** — users are responsible for calibrating distributional assumptions
- **Not validated** against industry benchmarks or actuarial software

See [`docs/methodology.md`](docs/methodology.md) for a full discussion of model risk.

---

## Disclaimer

This project is developed for research and educational purposes. It is **not** intended for production use, commercial pricing decisions, regulatory filings, or any professional actuarial application without independent review by a qualified actuary. Results are illustrative only.

---

## License

Released under the [MIT License](LICENSE).
