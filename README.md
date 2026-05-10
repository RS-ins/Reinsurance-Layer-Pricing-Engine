# Reinsurance Layer Pricing Engine

> A Python Monte Carlo engine for pricing Excess-of-Loss and Stop-Loss reinsurance layers, estimating ceded loss distributions, risk measures, and capital-loaded technical premiums.

[![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Educational](https://img.shields.io/badge/purpose-educational-green.svg)]()

---

## Live Dashboard

An interactive dashboard with 11,250+ pre-computed scenarios is available online — no installation required:

**[→ Open Dashboard](https://rs-ins.github.io/Reinsurance-Layer-Pricing-Engine/)**

Move the sliders to explore how treaty structure, frequency distribution, severity tail, cost of capital, AAL, and AAD affect the technical premium. Click any metric card for a plain-English explanation.

---

## Overview

`reinsurance-layer-pricing-engine` is an open-source Python library for actuarial researchers and quantitative analysts working in reinsurance pricing. It implements a frequency-severity Monte Carlo simulation framework to model aggregate annual loss distributions, apply reinsurance treaty structures, and derive technically sound premium estimates.

The engine supports Excess-of-Loss (XL) and Stop-Loss treaty types, including advanced features such as Aggregate Annual Limits, Annual Aggregate Deductibles, and reinstatement premiums. Distribution parameters can be specified manually or fitted automatically from historical loss data using Maximum Likelihood Estimation.

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
- **Bootstrapped confidence intervals** for all risk measures — quantifies estimate stability
- **Distribution fitting** from historical loss data via MLE with AIC-based model selection
- **Technical premium** with explicit expense, profit, and capital loads via a cost-of-capital approach
- **Net ECL pricing** — technical premium adjusted for expected reinstatement premium income
- **Rate on Line (ROL)** calculation
- **Excel export** — multi-sheet report with summary, risk measures, pricing, distribution, and bootstrap results
- **End-to-end workflow** via `fit_from_file.py` — load CSV, fit distributions, simulate, price, export
- **Interactive dashboard** — 11,250+ pre-computed scenarios, deployable as a static HTML file
- **Jupyter notebooks** — step-by-step walkthroughs of XL pricing, Stop-Loss pricing, and sensitivity analysis
- **Reproducible results** via configurable random seed

---

## Methodology

The engine follows a standard frequency-severity collective risk model. For a full mathematical treatment see [`docs/methodology.md`](docs/methodology.md).

**Step 1 — Frequency.** Draw claim count $N$ from the chosen frequency distribution (Poisson or Negative Binomial).

**Step 2 — Severity.** Draw $N$ individual losses $X_i$ independently from the chosen severity distribution (Lognormal, Gamma, or Pareto).

**Step 3 — Treaty.** Apply the reinsurance structure to each $X_i$ (XL) or to the aggregate $S = \sum X_i$ (Stop-Loss) to obtain the annual ceded loss $C$.

**Step 4 — Risk measures.** Compute ECL, VaR, TVaR, and other metrics from the empirical distribution of $\{C_1, \ldots, C_n\}$.

**Step 5 — Pricing.** Derive the technical premium using a cost-of-capital loading approach consistent with Solvency II.

---

## Supported Treaty Types

| Treaty Type | Applied to | Ceded Amount |
|---|---|---|
| Excess-of-Loss (XL) | Each individual loss $X_i$ | $C_i = \min(\max(X_i - R, 0), L)$ |
| Stop-Loss | Aggregate annual loss $S$ | $C = \min(\max(S - A, 0), M)$ |

---

## Technical Pricing Formula

$$\text{Technical Premium} = \text{Net ECL} + e \cdot \text{Net ECL} + p \cdot \text{Net ECL} + r_c \cdot \max(\text{TVaR}_{99} - \text{Net ECL},\ 0)$$

| Parameter | Default | Description |
|---|---|---|
| `expected_ceded_loss` | — | Mean simulated ceded loss (pure premium) |
| `tvar_99` | — | TVaR at 99% confidence from simulation |
| `expense_load` | 0.05 | Proportional load for acquisition and admin costs |
| `profit_load` | 0.08 | Target profit margin as a fraction of ECL |
| `cost_of_capital` | 0.10 | Required return on risk capital |
| `expected_reinstatement_premium` | 0.0 | Expected annual reinstatement premium income |

---

## Project Structure

```
reinsurance-layer-pricing-engine/
├── README.md
├── SECURITY.md
├── pyproject.toml
├── run.py                                  # Quick start script (8 examples)
├── fit_from_file.py                        # ✅ End-to-end workflow from CSV input
├── src/
│   └── reinsure_pricing/
│       ├── __init__.py
│       ├── frequency.py                    # ✅ Poisson, Negative Binomial
│       ├── severity.py                     # ✅ Lognormal, Gamma, Pareto
│       ├── treaties.py                     # ✅ ExcessOfLoss (AAL, AAD), StopLoss, ReinstatementProvision
│       ├── simulation.py                   # ✅ MonteCarloEngine, SimulationResults
│       ├── pricing.py                      # ✅ TechnicalPricer, PricingResult
│       ├── risk_measures.py                # ✅ compute_risk_measures, RiskMeasures
│       ├── plots.py                        # ✅ plot_ceded_loss_distribution, plot_sensitivity
│       ├── bootstrap.py                    # ✅ bootstrap_risk_measures, BootstrappedRiskMeasures
│       ├── fitting.py                      # ✅ fit_frequency, fit_severity, FittingComparison
│       └── io.py                           # ✅ load_losses, export_report
├── notebooks/
│   ├── 01_xol_pricing.ipynb               # ✅ Full XL pricing walkthrough
│   ├── 02_stop_loss_pricing.ipynb         # ✅ Stop-Loss pricing walkthrough
│   └── 03_sensitivity_analysis.ipynb      # ✅ Parameter sensitivity analysis
├── app/
│   ├── generate_scenarios.py              # ✅ Pre-computes 11,250+ scenarios
│   ├── dashboard.html                     # ✅ Static interactive dashboard
│   └── scenarios.json                     # ✅ Pre-computed scenario data
├── data/
│   └── sample_losses.csv                  # ✅ Sample historical loss data
├── outputs/                               # Generated Excel reports (gitignored)
├── tests/
│   ├── test_frequency.py
│   ├── test_severity.py
│   ├── test_treaties.py
│   ├── test_simulation.py
│   ├── test_pricing.py
│   └── test_risk_measures.py
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
pip install openpyxl   # required for Excel export
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
from reinsure_pricing.bootstrap import bootstrap_risk_measures

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

# Technical premium
pricer = TechnicalPricer(
    expected_ceded_loss=results.expected_ceded_loss,
    tvar_99=results.tvar_99,
    expense_load=0.05,
    profit_load=0.08,
    cost_of_capital=0.10,
    expected_reinstatement_premium=results.expected_reinstatement_premium,
)
print(pricer.price(treaty_limit=treaty.limit).summary())

# Bootstrapped confidence intervals
boot = bootstrap_risk_measures(results, treaty_limit=treaty.limit, n_bootstrap=1_000)
print(boot.summary())
```

For eight complete examples run `python run.py` or open the notebooks in `notebooks/`.

---

## Fitting from Historical Data

The `fit_from_file.py` script provides a complete end-to-end workflow:
load historical loss data → fit distributions → simulate → price → export to Excel.

### Input file format

```csv
year,claim_count,loss_amount
2000,8,234000
2000,8,567000
2001,11,890000
2001,11,1200000
```

Both `claim_count` and `loss_amount` can be in the same file. The engine deduplicates claim counts per year automatically.

### Usage

```bash
# Basic XL with default parameters
python fit_from_file.py --input data/sample_losses.csv

# Custom XL treaty
python fit_from_file.py --input data/sample_losses.csv \
    --retention 1000000 --limit 5000000

# XL with Phase 4 features
python fit_from_file.py --input data/sample_losses.csv \
    --retention 1000000 --limit 5000000 \
    --aal 15000000 --aad 500000 \
    --reinstatements 1

# Stop-Loss
python fit_from_file.py --input data/sample_losses.csv \
    --treaty-type sl --attachment 90000000 --cap 20000000

# Reproducible run, no plot
python fit_from_file.py --input data/sample_losses.csv \
    --seed 42 --no-plot

# Fit only losses above a threshold
python fit_from_file.py --input data/sample_losses.csv \
    --threshold 500000
```

### Available arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Path to CSV input file |
| `--treaty-type` | `xl` | `xl` or `sl` |
| `--retention` | 1,000,000 | XL retention |
| `--limit` | 5,000,000 | XL limit |
| `--aal` | None | Aggregate Annual Limit |
| `--aad` | None | Annual Aggregate Deductible |
| `--reinstatements` | 0 | Number of paid reinstatements |
| `--free-reinstatements` | 1 | Number of free reinstatements |
| `--attachment` | 10,000,000 | Stop-Loss attachment |
| `--cap` | 20,000,000 | Stop-Loss cap |
| `--threshold` | None | Fit severity above this loss level |
| `--expense-load` | 0.05 | Expense load rate |
| `--profit-load` | 0.08 | Profit load rate |
| `--cost-of-capital` | 0.10 | Cost of capital rate |
| `--n-sims` | 100,000 | Number of simulations |
| `--seed` | None | Random seed (None = random each run) |
| `--no-bootstrap` | False | Skip bootstrapped CIs |
| `--no-plot` | False | Skip matplotlib plot |

The report is saved automatically to `outputs/pricing_report.xlsx`. If a report already exists, a timestamp is appended to avoid overwriting.

---

## Expected Output

Running `python run.py` produces printed output for eight examples followed by two matplotlib plots. Each example follows the same structure showing risk measures and pricing breakdown. The eight examples cover:

| Example | Treaty | Feature demonstrated |
|---|---|---|
| 1 | 5M xs 1M XL | Basic pricing — baseline |
| 2 | 5M xs 1M XL + 15M AAL | Effect of Aggregate Annual Limit |
| 3 | 5M xs 1M XL + 500K AAD | Effect of Annual Aggregate Deductible |
| 4 | 5M xs 1M XL + reinstatements | Net ECL after reinstatement premium income |
| 5 | 20M xs 10M Stop-Loss | Aggregate treaty structure |
| 6 | Bootstrap on Example 1 | Confidence intervals on all risk measures |
| 7 | Export on Example 1 | Excel report saved to `outputs/` |
| 8 | Distribution fitting | Fit distributions from simulated historical data |

Example 6 produces the bootstrapped confidence interval table:

```
──────────────────────────────────────────────────────────────────────
                      BOOTSTRAPPED RISK MEASURES
                        n_simulations = 100000
                    n_bootstrap = 1000 | CI = 95%
──────────────────────────────────────────────────────────────────────
Measure                    Estimate     CI Lower     CI Upper  Rel Width
──────────────────────────────────────────────────────────────────────
ECL                         184,454      181,471      187,951       3.5%
Std Deviation               515,188      505,744      524,468       3.6%
VaR 95%                   1,083,651    1,059,956    1,103,589       4.0%
VaR 99%                   2,542,759    2,469,707    2,611,904       5.6%
VaR 99.5%                 3,374,600    3,258,739    3,539,819       8.3%
TVaR 95%                  2,002,855    1,965,287    2,040,574       3.8%
TVaR 99%                  3,675,235    3,583,314    3,761,884       4.9%
TVaR 99.5%                4,441,509    4,327,346    4,551,739       5.1%
Prob Attachment              29.1%       28.9%       29.4%       1.9%
Prob Exhaustion               0.2%        0.1%        0.2%      32.3%
──────────────────────────────────────────────────────────────────────
Rel Width < 5% = stable. > 10% = consider more simulations.
```

Two matplotlib plots follow sequentially:
- **Plot 1** — Ceded loss distribution histogram with VaR, TVaR, and treaty limit annotations
- **Plot 2** — Sensitivity chart showing premium and ECL as the retention varies

---

## Notebooks

Install Jupyter with `pip install jupyter ipykernel` then launch with `jupyter notebook` from the repo root.

| Notebook | Description |
|---|---|
| `01_xol_pricing.ipynb` | Full pricing walkthrough for a 5M xs 1M per-occurrence XL layer |
| `02_stop_loss_pricing.ipynb` | Pricing walkthrough for a Stop-Loss treaty with attachment sensitivity |
| `03_sensitivity_analysis.ipynb` | One-at-a-time sensitivity analysis varying λ, σ, retention, cost of capital, and frequency model |

---

## Interactive Dashboard

A static HTML dashboard is deployed at:

**[https://rs-ins.github.io/Reinsurance-Layer-Pricing-Engine/](https://rs-ins.github.io/Reinsurance-Layer-Pricing-Engine/)**

It covers 11,250+ pre-computed scenarios across:

| Parameter | Values |
|---|---|
| Retention / Attachment | 5 levels |
| Limit / Cap | 5 levels (XL), 3 levels (SL) |
| Aggregate Annual Limit (AAL) | None, 2× limit, 4× limit |
| Annual Aggregate Deductible (AAD) | None, €200K, €500K |
| Severity tail σ | 5 levels (0.8 → 1.6) |
| Mean claim count λ | 5 levels (60 → 180) |
| Cost of capital | 2 levels (5%, 15%) |

Each XL scenario includes reinstatement premium data. Click any metric card for a plain-English explanation with the underlying formula.

To regenerate scenarios locally:

```bash
python app/generate_scenarios.py   # ~30 minutes
```

To run the dashboard locally:

```bash
cd app
python -m http.server 8080
# then open http://127.0.0.1:8080/dashboard.html
```

---

## Running Tests

```bash
python -m pytest
```

Output includes a coverage report for all source modules.

---

## Roadmap

| Phase | Status | Features |
|---|---|---|
| Phase 1 — Core Engine | ✅ Complete | Frequency, severity, treaties, simulation, risk measures, pricing, plots |
| Phase 2 — Notebooks | ✅ Complete | XL pricing, Stop-Loss pricing, sensitivity analysis |
| Phase 3 — Dashboard | ✅ Complete | Static HTML dashboard, GitHub Pages deployment |
| Phase 4 — Advanced Features | ✅ Complete | AAL with per-claim tracking, AAD, reinstatement premiums |
| Phase 5 — Production Hardening | ✅ Complete | Distribution fitting, bootstrapped CIs, Excel export, end-to-end workflow |
| Phase 6 — Planned | 🔜 Planned | Copula dependence modelling, parameter uncertainty, credibility weighting |

---

## Limitations

- **Independence assumption** — frequency and severity are modelled as independent
- **No parameter uncertainty** — fitted parameters are treated as known with certainty
- **Reinstatement timing** — pro rata fraction uses a uniform timing approximation
- **Not validated** against industry benchmarks or actuarial software

See [`docs/methodology.md`](docs/methodology.md) for a full discussion of model risk.

---

## Disclaimer

This project is developed for research and educational purposes. It is **not** intended for production use, commercial pricing decisions, regulatory filings, or any professional actuarial application without independent review by a qualified actuary. Results are illustrative only.

---

## License

Released under the [Apache 2.0 License](LICENSE).