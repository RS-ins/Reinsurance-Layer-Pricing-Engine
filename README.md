# Reinsurance Layer Pricing Engine

> A Python Monte Carlo engine for pricing Excess-of-Loss and Stop-Loss reinsurance layers, estimating ceded loss distributions, risk measures, and capital-loaded technical premiums.

---

## Overview

`reinsurance-layer-pricing-engine` is an open-source Python library designed for actuarial researchers and quantitative analysts working in reinsurance pricing.

It implements a frequency-severity Monte Carlo simulation framework to model aggregate annual loss distributions, apply reinsurance treaty structures, and derive technically sound premium estimates. The engine supports Excess-of-Loss (XL) and Stop-Loss treaty types.

Given user-specified frequency and severity distributions, it simulates a large number of accident years, applies the selected treaty to each simulated year's losses, and computes a full suite of risk measures on the resulting ceded loss distribution. A technical premium is then derived using a cost-of-capital loading approach consistent with modern actuarial and regulatory practice.

The project is intended for educational and research purposes. It provides a transparent, modular codebase that can serve as a foundation for more sophisticated pricing models or as a reference implementation for actuarial students and practitioners.

---

## Key Features

- **Frequency-severity Monte Carlo simulation** over configurable numbers of accident-year scenarios
- **Modular distribution library** covering Poisson and Negative Binomial frequency models and Lognormal, Gamma, and Pareto severity models
- **Excess-of-Loss layer pricing** with configurable retention and limit
- **Stop-Loss treaty pricing** with configurable attachment point and cap
- **Full ceded loss distribution** output per simulation run
- **Risk measures**: Expected Ceded Loss, Standard Deviation, VaR, TVaR, probability of attachment, and probability of exhaustion
- **Technical premium computation** with explicit expense, profit, and capital loads via a cost-of-capital approach
- **Rate on Line (ROL)** calculation relative to treaty limit
- **Static browser dashboard** for interactive scenario comparison
- **Simulation distribution histogram** using pre-computed scenario distributions
- **Responsive layout** for desktop, tablet, and mobile devices
- **Reproducible results** via configurable random seed

---

## Methodology

The engine follows a standard frequency-severity collective risk model:

**Step 1 — Frequency simulation.** 
For each simulated accident year, a claim count $N$ is drawn from the chosen frequency distribution:

- Poisson
- Negative Binomial

**Step 2 — Severity simulation.** 
For each of the $N$ claims, an individual loss $X_i$ is drawn independently from the chosen severity distribution:

- Lognormal
- Gamma
- Pareto

**Step 3 — Aggregate loss.**
The aggregate annual loss is computed as:

$$
S = \sum_{i=1}^{N} X_i
$$

**Step 4 — Treaty application.**
The reinsurance structure is applied to each simulated $S$ for Stop-Loss, or to each individual $X_i$ before aggregation for per-occurrence Excess-of-Loss, to obtain the ceded loss $C$.

**Step 5 — Risk measure estimation.**
The empirical distribution of $\{C_1, \ldots, C_n\}$ is used to estimate all reported risk measures.

**Step 6 — Technical pricing.**
The technical premium is derived as described in the pricing formula section below.

---

## Supported Treaty Types

| Treaty Type | Description |
|---|---|
| Excess-of-Loss (XL) | Covers individual losses between a retention and a limit on a per-occurrence basis |
| Stop-Loss | Covers aggregate annual losses exceeding an attachment point, subject to a cap |

For an Excess-of-Loss treaty with retention $R$ and limit $L$, the ceded amount per occurrence is:

$$
C_i = \min\!\bigl(\max(X_i - R,\ 0),\ L\bigr)
$$

For a Stop-Loss treaty with attachment point $A$ and cap $M$, the ceded aggregate amount is:

$$
C = \min\!\bigl(\max(S - A,\ 0),\ M\bigr)
$$

---

## Risk Measures

The following metrics are computed on the simulated ceded loss distribution:

| Measure | Definition |
|---|---|
| Expected Ceded Loss | $\mathbb{E}[C]$ — mean of simulated ceded losses |
| Standard Deviation | $\sigma(C)$ — standard deviation of simulated ceded losses |
| VaR at level $\alpha$ | $\text{VaR}_\alpha(C)$ — empirical quantile of $C$ at confidence level $\alpha$ |
| TVaR at level $\alpha$ | $\mathbb{E}[C \mid C > \text{VaR}_\alpha(C)]$ — Tail Value-at-Risk, or conditional tail expectation |
| Probability of Attachment | $P(C > 0)$ — proportion of simulated years in which the layer is triggered |
| Probability of Exhaustion | $P(C = L)$ — proportion of simulated years in which the layer is fully exhausted |
| Rate on Line (ROL) | $\text{Technical Premium} / L$ — premium as a fraction of treaty limit |

---

## Technical Pricing Formula

The technical premium is built up from four components:

```text
Technical Premium =
    ECL
  + expense_load × ECL
  + profit_load × ECL
  + cost_of_capital × max(TVaR99 - ECL, 0)
```

| Parameter | Default | Description |
|---|---:|---|
| `expected_ceded_loss` | — | Mean simulated ceded loss; the pure premium |
| `tvar_99` | — | TVaR at 99% confidence from simulation results |
| `expense_load` | `0.05` | Proportional load for acquisition and admin expenses |
| `profit_load` | `0.08` | Target profit margin as a fraction of ECL |
| `cost_of_capital` | `0.10` | Required return on risk capital |

The capital load is applied to the unexpected loss:

```text
max(TVaR99 - ECL, 0)
```

This represents the portion of the tail that exceeds the expected loss and requires capital support. This is consistent with cost-of-capital principles underlying Solvency II.

---

## Project Structure

```text
reinsurance-layer-pricing-engine/
├── README.md
├── pyproject.toml
├── run.py                         # Quick start script
├── src/
│   └── reinsure_pricing/
│       ├── __init__.py
│       ├── frequency.py           # ✅ implemented
│       ├── severity.py            # ✅ implemented
│       ├── treaties.py            # ✅ implemented
│       ├── simulation.py          # ✅ implemented
│       ├── pricing.py             # ✅ implemented
│       ├── risk_measures.py       # ✅ implemented
│       └── plots.py               # ✅ implemented
├── notebooks/
│   ├── 01_xol_pricing.ipynb       # ✅ implemented
│   ├── 02_stop_loss_pricing.ipynb # ✅ implemented
│   └── 03_sensitivity_analysis.ipynb
├── app/
│   ├── dashboard.html             # ✅ static browser dashboard
│   ├── generate_scenarios.py      # ✅ pre-computes dashboard scenarios
│   └── scenarios.json             # generated output, required by dashboard
├── tests/
│   ├── test_frequency.py
│   ├── test_severity.py
│   ├── test_treaties.py
│   ├── test_simulation.py
│   ├── test_pricing.py
│   └── test_risk_measures.py
├── examples/
│   └── xol_pricing_example.py     # coming
└── docs/
    └── methodology.md             # coming
```

---

## Installation

Python 3.10 or later is required.

**From source, recommended for development:**

```bash
git clone https://github.com/RS-ins/Reinsurance-Layer-Pricing-Engine.git
cd Reinsurance-Layer-Pricing-Engine
pip install -e ".[dev]"
```

**Dependencies** are managed via `pyproject.toml`:

```text
numpy
scipy
pandas
matplotlib
streamlit
```

The static dashboard also uses Chart.js from a CDN inside `app/dashboard.html`.

---

## Quick Start Example

The following example prices a 1M xs 1M Excess-of-Loss layer using a Poisson frequency model and a Lognormal severity model over 100,000 simulated accident years.

For a full step-by-step walkthrough with explanations, see the notebooks in the `notebooks/` folder. The quick start below runs the complete pipeline in a single script.

```python
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
severity = LognormalSeverity(mu=10.5, sigma=1.2)

# Define the reinsurance treaty: 5M xs 1M per occurrence
treaty = ExcessOfLoss(retention=1_000_000, limit=5_000_000)

# Run the Monte Carlo simulation
engine = MonteCarloEngine(
    frequency=frequency,
    severity=severity,
    treaty=treaty,
    n_simulations=100_000,
    random_state=42,
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
    cost_of_capital=0.10,
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
    e = MonteCarloEngine(
        frequency,
        severity,
        t,
        n_simulations=50_000,
        random_state=42,
    )
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
    parameter_values=[r / 1e6 for r in retentions],
    premiums=premiums_list,
    ecls=ecls_list,
    parameter_name="Retention (M€)",
)

plt.show()
```

---

## Expected Output

Running `python run.py` produces printed output and two matplotlib plots.

**Printed output:**

```text
─────────────────────────────────────────────
RISK MEASURES
─────────────────────────────────────────────
Expected Ceded Loss : 184,454
Std Deviation       : 515,188
Coeff of Variation  : 2.793
Skewness            : 4.218
─────────────────────────────────────────────
VaR 95%             : 1,243,721
VaR 99%             : 2,542,759
VaR 99.5%           : 3,198,442
─────────────────────────────────────────────
TVaR 95%            : 2,187,334
TVaR 99%            : 3,675,235
TVaR 99.5%          : 4,102,918
─────────────────────────────────────────────
Prob of Attachment  : 29.1%
Prob of Exhaustion  : 2.3%
─────────────────────────────────────────────
Technical Premium   : 557,511
Rate on Line        : 11.15%
```

**Plot 1 — Ceded Loss Distribution:**  
Histogram of simulated annual ceded losses with VaR 95%, VaR 99%, VaR 99.5%, TVaR 99%, and treaty limit annotated as vertical lines.

**Plot 2 — Sensitivity Analysis:**  
Line chart showing how the technical premium and ECL vary as the retention increases from 500K to 3M, with all other parameters held constant.

---

## Notebooks

Three Jupyter notebooks are provided for deeper walkthroughs of the engine.

Install Jupyter with:

```bash
pip install jupyter ipykernel
```

Launch from the repo root:

```bash
jupyter notebook
```

| Notebook | Description |
|---|---|
| `01_xol_pricing.ipynb` | Full pricing walkthrough for a 5M xs 1M per-occurrence XL layer. Covers distribution setup, simulation, risk measures, technical pricing, and retention sensitivity. |
| `02_stop_loss_pricing.ipynb` | Pricing walkthrough for a Stop-Loss treaty. Explains the difference between per-occurrence and aggregate structures and includes attachment sensitivity analysis. |
| `03_sensitivity_analysis.ipynb` | Systematic one-at-a-time sensitivity analysis varying claim frequency, severity tail, retention, cost of capital, and frequency distribution. Includes parameter impact ranking. |

---

## Static Dashboard

A static interactive dashboard is provided in:

```text
app/dashboard.html
```

The dashboard does not run simulations in the browser. Instead, all pricing scenarios are pre-computed by:

```text
app/generate_scenarios.py
```

and saved into:

```text
app/scenarios.json
```

The dashboard then loads `scenarios.json` client-side for instant interaction.

### Dashboard features

The dashboard includes:

- treaty type toggle:
  - Excess-of-Loss
  - Stop-Loss
- sidebar controls for treaty structure
- sidebar controls for claim frequency
- sidebar controls for severity tail parameter
- sidebar controls for cost of capital
- risk measure cards
- technical pricing breakdown
- premium sensitivity charts
- simulation distribution histogram
- responsive layout for desktop, tablet, and mobile devices

### Scenario grid

The current pre-computed scenario grid includes:

**Excess-of-Loss parameters**

```text
Retentions: 500K, 750K, 1M, 1.5M, 2M
Limits:     2M, 3M, 5M, 7M, 10M
```

**Stop-Loss parameters**

```text
Attachments: 10M, 20M, 30M, 40M, 50M
Caps:        10M, 20M, 30M
```

**Shared parameters**

```text
Severity sigma:   0.8, 1.0, 1.2, 1.4, 1.6
Frequency lambda: 60, 90, 120, 150, 180
Cost of capital:  5%, 10%, 15%, 20%
Simulations:      20,000 per scenario
```

Each scenario stores:

- expected ceded loss
- standard deviation
- VaR 95%
- VaR 99%
- VaR 99.5%
- TVaR 99%
- probability of attachment
- probability of exhaustion
- coefficient of variation
- skewness
- expense loading
- profit loading
- capital load
- technical premium
- rate on line
- compact histogram distribution bins

The histogram bins allow the dashboard to display the simulated ceded loss distribution without storing every simulated loss value.

### Generate dashboard scenarios

From the project root:

```bash
python app/generate_scenarios.py
```

This writes:

```text
app/scenarios.json
```

The generation step may take several minutes depending on machine speed.

### Run the dashboard locally

From the project root:

```bash
cd app
python -m http.server 8000
```

Open:

```text
http://localhost:8000/dashboard.html
```

Do not open `dashboard.html` directly with a `file://` path. The browser may block loading `scenarios.json`.

---

## Roadmap

The following features are under consideration for future development:

- Reinstatement provisions for Excess-of-Loss treaties
- Aggregate annual limit (AAL) and annual aggregate deductible (AAD) structures
- Correlation modelling across lines of business
- Reinsurance programme optimisation, stacking multiple layers
- Bootstrapped confidence intervals for all risk measures
- Export of simulation results to CSV and Excel
- Integration with external loss development triangle inputs

Contributions and suggestions are welcome via GitHub Issues.

---

## Limitations

This project is a research and educational tool. Users should be aware of the following limitations before applying it in any applied context:

- **Model risk**: The engine assumes independence between claim frequency and severity. Dependence structures, such as copulas, are not currently implemented.
- **Parameter uncertainty**: No parameter uncertainty is propagated through the simulation. All distribution parameters are treated as known.
- **Individual loss data**: The per-occurrence XL implementation simulates individual losses independently. Loss correlation within an accident year is not modelled.
- **Reinstatements**: Reinstatement premiums are not calculated. The treaty limit is treated as fully available throughout the year.
- **Inflation**: No loss development, tail factors, or future inflation adjustments are applied to severity inputs.
- **Calibration**: No functionality is provided for fitting distributions to historical loss data. Users are responsible for selecting and calibrating their own distributional assumptions.
- **Validation**: The model has not been independently validated against industry benchmarks or actuarial software. Output should be cross-checked against established tools before use in any professional context.

---

## Disclaimer

This project is developed as a personal side project for research and educational purposes.

It is **not** intended for production use, commercial pricing decisions, regulatory filings, or any other professional actuarial application without independent review and validation by a qualified actuary.

The technical premium formula implemented here is a simplified representation of actuarial pricing methodology. It does not constitute professional actuarial advice.

Results produced by this engine are illustrative only and should not be relied upon for any commercial, regulatory, or risk management purpose.

The author accept no liability for decisions made on the basis of output generated by this software.

---

## License

This project is released under the [MIT License](LICENSE).

```text
MIT License

Copyright (c) 2024 RS-ins

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
