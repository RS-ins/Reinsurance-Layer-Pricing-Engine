# Reinsurance Layer Pricing Engine

> A Python Monte Carlo engine for pricing Excess-of-Loss and Stop-Loss reinsurance layers, estimating ceded loss distributions, risk measures, and capital-loaded technical premiums.

---

## Overview

`reinsurance-layer-pricing-engine` is an open-source Python library designed for actuarial researchers and quantitative analysts working in reinsurance pricing. It implements a frequency-severity Monte Carlo simulation framework to model aggregate annual loss distributions, apply reinsurance treaty structures, and derive technically sound premium estimates.

The engine supports Excess-of-Loss (XL) and Stop-Loss treaty types. Given user-specified frequency and severity distributions, it simulates a large number of accident years, applies the selected treaty to each simulated year's losses, and computes a full suite of risk measures on the resulting ceded loss distribution. A technical premium is then derived using a cost-of-capital loading approach consistent with modern actuarial and regulatory practice.

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
- **Streamlit dashboard** for interactive parameterisation and scenario comparison
- **Reproducible results** via configurable random seed

---

## Methodology

The engine follows a standard frequency-severity collective risk model:

**Step 1 — Frequency simulation.** For each simulated accident year, a claim count $N$ is drawn from the chosen frequency distribution (Poisson or Negative Binomial).

**Step 2 — Severity simulation.** For each of the $N$ claims, an individual loss $X_i$ is drawn independently from the chosen severity distribution (Lognormal, Gamma, or Pareto).

**Step 3 — Aggregate loss.** The aggregate annual loss is computed as:

$$S = \sum_{i=1}^{N} X_i$$

**Step 4 — Treaty application.** The reinsurance structure is applied to each simulated $S$ (for Stop-Loss) or to each individual $X_i$ before aggregation (for per-occurrence Excess-of-Loss) to obtain the ceded loss $C$.

**Step 5 — Risk measure estimation.** The empirical distribution of $\{C_1, \ldots, C_n\}$ is used to estimate all reported risk measures.

**Step 6 — Technical pricing.** The technical premium is derived as described in the pricing formula section below.

---

## Supported Treaty Types

| Treaty Type | Description |
|---|---|
| Excess-of-Loss (XL) | Covers individual losses between a retention and a limit on a per-occurrence basis |
| Stop-Loss | Covers aggregate annual losses exceeding an attachment point, subject to a cap |

For an Excess-of-Loss treaty with retention $R$ and limit $L$, the ceded amount per occurrence is:

$$C_i = \min\!\bigl(\max(X_i - R,\ 0),\ L\bigr)$$

For a Stop-Loss treaty with attachment point $A$ and cap $M$, the ceded aggregate amount is:

$$C = \min\!\bigl(\max(S - A,\ 0),\ M\bigr)$$

---

## Risk Measures

The following metrics are computed on the simulated ceded loss distribution:

| Measure | Definition |
|---|---|
| Expected Ceded Loss | $\mathbb{E}[C]$ — mean of simulated ceded losses |
| Standard Deviation | $\sigma(C)$ — standard deviation of simulated ceded losses |
| VaR at level $\alpha$ | $\text{VaR}_\alpha(C)$ — empirical quantile of $C$ at confidence level $\alpha$ |
| TVaR at level $\alpha$ | $\mathbb{E}[C \mid C > \text{VaR}_\alpha(C)]$ — Tail Value-at-Risk (conditional tail expectation) |
| Probability of Attachment | $P(C > 0)$ — proportion of simulated years in which the layer is triggered |
| Probability of Exhaustion | $P(C = L)$ — proportion of simulated years in which the layer is fully exhausted |
| Rate on Line (ROL) | $\text{Technical Premium} / L$ — premium as a fraction of treaty limit |

---

## Technical Pricing Formula

The technical premium is built up from four components:

$$\text{Technical Premium} = \underbrace{\mathbb{E}[C]}_{\text{Pure Premium}} + \underbrace{e \cdot \mathbb{E}[C]}_{\text{Expense Load}} + \underbrace{p \cdot \mathbb{E}[C]}_{\text{Profit Load}} + \underbrace{\text{Capital Load}}_{}$$

where the capital load is computed using a cost-of-capital approach:

$$\text{Capital Load} = r_c \cdot \max\!\bigl(\text{TVaR}_{99}(C) - \mathbb{E}[C],\ 0\bigr)$$

| Parameter | Symbol | Description |
|---|---|---|
| Expected Ceded Loss | $\mathbb{E}[C]$ | Mean simulated ceded loss; the pure premium |
| Expense load rate | $e$ | Proportional load for acquisition and administrative expenses |
| Profit load rate | $p$ | Target profit margin expressed as a fraction of pure premium |
| Cost of capital | $r_c$ | Required return on risk capital (e.g., 10%) |
| Risk capital proxy | $\text{TVaR}_{99} - \mathbb{E}[C]$ | Unexpected loss requiring capital support |

This formulation is consistent with the cost-of-capital principles underlying Solvency II and standard actuarial practice for risk-adjusted pricing.

---

## Project Structure

```
reinsurance-layer-pricing-engine/
  ├── README.md
  ├── pyproject.toml
  ├── src/
  │  └── reinsure_pricing/
  │    ├── __init__.py
  │    ├── frequency.py          # Poisson, Negative Binomial frequency models
  │    ├── severity.py           # Lognormal, Gamma, Pareto severity models
  │    ├── simulation.py         # MonteCarloEngine: orchestrates simulation runs
  │    ├── treaties.py           # ExcessOfLoss, StopLoss treaty structures
  │    ├── pricing.py            # TechnicalPricer: premium computation
  │    ├── risk_measures.py      # VaR, TVaR, attachment/exhaustion probabilities
  │    └── plots.py              # Ceded loss distribution visualisations
  ├── notebooks/
  │  ├── 01_xol_pricing.ipynb
  │  ├── 02_stop_loss_pricing.ipynb
  │  └── 03_sensitivity_analysis.ipynb
  ├── app/
  │  └── streamlit_app.py
  ├── tests/
  │  ├── test_treaties.py
  │  ├── test_pricing.py
  │  └── test_risk_measures.py
  ├── examples/
  │  ├── xol_pricing_example.py
  │  └── stop_loss_example.py
  └── docs/
     └── methodology.md
```

---

## Installation

Python 3.10 or later is required.

**From source (recommended for development):**

```bash
git clone https://github.com/RS-ins/reinsurance-layer-pricing-engine.git
cd reinsurance-layer-pricing-engine
pip install -e ".[dev]"
```

**Dependencies** (managed via `pyproject.toml`):

```
numpy
scipy
pandas
matplotlib
streamlit
```

---

## Quick Start Example

The following example prices a 1M xs 1M Excess-of-Loss layer using a Poisson frequency model and a Lognormal severity model over 100,000 simulated accident years.

```python
from reinsure_pricing.frequency import PoissonFrequency
from reinsure_pricing.severity import LognormalSeverity
from reinsure_pricing.treaties import ExcessOfLoss
from reinsure_pricing.simulation import MonteCarloEngine
from reinsure_pricing.pricing import TechnicalPricer

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
print(f"Technical Premium : {premium:,.0f}")
print(f"Rate on Line      : {pricer.rate_on_line(treaty_limit=5_000_000):.2%}")
```

---

## Example Output

The table below shows illustrative output for the quick start example above. All values are placeholder figures for demonstration purposes only and do not reflect any real portfolio or market data.

| Metric | Value (Illustrative) |
|---|---|
| Expected Ceded Loss | 1,243,500 |
| Standard Deviation | 876,200 |
| VaR 99% | 4,102,000 |
| TVaR 99% | 4,687,000 |
| Probability of Attachment | 34.2% |
| Probability of Exhaustion | 2.1% |
| Technical Premium | 1,718,960 |
| Rate on Line | 34.38% |

---

## Streamlit Dashboard

An interactive Streamlit dashboard is provided in `app/streamlit_app.py`. It allows users to configure treaty parameters, distribution assumptions, and loading factors through a browser-based interface and view updated results in real time.

**To launch the dashboard:**

```bash
streamlit run app/streamlit_app.py
```

The dashboard includes:

- Sidebar controls for frequency distribution type and parameters
- Sidebar controls for severity distribution type and parameters
- Treaty configuration (treaty type, retention, limit or attachment/cap)
- Loading factor inputs (expense, profit, cost of capital)
- Output panel displaying the full risk measure table and technical premium breakdown
- Ceded loss distribution histogram with VaR and TVaR annotations
- Sensitivity analysis tab for varying a single parameter across a range

---

## Roadmap

The following features are under consideration for future development:

- Reinstatement provisions for Excess-of-Loss treaties
- Aggregate annual limit (AAL) and annual aggregate deductible (AAD) structures
- Correlation modelling across lines of business
- Reinsurance programme optimisation (stacking multiple layers)
- Bootstrapped confidence intervals for all risk measures
- Export of simulation results to CSV and Excel
- Integration with external loss development triangle inputs

Contributions and suggestions are welcome via GitHub Issues.

---

## Limitations

This project is a research and educational tool. Users should be aware of the following limitations before applying it in any applied context:

- **Model risk**: The engine assumes independence between claim frequency and severity. Dependence structures (e.g., copulas) are not currently implemented.
- **Parameter uncertainty**: No parameter uncertainty is propagated through the simulation. All distribution parameters are treated as known.
- **Individual loss data**: The per-occurrence XL implementation simulates individual losses independently. Loss correlation within an accident year is not modelled.
- **Reinstatements**: Reinstatement premiums are not calculated. The treaty limit is treated as fully available throughout the year.
- **Inflation**: No loss development, tail factors, or future inflation adjustments are applied to severity inputs.
- **Calibration**: No functionality is provided for fitting distributions to historical loss data. Users are responsible for selecting and calibrating their own distributional assumptions.
- **Validation**: The model has not been independently validated against industry benchmarks or actuarial software. Output should be cross-checked against established tools before use in any professional context.

---

## Disclaimer

This project is developed as a personal side project for research and educational purposes. It is **not** intended for production use, commercial pricing decisions, regulatory filings, or any other professional actuarial application without independent review and validation by a qualified actuary.

The technical premium formula implemented here is a simplified representation of actuarial pricing methodology. It does not constitute professional actuarial advice. Results produced by this engine are illustrative only and should not be relied upon for any commercial, regulatory, or risk management purpose.

The author(s) accept no liability for decisions made on the basis of output generated by this software.

---

## License

This project is released under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2024 RS-ins

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

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
