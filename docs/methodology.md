# Methodology

## Reinsurance Layer Pricing Engine

This document describes the mathematical foundations of the engine, covering the collective risk model, treaty structures, risk measure definitions, and the technical pricing formula.

---

## 1. Collective Risk Model

The engine implements the **frequency-severity collective risk model**, also known as the compound distribution model. This is the standard framework for modelling aggregate insurance losses.

For a single accident year, the aggregate gross loss is:

$$S = \sum_{i=1}^{N} X_i$$

where:

- $N$ is the **claim count** (frequency), a non-negative integer-valued random variable
- $X_i$ are the **individual loss amounts** (severity), assumed independent and identically distributed, and independent of $N$

The key actuarial quantities are:

$$\mathbb{E}[S] = \mathbb{E}[N] \cdot \mathbb{E}[X]$$

$$\text{Var}(S) = \mathbb{E}[N] \cdot \text{Var}(X) + \text{Var}(N) \cdot \mathbb{E}[X]^2$$

The second equation shows that aggregate loss variance has two components: **severity risk** (first term) and **frequency risk** (second term). For a Poisson frequency distribution $\text{Var}(N) = \mathbb{E}[N]$, simplifying the expression. For the Negative Binomial, $\text{Var}(N) > \mathbb{E}[N]$ (overdispersion), inflating the aggregate variance beyond the Poisson case.

---

## 2. Frequency Distributions

### 2.1 Poisson

The Poisson distribution is the natural starting point for claim frequency modelling. It arises when claims occur independently at a constant rate $\lambda > 0$:

$$P(N = k) = \frac{e^{-\lambda} \lambda^k}{k!}, \quad k = 0, 1, 2, \ldots$$

$$\mathbb{E}[N] = \text{Var}(N) = \lambda$$

The Poisson distribution exhibits **equidispersion** — mean equals variance. This is a strong assumption that may not hold in practice when the portfolio exhibits heterogeneity across insureds.

### 2.2 Negative Binomial

The Negative Binomial generalises the Poisson by allowing **overdispersion** — variance exceeding the mean. This is appropriate when the claim rate varies across the portfolio (e.g. due to unobserved risk factors).

The engine parameterises the Negative Binomial using the actuarial $(\mu, \phi)$ convention:

$$\mathbb{E}[N] = \mu, \quad \text{Var}(N) = \mu + \phi \mu^2$$

where $\phi > 0$ is the overdispersion parameter. As $\phi \to 0$ the distribution converges to Poisson($\mu$).

The conversion to numpy's $(n, p)$ parameterisation is:

$$n = \frac{1}{\phi}, \quad p = \frac{n}{n + \mu}$$

---

## 3. Severity Distributions

### 3.1 Lognormal

The lognormal distribution is the most widely used severity model in non-life insurance. It arises naturally when individual losses are the product of many independent multiplicative factors.

If $\log X \sim \mathcal{N}(\mu, \sigma^2)$, then:

$$\mathbb{E}[X] = e^{\mu + \sigma^2/2}$$

$$\text{Var}(X) = (e^{\sigma^2} - 1) \cdot e^{2\mu + \sigma^2}$$

The parameter $\sigma$ controls the heaviness of the tail — higher $\sigma$ produces more extreme losses and a heavier right tail. For a high-excess XL layer, $\sigma$ is typically the most influential parameter in the pricing output.

### 3.2 Gamma

The Gamma distribution is a flexible, right-skewed severity model with a lighter tail than the lognormal. The engine parameterises it using the actuarial $(mean, CV)$ convention, where $CV = \sigma / \mu$ is the coefficient of variation:

$$\text{shape} = k = \frac{1}{CV^2}, \quad \text{scale} = \theta = mean \cdot CV^2$$

so that $\mathbb{E}[X] = mean$ and $\text{Var}(X) = (CV \cdot mean)^2$.

### 3.3 Pareto

The single-parameter Pareto distribution models heavy-tailed severity with a power-law tail. It is appropriate for catastrophic loss modelling where the probability of extreme events is significantly higher than the lognormal or gamma would predict.

$$F(x) = 1 - \left(\frac{x_m}{x}\right)^\alpha, \quad x \geq x_m$$

$$\mathbb{E}[X] = \frac{\alpha x_m}{\alpha - 1}, \quad \alpha > 1$$

where $\alpha$ is the tail index and $x_m > 0$ is the minimum possible loss. Lower $\alpha$ produces a heavier tail and more extreme losses. The engine samples using the inverse CDF method:

$$X = \frac{x_m}{U^{1/\alpha}}, \quad U \sim \text{Uniform}(0, 1)$$

This guarantees all sampled losses satisfy $X \geq x_m$ by construction.

---

## 4. Treaty Structures

### 4.1 Per-Occurrence Excess-of-Loss (XL)

The XL treaty applies to **individual losses**. For a treaty with retention $R$ and limit $L$, the ceded amount per occurrence is:

$$C_i = \min\bigl(\max(X_i - R,\ 0),\ L\bigr)$$

The total annual ceded loss is the sum of per-occurrence ceded amounts:

$$C = \sum_{i=1}^{N} C_i$$

The treaty layer spans from $R$ (attachment) to $R + L$ (exhaustion point). Losses below $R$ are fully retained by the cedant. Losses above $R + L$ are partially retained above the exhaustion point.

### 4.2 Stop-Loss

The Stop-Loss treaty applies to the **aggregate annual loss** $S$. For a treaty with attachment point $A$ and cap $M$:

$$C = \min\bigl(\max(S - A,\ 0),\ M\bigr)$$

Unlike the per-occurrence XL, the Stop-Loss protects against bad aggregate years regardless of the number or size of individual claims. It is typically used to protect against frequency risk — years where an unusually large number of moderate claims accumulate into a damaging aggregate.

---

## 5. Monte Carlo Simulation

The engine simulates $n$ accident years ($n = 100{,}000$ by default). For each year $j = 1, \ldots, n$:

1. Draw claim count: $N_j \sim \text{Frequency}(\theta_f)$
2. If $N_j > 0$, draw individual losses: $X_{j,1}, \ldots, X_{j,N_j} \sim \text{Severity}(\theta_s)$
3. Apply the treaty to obtain ceded loss $C_j$
4. Store $C_j$

The output is the empirical distribution $\{C_1, \ldots, C_n\}$ from which all risk measures are estimated.

For the Stop-Loss treaty, the simulation is vectorised by drawing all individual losses across all years in a single numpy call, then splitting into per-year chunks using cumulative claim counts. This reduces runtime from $O(n)$ Python loop iterations to a single array operation.

---

## 6. Risk Measures

All risk measures are computed from the empirical ceded loss distribution $\{C_1, \ldots, C_n\}$.

### 6.1 Expected Ceded Loss (ECL)

$$\text{ECL} = \hat{\mathbb{E}}[C] = \frac{1}{n} \sum_{j=1}^{n} C_j$$

The pure premium — the minimum the reinsurer must charge to break even in expectation.

### 6.2 Value at Risk (VaR)

$$\text{VaR}_\alpha(C) = \hat{F}_C^{-1}(\alpha)$$

The empirical $\alpha$-quantile of the ceded loss distribution. VaR$_{99\%}$ is the loss level exceeded in only 1% of simulated years. It is a threshold measure — it does not describe the severity of losses beyond the threshold.

### 6.3 Tail Value at Risk (TVaR)

$$\text{TVaR}_\alpha(C) = \mathbb{E}\bigl[C \mid C > \text{VaR}_\alpha(C)\bigr]$$

Also known as Conditional Tail Expectation (CTE) or Expected Shortfall (ES). TVaR answers the question: given that losses exceed the VaR threshold, what is the average outcome? TVaR is always $\geq$ VaR at the same confidence level. The gap TVaR$_{99}$ $-$ VaR$_{99}$ measures tail severity — how much worse outcomes get once the threshold is breached.

TVaR is a **coherent risk measure** (satisfying subadditivity, monotonicity, positive homogeneity, and translation invariance), unlike VaR which violates subadditivity in general.

### 6.4 Probability of Attachment

$$p_{\text{att}} = P(C > 0) = \frac{1}{n} \sum_{j=1}^{n} \mathbf{1}[C_j > 0]$$

The fraction of simulated years where the layer is triggered. For high-excess XL layers this is typically well below 50%.

### 6.5 Probability of Exhaustion

$$p_{\text{exh}} = P(C \geq L) = \frac{1}{n} \sum_{j=1}^{n} \mathbf{1}[C_j \geq L]$$

The fraction of simulated years where the full treaty limit is consumed. A high exhaustion probability signals a heavily exposed layer.

### 6.6 Coefficient of Variation

$$CV = \frac{\hat{\sigma}(C)}{\hat{\mathbb{E}}[C]}$$

A scale-free measure of volatility. Useful for comparing layers of different sizes.

### 6.7 Skewness

$$\hat{\gamma} = \frac{1}{n} \sum_{j=1}^{n} \left(\frac{C_j - \hat{\mathbb{E}}[C]}{\hat{\sigma}(C)}\right)^3$$

The third standardised central moment. Positive skewness (typical for reinsurance layers) confirms a heavy right tail — extreme years are significantly worse than the average.

---

## 7. Technical Pricing Formula

The engine implements a **cost-of-capital** pricing approach, consistent with Solvency II principles and standard actuarial practice for risk-adjusted pricing.

$$\text{Technical Premium} = \text{ECL} + e \cdot \text{ECL} + p \cdot \text{ECL} + r_c \cdot \max(\text{TVaR}_{99} - \text{ECL},\ 0)$$

The four components are:

| Component | Formula | Description |
|---|---|---|
| Pure premium | ECL | Expected annual ceded loss — the break-even floor |
| Expense load | $e \cdot \text{ECL}$ | Proportional load for acquisition and admin costs |
| Profit load | $p \cdot \text{ECL}$ | Target profit margin as a fraction of ECL |
| Capital load | $r_c \cdot \max(\text{TVaR}_{99} - \text{ECL},\ 0)$ | Required return on risk capital |

### 7.1 Capital Load Rationale

The capital load compensates the reinsurer's shareholders for holding **risk capital** against unexpected losses. The unexpected loss is proxied by:

$$\text{Capital requirement} = \text{TVaR}_{99} - \text{ECL}$$

This represents the average loss in the worst 1% of years in excess of the expected loss. The expected loss is funded by the pure premium; the unexpected loss requires capital backing. The cost of capital $r_c$ is the minimum return shareholders require on that capital — typically set to the reinsurer's internal hurdle rate (10% in the default parameterisation).

### 7.2 Rate on Line

The Rate on Line (ROL) is the standard reinsurance market metric for comparing layer pricing:

$$\text{ROL} = \frac{\text{Technical Premium}}{L}$$

where $L$ is the treaty limit. A ROL of 10% means the reinsurer charges 10% of the limit as annual premium.

---

## 8. Limitations and Model Risk

This engine is an educational and research tool. Key limitations include:

**Independence assumption.** Frequency and severity are assumed independent. In practice, large events often produce both more claims and larger individual losses (e.g. catastrophes), introducing positive dependence that this model ignores.

**Parameter uncertainty.** All distribution parameters are treated as known with certainty. No parameter uncertainty is propagated through the simulation. In practice, parameter estimation error can be a significant source of pricing uncertainty, particularly in the tail.

**No reinstatements.** The treaty limit is treated as fully available throughout the accident year. XL treaties typically include reinstatement provisions that restore the limit after a loss event in exchange for an additional premium.

**No aggregate limits.** The current XL implementation has no Aggregate Annual Limit (AAL). In practice, reinsurers cap their total annual exposure across all occurrences. The AAL implementation is planned for Phase 4.

**Homogeneous portfolio.** All claims are drawn from the same severity distribution. In reality, a portfolio contains heterogeneous risks with different expected loss severities.

**No tail fitting.** The engine takes distribution parameters as inputs. No functionality is provided for fitting distributions to historical loss data. Users are responsible for calibration.

---

## 9. References

The methodology draws on standard actuarial and risk management literature:

- Klugman, S.A., Panjer, H.H., Willmot, G.E. (2012). *Loss Models: From Data to Decisions*. Wiley.
- McNeil, A.J., Frey, R., Embrechts, P. (2015). *Quantitative Risk Management*. Princeton University Press.
- Embrechts, P., Klüppelberg, C., Mikosch, T. (1997). *Modelling Extremal Events*. Springer.
- European Insurance and Occupational Pensions Authority (2015). *Solvency II Technical Specifications*.
- Daykin, C.D., Pentikäinen, T., Pesonen, M. (1994). *Practical Risk Theory for Actuaries*. Chapman & Hall.
