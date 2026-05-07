# Methodology

## Reinsurance Layer Pricing Engine

This document describes the mathematical foundations of the engine, covering the collective risk model, treaty structures, risk measure definitions, the technical pricing formula, reinstatement provisions, and bootstrapped confidence intervals.

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

### 4.2 Aggregate Annual Limit (AAL)

The AAL caps the reinsurer's total annual ceded loss across all occurrences. Let $M$ denote the AAL. The per-claim ceded amounts are adjusted so that:

$$\sum_{i=1}^{N} C_i^{\text{AAL}} \leq M$$

The AAL is applied claim by claim using cumulative sums. Let $\tilde{C}_i$ denote the cumulative ceded loss before claim $i$:

$$\tilde{C}_i = \sum_{j=1}^{i-1} C_j^{\text{AAL}}$$

The ceded amount for claim $i$ after the AAL is:

$$C_i^{\text{AAL}} = \min\bigl(C_i,\ \max(M - \tilde{C}_i,\ 0)\bigr)$$

This ensures the claim that straddles the AAL threshold is **partially ceded** rather than simply zeroed out. This is actuarially correct and allows precise identification of which individual claims consumed the AAL — which is required for reinstatement premium calculations.

### 4.3 Annual Aggregate Deductible (AAD)

The AAD is a minimum annual ceded loss threshold below which the reinsurer pays nothing. Let $D$ denote the AAD. After applying the per-occurrence treaty and AAL:

$$C^{\text{net}} = \max\bigl(\sum_{i} C_i^{\text{AAL}} - D,\ 0\bigr)$$

The AAD is deducted from the first claims in order using a cumulative deduction approach, so only the excess over $D$ is paid by the reinsurer.

### 4.4 Stop-Loss

The Stop-Loss treaty applies to the **aggregate annual loss** $S$. For a treaty with attachment point $A$ and cap $M$:

$$C = \min\bigl(\max(S - A,\ 0),\ M\bigr)$$

Unlike the per-occurrence XL, the Stop-Loss protects against bad aggregate years regardless of the number or size of individual claims. It is typically used to protect against frequency risk — years where an unusually large number of moderate claims accumulate into a damaging aggregate.

---

## 5. Reinstatement Premiums

After a loss event exhausts the per-occurrence limit, the cedant can reinstate the limit by paying an additional premium to the reinsurer. This is standard market practice for XL treaties.

The standard structure implemented is **1 free + 1 paid reinstatement at 100% pro rata as to time**.

### 5.1 Free Reinstatement

The first exhaustion of the per-occurrence limit triggers an automatic reinstatement at no cost to the cedant. The reinsurer restores the full limit without charging an additional premium.

### 5.2 Paid Reinstatement

Subsequent exhaustions trigger a paid reinstatement. The reinstatement premium is computed pro rata as to time — scaled by the fraction of the accident year remaining at the time of the reinstatement:

$$RP_k = P \cdot \left(1 - \frac{k}{n_{\text{eff}} + 1}\right)$$

where:
- $P$ is the original technical premium
- $k$ is the exhaustion number (starting from 1)
- $n_{\text{eff}} = \min(n_{\text{reinstatements}},\ n_{\text{free}} + n_{\text{paid}})$

The first $n_{\text{free}}$ exhaustions are free ($RP_k = 0$ for $k \leq n_{\text{free}}$). The total reinstatement premium for a year with $n$ exhaustions is:

$$RP_{\text{total}} = \sum_{k=n_{\text{free}}+1}^{n_{\text{eff}}} P \cdot \left(1 - \frac{k}{n_{\text{eff}} + 1}\right)$$

### 5.3 Timing Approximation

In the Monte Carlo simulation, the exact timing of each exhaustion within the accident year is not tracked. The pro rata fraction is approximated by assuming exhaustions occur at evenly spaced intervals within the year. This is a simplification — in practice, loss event timing would be modelled explicitly using event dates.

### 5.4 Net ECL

The expected annual reinstatement premium flows back to the reinsurer, reducing the net cost of the layer:

$$\text{Net ECL} = \mathbb{E}[C] - \mathbb{E}[RP_{\text{total}}]$$

The technical premium is computed on the Net ECL rather than the gross ECL.

---

## 6. Monte Carlo Simulation

The engine simulates $n$ accident years ($n = 100{,}000$ by default). For each year $j = 1, \ldots, n$:

1. Draw claim count: $N_j \sim \text{Frequency}(\theta_f)$
2. If $N_j > 0$, draw individual losses: $X_{j,1}, \ldots, X_{j,N_j} \sim \text{Severity}(\theta_s)$
3. Apply the treaty (including AAL and AAD if active) to obtain ceded loss $C_j$
4. If a reinstatement provision is active, compute the reinstatement premium $RP_j$
5. Store $C_j$ and $RP_j$

The output is the empirical distribution $\{C_1, \ldots, C_n\}$ and $\{RP_1, \ldots, RP_n\}$ from which all risk measures are estimated.

For the Stop-Loss treaty, the simulation is vectorised by drawing all individual losses across all years in a single numpy call, then splitting into per-year chunks using cumulative claim counts. This reduces runtime significantly compared to a Python loop.

---

## 7. Risk Measures

All risk measures are computed from the empirical ceded loss distribution $\{C_1, \ldots, C_n\}$.

### 7.1 Expected Ceded Loss (ECL)

$$\text{ECL} = \hat{\mathbb{E}}[C] = \frac{1}{n} \sum_{j=1}^{n} C_j$$

The pure premium — the minimum the reinsurer must charge to break even in expectation.

### 7.2 Value at Risk (VaR)

$$\text{VaR}_\alpha(C) = \hat{F}_C^{-1}(\alpha)$$

The empirical $\alpha$-quantile of the ceded loss distribution. VaR is a threshold measure — it does not describe the severity of losses beyond the threshold.

### 7.3 Tail Value at Risk (TVaR)

$$\text{TVaR}_\alpha(C) = \mathbb{E}\bigl[C \mid C > \text{VaR}_\alpha(C)\bigr]$$

Also known as Conditional Tail Expectation (CTE) or Expected Shortfall (ES). TVaR is always $\geq$ VaR at the same confidence level. TVaR is a **coherent risk measure** unlike VaR which violates subadditivity in general.

### 7.4 Probability of Attachment

$$p_{\text{att}} = P(C > 0) = \frac{1}{n} \sum_{j=1}^{n} \mathbf{1}[C_j > 0]$$

The fraction of simulated years where the layer was triggered.

### 7.5 Probability of Exhaustion

$$p_{\text{exh}} = P(C \geq L) = \frac{1}{n} \sum_{j=1}^{n} \mathbf{1}[C_j \geq L]$$

The fraction of simulated years where the full treaty limit was consumed.

### 7.6 Coefficient of Variation

$$CV = \frac{\hat{\sigma}(C)}{\hat{\mathbb{E}}[C]}$$

A scale-free measure of volatility. Useful for comparing layers of different sizes.

### 7.7 Skewness

$$\hat{\gamma} = \frac{1}{n} \sum_{j=1}^{n} \left(\frac{C_j - \hat{\mathbb{E}}[C]}{\hat{\sigma}(C)}\right)^3$$

The third standardised central moment. Positive skewness confirms a heavy right tail.

---

## 8. Technical Pricing Formula

The engine implements a **cost-of-capital** pricing approach, consistent with Solvency II principles.

$$\text{Technical Premium} = \text{Net ECL} + e \cdot \text{Net ECL} + p \cdot \text{Net ECL} + r_c \cdot \max(\text{TVaR}_{99} - \text{Net ECL},\ 0)$$

The four components are:

| Component | Formula | Description |
|---|---|---|
| Pure premium | Net ECL | Expected annual net ceded loss after reinstatement premiums |
| Expense load | $e \cdot \text{Net ECL}$ | Proportional load for acquisition and admin costs |
| Profit load | $p \cdot \text{Net ECL}$ | Target profit margin as a fraction of Net ECL |
| Capital load | $r_c \cdot \max(\text{TVaR}_{99} - \text{Net ECL},\ 0)$ | Required return on risk capital |

### 8.1 Capital Load Rationale

The capital load compensates the reinsurer's shareholders for holding **risk capital** against unexpected losses. The unexpected loss is proxied by:

$$\text{Capital requirement} = \text{TVaR}_{99} - \text{Net ECL}$$

The cost of capital $r_c$ is the minimum return shareholders require on that capital — typically set to the reinsurer's internal hurdle rate (10% in the default parameterisation).

### 8.2 Rate on Line

$$\text{ROL} = \frac{\text{Technical Premium}}{L}$$

The standard reinsurance market metric for comparing layer pricing. A ROL of 10% means the reinsurer charges 10% of the limit as annual premium.

---

## 9. Bootstrapped Confidence Intervals

### 9.1 Motivation

Risk measures computed from Monte Carlo simulation are **estimates** — they carry sampling uncertainty that depends on the number of simulations. A VaR 99% estimate from 100,000 simulations is more stable than one from 10,000 simulations, but neither is exact. Bootstrapped confidence intervals quantify this uncertainty and tell the user whether more simulations are needed.

### 9.2 Method

The non-parametric bootstrap resamples the simulated ceded loss array with replacement $B$ times (default $B = 1{,}000$). For each resample $b = 1, \ldots, B$:

1. Draw $n$ losses with replacement from $\{C_1, \ldots, C_n\}$
2. Compute the risk measure of interest on the resample

The 95% confidence interval is the 2.5th and 97.5th percentiles of the $B$ bootstrap estimates:

$$\text{CI}_{95\%} = \bigl[\hat{\theta}_{(0.025)},\ \hat{\theta}_{(0.975)}\bigr]$$

### 9.3 Interpretation

The **relative width** of the confidence interval measures estimate stability:

$$\text{Rel Width} = \frac{\text{CI upper} - \text{CI lower}}{\text{point estimate}}$$

| Rel Width | Interpretation |
|---|---|
| < 5% | Stable — the estimate can be trusted |
| 5–10% | Moderate uncertainty — acceptable for most purposes |
| > 10% | High uncertainty — consider increasing n_simulations |

Tail measures (VaR 99.5%, TVaR 99.5%, Prob Exhaustion) typically have wider intervals than central measures (ECL, VaR 95%) because fewer observations fall in the extreme tail.

### 9.4 Memory Considerations

The bootstrap implementation draws all resamples simultaneously as a 2D numpy array of shape $(B, n)$. For $B = 1{,}000$ and $n = 100{,}000$ this requires approximately 800MB of memory. Reduce $B$ or $n$ if memory is constrained.

---

## 10. Limitations and Model Risk

**Independence assumption.** Frequency and severity are assumed independent. In practice, large events often produce both more claims and larger individual losses, introducing positive dependence that this model ignores.

**Parameter uncertainty.** All distribution parameters are treated as known with certainty. No parameter uncertainty is propagated through the simulation. In practice, parameter estimation error can be a significant source of pricing uncertainty, particularly in the tail.

**Reinstatement timing approximation.** The pro rata fraction uses a uniform timing approximation — exhaustions are assumed to occur at evenly spaced intervals within the year. In practice, loss event timing would be modelled explicitly using event dates.

**No distribution fitting.** The engine takes distribution parameters as inputs. No functionality is provided for fitting distributions to historical loss data. Users are responsible for calibration.

**Homogeneous portfolio.** All claims are drawn from the same severity distribution. In reality, a portfolio contains heterogeneous risks with different expected loss severities.

**Not validated.** The engine has not been validated against commercial actuarial software or industry benchmarks. Results are illustrative only and should not be used for professional pricing without independent review by a qualified actuary.

---

## 11. References

- Klugman, S.A., Panjer, H.H., Willmot, G.E. (2012). *Loss Models: From Data to Decisions*. Wiley.
- McNeil, A.J., Frey, R., Embrechts, P. (2015). *Quantitative Risk Management*. Princeton University Press.
- Embrechts, P., Klüppelberg, C., Mikosch, T. (1997). *Modelling Extremal Events*. Springer.
- European Insurance and Occupational Pensions Authority (2015). *Solvency II Technical Specifications*.
- Daykin, C.D., Pentikäinen, T., Pesonen, M. (1994). *Practical Risk Theory for Actuaries*. Chapman & Hall.
- Efron, B., Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
