"""
pricing.py
----------
Technical premium computation for reinsurance layers.

The TechnicalPricer takes the output of a Monte Carlo simulation and
computes a risk-adjusted technical premium using a cost-of-capital
loading approach consistent with Solvency II principles.

The base premium formula is:

    Technical Premium = ECL
                      + expense_load  × ECL
                      + profit_load   × ECL
                      + cost_of_capital × max(TVaR99 - ECL, 0)

Phase 4 addition — Reinstatement Premium Adjustment:

When reinstatement provisions are active, the technical premium is
adjusted to account for the expected reinstatement premium income
the reinsurer will receive from the cedant. The net cost to the
cedant is:

    Net ECL = ECL - E[Reinstatement Premium]

The technical premium is computed on the Net ECL, and the loadings
are applied to the net figure. This reflects the economic reality
that reinstatement premiums partially offset the reinsurer's
expected loss payments.

The pricing result reports both gross and net figures so the user
can see the full impact of the reinstatement provision.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PricingResult:
    """
    Container for the output of a technical pricing calculation.

    Stores all premium components separately so the user can inspect
    how the technical premium was built up and identify which loading
    drives the result.

    Parameters
    ----------
    expected_ceded_loss : float
        Gross ECL — mean simulated ceded loss before reinstatement
        premium adjustment.
    expected_reinstatement_premium : float
        Expected annual reinstatement premium payable by the cedant.
        Zero if no reinstatement provision is active.
    net_expected_ceded_loss : float
        Net ECL after subtracting expected reinstatement premium.
        This is the base for all loadings when reinstatements are active.
    expense_loading : float
        Absolute expense load in currency units (expense_load × net ECL).
    profit_loading : float
        Absolute profit load in currency units (profit_load × net ECL).
    capital_load : float
        Absolute capital load in currency units
        (cost_of_capital × max(TVaR99 - net ECL, 0)).
    technical_premium : float
        Sum of all components — the final risk-adjusted premium.
    rate_on_line : float
        Technical premium as a fraction of the treaty limit.
    treaty_limit : float
        Treaty limit used to compute the rate on line.
    """

    expected_ceded_loss:            float
    expected_reinstatement_premium: float
    net_expected_ceded_loss:        float
    expense_loading:                float
    profit_loading:                 float
    capital_load:                   float
    technical_premium:              float
    rate_on_line:                   float
    treaty_limit:                   float

    def summary(self) -> str:
        """Return a formatted breakdown of all premium components."""
        lines = [
            f"Gross ECL            : {self.expected_ceded_loss:>15,.0f}",
        ]
        if self.expected_reinstatement_premium > 0:
            lines += [
                f"Exp Reinst Premium   : {self.expected_reinstatement_premium:>15,.0f}",
                f"Net ECL              : {self.net_expected_ceded_loss:>15,.0f}",
            ]
        lines += [
            f"Expense Loading      : {self.expense_loading:>15,.0f}",
            f"Profit Loading       : {self.profit_loading:>15,.0f}",
            f"Capital Load         : {self.capital_load:>15,.0f}",
            f"{'─' * 40}",
            f"Technical Premium    : {self.technical_premium:>15,.0f}",
            f"Rate on Line         : {self.rate_on_line:>15.2%}",
        ]
        return "\n".join(lines)


class TechnicalPricer:
    """
    Computes a risk-adjusted technical premium from simulation results.

    The pricing formula follows a cost-of-capital approach:

        Premium = net_ECL
                + expense_load  × net_ECL
                + profit_load   × net_ECL
                + cost_of_capital × max(TVaR99 - net_ECL, 0)

    where net_ECL = ECL - E[reinstatement_premium].

    When no reinstatement provision is active, net_ECL = ECL and
    the formula reduces to the original four-component premium.

    Parameters
    ----------
    expected_ceded_loss : float
        Mean simulated ceded loss from MonteCarloEngine.run().
        Must be non-negative.
    tvar_99 : float
        TVaR at 99% confidence from SimulationResults.tvar_99.
        Used as the capital proxy.
    expense_load : float
        Proportional expense load as a fraction of net ECL.
        Default is 0.05 (5%).
    profit_load : float
        Target profit margin as a fraction of net ECL.
        Default is 0.08 (8%).
    cost_of_capital : float
        Required return on risk capital. Default is 0.10 (10%).
    expected_reinstatement_premium : float, optional
        Expected annual reinstatement premium from
        SimulationResults.expected_reinstatement_premium.
        Default is 0.0 (no reinstatement provision).
    """

    def __init__(self,
                 expected_ceded_loss:            float,
                 tvar_99:                        float,
                 expense_load:                   float = 0.05,
                 profit_load:                    float = 0.08,
                 cost_of_capital:                float = 0.10,
                 expected_reinstatement_premium: float = 0.0):

        if expected_ceded_loss < 0:
            raise ValueError("expected_ceded_loss must be non-negative")
        if not 0 <= expense_load < 1:
            raise ValueError("expense_load must be between 0 and 1")
        if not 0 <= profit_load < 1:
            raise ValueError("profit_load must be between 0 and 1")
        if not 0 <= cost_of_capital < 1:
            raise ValueError("cost_of_capital must be between 0 and 1")
        if expected_reinstatement_premium < 0:
            raise ValueError("expected_reinstatement_premium must be non-negative")

        self.expected_ceded_loss            = expected_ceded_loss
        self.tvar_99                        = tvar_99
        self.expense_load                   = expense_load
        self.profit_load                    = profit_load
        self.cost_of_capital                = cost_of_capital
        self.expected_reinstatement_premium = expected_reinstatement_premium

    @property
    def net_ecl(self) -> float:
        """
        Net ECL after subtracting expected reinstatement premium.

        This is the base for all loadings. When no reinstatement
        provision is active, net_ecl == expected_ceded_loss.
        """
        return max(self.expected_ceded_loss - self.expected_reinstatement_premium, 0.0)

    def technical_premium(self) -> float:
        """
        Compute the technical premium.

        Builds the premium on the net ECL base:
            net_ECL + expense loading + profit loading + capital load

        Returns
        -------
        float
            The risk-adjusted technical premium in currency units.
        """
        ecl     = self.net_ecl
        expense = self.expense_load * ecl
        profit  = self.profit_load * ecl

        # Capital load applied to unexpected loss above net ECL.
        # max(..., 0) guards against degenerate cases where TVaR < net_ECL.
        capital = self.cost_of_capital * max(self.tvar_99 - ecl, 0)

        return ecl + expense + profit + capital

    def rate_on_line(self, treaty_limit: float) -> float:
        """
        Rate on Line — technical premium as a fraction of treaty limit.

        Parameters
        ----------
        treaty_limit : float
            The maximum ceded loss under the treaty.

        Returns
        -------
        float
            Technical premium divided by treaty limit.
        """
        return self.technical_premium() / treaty_limit

    def price(self, treaty_limit: float) -> PricingResult:
        """
        Compute the full pricing breakdown and return a PricingResult.

        Parameters
        ----------
        treaty_limit : float
            Treaty limit used to compute the rate on line.

        Returns
        -------
        PricingResult
            Dataclass containing all premium components, net figures,
            and the rate on line.
        """
        ecl     = self.net_ecl
        expense = self.expense_load * ecl
        profit  = self.profit_load * ecl
        capital = self.cost_of_capital * max(self.tvar_99 - ecl, 0)
        premium = ecl + expense + profit + capital

        return PricingResult(
            expected_ceded_loss            = self.expected_ceded_loss,
            expected_reinstatement_premium = self.expected_reinstatement_premium,
            net_expected_ceded_loss        = ecl,
            expense_loading                = expense,
            profit_loading                 = profit,
            capital_load                   = capital,
            technical_premium              = premium,
            rate_on_line                   = premium / treaty_limit,
            treaty_limit                   = treaty_limit,
        )