"""
io.py
-----
Input/output utilities for the Reinsurance Layer Pricing Engine.

Provides functions for:
    - Loading historical loss data from CSV or Excel files
    - Exporting simulation results, risk measures, and pricing
      to a structured Excel report with multiple sheets

Usage:
    from reinsure_pricing.io import export_report

    export_report(
        results=results,
        rm=rm,
        pricing=pricing,
        treaty=treaty,
        path="output/pricing_report.xlsx",
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from reinsure_pricing.simulation import SimulationResults
from reinsure_pricing.risk_measures import RiskMeasures
from reinsure_pricing.pricing import PricingResult


def load_losses(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load historical loss data from a CSV or Excel file.

    The file must contain at least one of the following columns:
        - 'claim_count'  : integer claim counts per accident year
        - 'loss_amount'  : individual loss amounts

    Both columns may be present in the same file, or in separate files.
    Rows with missing values are dropped automatically.

    Parameters
    ----------
    path : str
        Path to a CSV (.csv) or Excel (.xlsx, .xls) file.

    Returns
    -------
    claim_counts : np.ndarray
        Integer array of claim counts per year. Empty if column absent.
    loss_amounts : np.ndarray
        Float array of individual loss amounts. Empty if column absent.

    Examples
    --------
    A CSV with both columns:

        year,claim_count,loss_amount
        2015,112,345000
        2015,112,890000
        2016,98,234000

    A CSV with only claim counts (one row per year):

        year,claim_count
        2015,112
        2016,98
        2017,134
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv or .xlsx")

    claim_counts = np.array([])
    loss_amounts = np.array([])

    if "claim_count" in df.columns:
        claim_counts = df["claim_count"].dropna().astype(int).values

    if "loss_amount" in df.columns:
        loss_amounts = df["loss_amount"].dropna().astype(float).values

    if len(claim_counts) == 0 and len(loss_amounts) == 0:
        raise ValueError(
            "File must contain at least one of: 'claim_count', 'loss_amount'"
        )

    return claim_counts, loss_amounts


def export_report(
        results: SimulationResults,
        rm: RiskMeasures,
        pricing: PricingResult,
        treaty,
        path: str = "outputs/pricing_report.xlsx",
        frequency=None,
        severity=None,
        bootstrap=None,
) -> str:
    """
    Export simulation results, risk measures, and pricing to an Excel report.

    Produces a multi-sheet Excel workbook with:
        - Summary     : treaty parameters, distributions, key metrics
        - Risk Measures : full VaR/TVaR table with all confidence levels
        - Pricing     : premium breakdown with all components
        - Distribution : full simulated ceded loss distribution
        - Bootstrap   : bootstrapped confidence intervals (if provided)

    Parameters
    ----------
    results : SimulationResults
        Output from MonteCarloEngine.run().
    rm : RiskMeasures
        Output from compute_risk_measures().
    pricing : PricingResult
        Output from TechnicalPricer.price().
    treaty : ExcessOfLoss or StopLoss
        The treaty object — used to extract structure parameters.
    path : str
        Output file path. Must end in .xlsx. Default is 'pricing_report.xlsx'.
    frequency : optional
        Frequency distribution object — used for summary sheet.
    severity : optional
        Severity distribution object — used for summary sheet.
    bootstrap : BootstrappedRiskMeasures, optional
        Output from bootstrap_risk_measures() — adds a Bootstrap sheet.

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    from reinsure_pricing.treaties import ExcessOfLoss, StopLoss

    path = Path(path)

    # Always save to outputs/ folder
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # If no directory specified in path, use outputs/
    if path.parent == Path("."):
        path = output_dir / path.name

    # If file already exists, append timestamp to avoid overwriting
    if path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path.parent / f"{path.stem}_{timestamp}{path.suffix}"

    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:

        # ── Sheet 1: Summary ──
        summary_rows = [
            ("Report generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("", ""),
            ("TREATY", ""),
        ]

        if isinstance(treaty, ExcessOfLoss):
            summary_rows += [
                ("Type",            "Excess-of-Loss (XL)"),
                ("Retention",       f"€{treaty.retention:,.0f}"),
                ("Limit",           f"€{treaty.limit:,.0f}"),
                ("Exhaustion point",f"€{treaty.upper_limit():,.0f}"),
            ]
            if treaty.aggregate_limit is not None:
                summary_rows.append(
                    ("Aggregate Annual Limit", f"€{treaty.aggregate_limit:,.0f}")
                )
            if treaty.aggregate_deductible is not None:
                summary_rows.append(
                    ("Annual Aggregate Deductible", f"€{treaty.aggregate_deductible:,.0f}")
                )
        elif isinstance(treaty, StopLoss):
            summary_rows += [
                ("Type",       "Stop-Loss"),
                ("Attachment", f"€{treaty.attachment:,.0f}"),
                ("Cap",        f"€{treaty.cap:,.0f}"),
            ]

        summary_rows.append(("", ""))
        summary_rows.append(("DISTRIBUTIONS", ""))

        if frequency is not None:
            summary_rows.append(("Frequency model", type(frequency).__name__))
            if hasattr(frequency, "lambda_"):
                summary_rows.append(("  λ (mean claims)", frequency.lambda_))
            elif hasattr(frequency, "mu"):
                summary_rows.append(("  μ (mean claims)", frequency.mu))
                summary_rows.append(("  φ (overdispersion)", frequency.phi))

        if severity is not None:
            summary_rows.append(("Severity model", type(severity).__name__))
            if hasattr(severity, "mu") and hasattr(severity, "sigma"):
                summary_rows.append(("  μ (log-scale mean)", severity.mu))
                summary_rows.append(("  σ (log-scale std)",  severity.sigma))
            elif hasattr(severity, "cv"):
                summary_rows.append(("  Mean", severity.mean))
                summary_rows.append(("  CV",   severity.cv))
            elif hasattr(severity, "alpha"):
                summary_rows.append(("  α (tail index)",    severity.alpha))
                summary_rows.append(("  x_m (minimum loss)", severity.x_m))

        summary_rows += [
            ("", ""),
            ("SIMULATION", ""),
            ("n_simulations",          len(results.ceded_losses)),
            ("", ""),
            ("KEY RESULTS", ""),
            ("Expected Ceded Loss",    rm.expected_ceded_loss),
            ("Technical Premium",      pricing.technical_premium),
            ("Rate on Line",           pricing.rate_on_line),
            ("Prob of Attachment",     rm.prob_attachment),
            ("Prob of Exhaustion",     rm.prob_exhaustion),
        ]

        if results.reinstatement_premiums is not None:
            summary_rows.append(
                ("Exp Reinstatement Premium", results.expected_reinstatement_premium)
            )
            summary_rows.append(
                ("Net Expected Recovery", results.net_expected_recovery)
            )

        df_summary = pd.DataFrame(summary_rows, columns=["Parameter", "Value"])
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

        # ── Sheet 2: Risk Measures ──
        risk_data = {
            "Measure": [
                "Expected Ceded Loss", "Std Deviation",
                "Coeff of Variation", "Skewness",
                "VaR 95%", "VaR 99%", "VaR 99.5%",
                "TVaR 95%", "TVaR 99%", "TVaR 99.5%",
                "Prob of Attachment", "Prob of Exhaustion",
            ],
            "Value": [
                rm.expected_ceded_loss, rm.std_deviation,
                rm.coefficient_of_variation, rm.skewness,
                rm.var_95, rm.var_99, rm.var_995,
                rm.tvar_95, rm.tvar_99, rm.tvar_995,
                rm.prob_attachment, rm.prob_exhaustion,
            ],
            "Format": [
                "currency", "currency",
                "number", "number",
                "currency", "currency", "currency",
                "currency", "currency", "currency",
                "percent", "percent",
            ],
        }
        df_risk = pd.DataFrame(risk_data)
        df_risk.to_excel(writer, sheet_name="Risk Measures", index=False)

        # ── Sheet 3: Pricing ──
        pricing_data = {
            "Component": [
                "Gross ECL",
                "Expected Reinstatement Premium",
                "Net ECL",
                "Expense Loading",
                "Profit Loading",
                "Capital Load",
                "Technical Premium",
                "Rate on Line",
            ],
            "Value": [
                pricing.expected_ceded_loss,
                pricing.expected_reinstatement_premium,
                pricing.net_expected_ceded_loss,
                pricing.expense_loading,
                pricing.profit_loading,
                pricing.capital_load,
                pricing.technical_premium,
                pricing.rate_on_line,
            ],
            "Notes": [
                "Mean simulated ceded loss",
                "Expected annual reinstatement premium income",
                "Gross ECL minus reinstatement premium",
                f"Expense load applied to Net ECL",
                f"Profit load applied to Net ECL",
                "Cost of capital × max(TVaR99 - Net ECL, 0)",
                "Sum of all components",
                "Technical Premium / Treaty Limit",
            ],
        }
        df_pricing = pd.DataFrame(pricing_data)
        df_pricing.to_excel(writer, sheet_name="Pricing", index=False)

        # ── Sheet 4: Distribution ──
        losses = results.ceded_losses
        df_dist = pd.DataFrame({
            "Annual Ceded Loss": losses,
            "Reinstatement Premium": (
                results.reinstatement_premiums
                if results.reinstatement_premiums is not None
                else np.zeros(len(losses))
            ),
        })
        df_dist.to_excel(writer, sheet_name="Distribution", index=False)

        # ── Sheet 5: Bootstrap (optional) ──
        if bootstrap is not None:
            boot_data = {
                "Measure": [
                    "ECL", "Std Deviation",
                    "VaR 95%", "VaR 99%", "VaR 99.5%",
                    "TVaR 95%", "TVaR 99%", "TVaR 99.5%",
                    "Prob Attachment", "Prob Exhaustion",
                ],
                "Point Estimate": [
                    bootstrap.ecl.point_estimate,
                    bootstrap.std.point_estimate,
                    bootstrap.var_95.point_estimate,
                    bootstrap.var_99.point_estimate,
                    bootstrap.var_995.point_estimate,
                    bootstrap.tvar_95.point_estimate,
                    bootstrap.tvar_99.point_estimate,
                    bootstrap.tvar_995.point_estimate,
                    bootstrap.prob_attachment.point_estimate,
                    bootstrap.prob_exhaustion.point_estimate,
                ],
                "CI Lower": [
                    bootstrap.ecl.ci_lower,
                    bootstrap.std.ci_lower,
                    bootstrap.var_95.ci_lower,
                    bootstrap.var_99.ci_lower,
                    bootstrap.var_995.ci_lower,
                    bootstrap.tvar_95.ci_lower,
                    bootstrap.tvar_99.ci_lower,
                    bootstrap.tvar_995.ci_lower,
                    bootstrap.prob_attachment.ci_lower,
                    bootstrap.prob_exhaustion.ci_lower,
                ],
                "CI Upper": [
                    bootstrap.ecl.ci_upper,
                    bootstrap.std.ci_upper,
                    bootstrap.var_95.ci_upper,
                    bootstrap.var_99.ci_upper,
                    bootstrap.var_995.ci_upper,
                    bootstrap.tvar_95.ci_upper,
                    bootstrap.tvar_99.ci_upper,
                    bootstrap.tvar_995.ci_upper,
                    bootstrap.prob_attachment.ci_upper,
                    bootstrap.prob_exhaustion.ci_upper,
                ],
                "Relative Width": [
                    bootstrap.ecl.relative_width,
                    bootstrap.std.relative_width,
                    bootstrap.var_95.relative_width,
                    bootstrap.var_99.relative_width,
                    bootstrap.var_995.relative_width,
                    bootstrap.tvar_95.relative_width,
                    bootstrap.tvar_99.relative_width,
                    bootstrap.tvar_995.relative_width,
                    bootstrap.prob_attachment.relative_width,
                    bootstrap.prob_exhaustion.relative_width,
                ],
                "Stable?": [
                    "Yes" if b.relative_width < 0.05 else
                    "Moderate" if b.relative_width < 0.10 else "No"
                    for b in [
                        bootstrap.ecl, bootstrap.std,
                        bootstrap.var_95, bootstrap.var_99, bootstrap.var_995,
                        bootstrap.tvar_95, bootstrap.tvar_99, bootstrap.tvar_995,
                        bootstrap.prob_attachment, bootstrap.prob_exhaustion,
                    ]
                ],
            }
            df_boot = pd.DataFrame(boot_data)
            df_boot.to_excel(writer, sheet_name="Bootstrap", index=False)

    print(f"Report saved to {path.resolve()}")
    return str(path.resolve())