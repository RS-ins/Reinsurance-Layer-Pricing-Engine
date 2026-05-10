"""
fit_from_file.py
----------------
Fits frequency and severity distributions from a CSV input file,
runs the Monte Carlo simulation with the fitted distributions,
and exports a full pricing report to Excel.

Input file format (CSV):
    year         : accident year (integer)
    claim_count  : number of claims in that year (one row per year)
    loss_amount  : individual loss amount (one row per loss)

Both columns can be in the same file. If claim_count repeats for
each loss in a year, the engine deduplicates automatically.

Usage:
    python fit_from_file.py --input data/sample_losses.csv

    python fit_from_file.py --input data/sample_losses.csv \\
        --treaty-type xl \\
        --retention 1000000 \\
        --limit 5000000 \\
        --aal 15000000 \\
        --aad 500000 \\
        --reinstatements 1 \\
        --free-reinstatements 1 \\
        --seed 42

    python fit_from_file.py --input data/sample_losses.csv \\
        --treaty-type sl \\
        --attachment 30000000 \\
        --cap 20000000
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from reinsure_pricing.frequency import PoissonFrequency
from reinsure_pricing.severity import LognormalSeverity
from reinsure_pricing.treaties import ExcessOfLoss, StopLoss, ReinstatementProvision
from reinsure_pricing.simulation import MonteCarloEngine
from reinsure_pricing.risk_measures import compute_risk_measures
from reinsure_pricing.pricing import TechnicalPricer
from reinsure_pricing.bootstrap import bootstrap_risk_measures
from reinsure_pricing.io import export_report
from reinsure_pricing.fitting import fit_frequency, fit_severity
from reinsure_pricing.plots import plot_ceded_loss_distribution
import matplotlib.pyplot as plt


def load_and_parse(path: str):
    """
    Load CSV and extract claim counts per year and individual losses.

    Handles two layouts:
    1. One row per year with claim_count, no loss_amount
    2. One row per loss with loss_amount, claim_count repeated per year

    Parameters
    ----------
    path : str
        Path to a CSV file with 'year', 'claim_count', and/or 'loss_amount' columns.

    Returns
    -------
    counts_per_year : np.ndarray or None
        Integer array of claim counts, one per accident year.
        None if 'claim_count' column is absent.
    losses : np.ndarray or None
        Float array of individual loss amounts.
        None if 'loss_amount' column is absent.
    """
    df = pd.read_csv(path)

    # Extract claim counts — deduplicate per year if needed
    if "claim_count" in df.columns and "year" in df.columns:
        counts_per_year = (
            df[["year", "claim_count"]]
            .drop_duplicates("year")
            .sort_values("year")["claim_count"]
            .values.astype(int)
        )
    elif "claim_count" in df.columns:
        counts_per_year = df["claim_count"].dropna().astype(int).values
    else:
        counts_per_year = None

    # Extract individual losses
    if "loss_amount" in df.columns:
        losses = df["loss_amount"].dropna().astype(float).values
    else:
        losses = None

    return counts_per_year, losses


def main():
    parser = argparse.ArgumentParser(
        description="Fit distributions from CSV and price a reinsurance layer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input ──
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV file with claim_count and/or loss_amount columns"
    )

    # ── Treaty type ──
    parser.add_argument(
        "--treaty-type", type=str, default="xl",
        choices=["xl", "sl"],
        help="Treaty type: 'xl' for Excess-of-Loss, 'sl' for Stop-Loss"
    )

    # ── XL parameters ──
    parser.add_argument(
        "--retention", type=float, default=1_000_000,
        help="XL retention (per-occurrence)"
    )
    parser.add_argument(
        "--limit", type=float, default=5_000_000,
        help="XL per-occurrence limit"
    )
    parser.add_argument(
        "--aal", type=float, default=None,
        help="Aggregate Annual Limit — caps total annual ceded loss (XL only)"
    )
    parser.add_argument(
        "--aad", type=float, default=None,
        help="Annual Aggregate Deductible — reinsurer pays only excess (XL only)"
    )
    parser.add_argument(
        "--reinstatements", type=int, default=0,
        help="Number of paid reinstatements (XL only)"
    )
    parser.add_argument(
        "--free-reinstatements", type=int, default=1,
        help="Number of free reinstatements (XL only)"
    )

    # ── Stop-Loss parameters ──
    parser.add_argument(
        "--attachment", type=float, default=10_000_000,
        help="Stop-Loss attachment point"
    )
    parser.add_argument(
        "--cap", type=float, default=20_000_000,
        help="Stop-Loss cap"
    )

    # ── Severity fitting ──
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Fit severity only to losses above this threshold"
    )

    # ── Pricing parameters ──
    parser.add_argument(
        "--expense-load", type=float, default=0.05,
        help="Expense load rate"
    )
    parser.add_argument(
        "--profit-load", type=float, default=0.08,
        help="Profit load rate"
    )
    parser.add_argument(
        "--cost-of-capital", type=float, default=0.10,
        help="Cost of capital rate"
    )

    # ── Simulation ──
    parser.add_argument(
        "--n-sims", type=int, default=100_000,
        help="Number of Monte Carlo simulations"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: random each run)"
    )

    # ── Output options ──
    parser.add_argument(
        "--no-bootstrap", action="store_true",
        help="Skip bootstrapped confidence intervals (faster)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib plot"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("REINSURANCE LAYER PRICING — DISTRIBUTION FITTING")
    print("=" * 65)
    print(f"\nInput file : {args.input}")

    # ─────────────────────────────────────────────
    # Load data
    # ─────────────────────────────────────────────
    print("\nLoading data...")
    counts_per_year, losses = load_and_parse(args.input)

    if counts_per_year is None and losses is None:
        print("ERROR: Input file must contain 'claim_count' and/or 'loss_amount'.")
        return

    # ─────────────────────────────────────────────
    # Fit frequency
    # ─────────────────────────────────────────────
    if counts_per_year is not None:
        print(f"\nFitting frequency ({len(counts_per_year)} years of data)...")
        freq_comp = fit_frequency(counts_per_year)
        print(freq_comp.summary())
        frequency = freq_comp.best.distribution
    else:
        print("\nNo claim count data — using Poisson(λ=100) as default.")
        frequency = PoissonFrequency(lambda_=100)
        freq_comp = None

    # ─────────────────────────────────────────────
    # Fit severity
    # ─────────────────────────────────────────────
    if losses is not None:
        n_above = (
            len(losses[losses > args.threshold])
            if args.threshold else len(losses)
        )
        print(f"\nFitting severity ({n_above} individual losses"
              + (f" above threshold €{args.threshold:,.0f}"
                 if args.threshold else "") + ")...")
        sev_comp = fit_severity(losses, threshold=args.threshold)
        print(sev_comp.summary())
        severity = sev_comp.best.distribution
        
    else:
        print("\nNo loss amount data — using Lognormal(μ=10.5, σ=1.2) as default.")
        severity = LognormalSeverity(mu=10.5, sigma=1.2)
        sev_comp = None
    

    # ─────────────────────────────────────────────
    # Build treaty
    # ─────────────────────────────────────────────
    if args.treaty_type == "xl":
        treaty = ExcessOfLoss(
            retention=args.retention,
            limit=args.limit,
            aggregate_limit=args.aal,
            aggregate_deductible=args.aad,
        )
        treaty_limit = args.limit
        treaty_desc  = (
            f"{args.limit/1e6:.0f}M xs {args.retention/1e6:.0f}M XL"
            + (f" | AAL: €{args.aal/1e6:.1f}M" if args.aal else "")
            + (f" | AAD: €{args.aad/1e6:.1f}M" if args.aad else "")
            + (f" | {args.free_reinstatements} free + {args.reinstatements}"
               f" paid reinst." if args.reinstatements > 0 else "")
        )
    else:
        treaty       = StopLoss(attachment=args.attachment, cap=args.cap)
        treaty_limit = args.cap
        treaty_desc  = (
            f"{args.cap/1e6:.0f}M xs {args.attachment/1e6:.0f}M Stop-Loss"
        )
        # reinstatements not applicable to Stop-Loss
        args.reinstatements = 0

    print(f"\nTreaty     : {treaty_desc}")

    # ─────────────────────────────────────────────
    # Reinstatement provision (XL only)
    # ─────────────────────────────────────────────
    rp = None
    if args.reinstatements > 0:
        rp = ReinstatementProvision(
            n_free=args.free_reinstatements,
            n_paid=args.reinstatements,
            original_premium=0.0,  # placeholder — updated after pricing
        )

    # ─────────────────────────────────────────────
    # Run simulation
    # ─────────────────────────────────────────────
    print(f"\nRunning Monte Carlo simulation ({args.n_sims:,} years)...")
    engine  = MonteCarloEngine(
        frequency=frequency,
        severity=severity,
        treaty=treaty,
        n_simulations=args.n_sims,
        random_state=args.seed,
    )
    results = engine.run(reinstatement=rp)
    rm      = compute_risk_measures(results, treaty_limit=treaty_limit)

    print(rm.summary())

    # ─────────────────────────────────────────────
    # Price
    # ─────────────────────────────────────────────
    pricer = TechnicalPricer(
        expected_ceded_loss=results.expected_ceded_loss,
        tvar_99=results.tvar_99,
        expense_load=args.expense_load,
        profit_load=args.profit_load,
        cost_of_capital=args.cost_of_capital,
        expected_reinstatement_premium=results.expected_reinstatement_premium,
    )
    pricing = pricer.price(treaty_limit=treaty_limit)
    print(pricing.summary())

    # ─────────────────────────────────────────────
    # Bootstrap
    # ─────────────────────────────────────────────
    boot = None
    if not args.no_bootstrap:
        print("\nRunning bootstrapped confidence intervals...")
        boot = bootstrap_risk_measures(
            results,
            treaty_limit=treaty_limit,
            n_bootstrap=1_000,
        )
        print(boot.summary())

    # ─────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────
    print("\nExporting report...")
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
    # Plot
    # ─────────────────────────────────────────────
    if not args.no_plot:
        plot_ceded_loss_distribution(
            results=results,
            risk_measures=rm,
            treaty_limit=treaty_limit,
            title=f"{treaty_desc} — Fitted Model",
        )
        plt.show()


if __name__ == "__main__":
    main()