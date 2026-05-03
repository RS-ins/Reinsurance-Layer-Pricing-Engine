"""
generate_scenarios.py
---------------------
Pre-computes all pricing scenarios and saves them to app/scenarios.json.
The JSON is loaded by dashboard.html for instant client-side rendering.

Run with:
    python app/generate_scenarios.py

This will take approximately 5-10 minutes depending on your machine.
Progress is printed to the console.
"""

import json
import time
import itertools
import numpy as np

from reinsure_pricing.frequency import PoissonFrequency
from reinsure_pricing.severity import LognormalSeverity
from reinsure_pricing.treaties import ExcessOfLoss, StopLoss
from reinsure_pricing.simulation import MonteCarloEngine
from reinsure_pricing.risk_measures import compute_risk_measures
from reinsure_pricing.pricing import TechnicalPricer

# ─────────────────────────────────────────────
# Fixed parameters
# ─────────────────────────────────────────────
MU           = 10.5
EXPENSE_LOAD = 0.05
PROFIT_LOAD  = 0.08
N_SIMS       = 20_000
SEED         = 42


# ─────────────────────────────────────────────
# Histogram settings for client-side distribution chart
# ─────────────────────────────────────────────
HIST_BINS = 40


def _extract_ceded_losses(results):
    """Return the simulated ceded-loss vector from the engine result object.

    The exact attribute name can vary by implementation, so this keeps the
    dashboard generator robust while still failing loudly if the engine does
    not expose simulation-level ceded losses.
    """
    candidates = (
        "ceded_losses",
        "ceded_loss",
        "simulated_ceded_losses",
        "aggregate_ceded_losses",
        "losses",
        "treaty_losses",
    )
    for name in candidates:
        if hasattr(results, name):
            values = np.asarray(getattr(results, name), dtype=float)
            if values.ndim == 1 and values.size > 0:
                return values

    if isinstance(results, dict):
        for name in candidates:
            if name in results:
                values = np.asarray(results[name], dtype=float)
                if values.ndim == 1 and values.size > 0:
                    return values

    raise AttributeError(
        "MonteCarloEngine.run() result must expose a one-dimensional ceded "
        "loss vector, e.g. results.ceded_losses, so dashboard.html can draw "
        "the simulation histogram."
    )


def build_histogram(results, bins=HIST_BINS):
    """Build compact histogram data for dashboard.html."""
    losses = _extract_ceded_losses(results)
    upper = float(np.percentile(losses, 99.5))
    clipped = np.clip(losses, 0, upper)
    counts, edges = np.histogram(clipped, bins=bins, range=(0, upper))

    return {
        "bin_edges": [round(float(x), 0) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
        "clipped_at": round(upper, 0),
    }

# ─────────────────────────────────────────────
# XL parameter grid
# ─────────────────────────────────────────────
XL_RETENTIONS = [500_000, 750_000, 1_000_000, 1_500_000, 2_000_000]
XL_LIMITS     = [2_000_000, 3_000_000, 5_000_000, 7_000_000, 10_000_000]
SIGMAS        = [0.8, 1.0, 1.2, 1.4, 1.6]
LAMBDAS       = [60, 90, 120, 150, 180]
COCS          = [0.05, 0.10, 0.15, 0.20]

# ─────────────────────────────────────────────
# Stop-Loss parameter grid
# ─────────────────────────────────────────────
SL_ATTACHMENTS = [10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000]
SL_CAPS        = [10_000_000, 20_000_000, 30_000_000]


def run_scenario(freq, sev, treaty, treaty_limit, coc):
    """Run one simulation and return a result dict."""
    engine  = MonteCarloEngine(freq, sev, treaty,
                               n_simulations=N_SIMS, random_state=SEED)
    results = engine.run()
    rm      = compute_risk_measures(results, treaty_limit=treaty_limit)
    pricer  = TechnicalPricer(
        expected_ceded_loss=results.expected_ceded_loss,
        tvar_99=results.tvar_99,
        expense_load=EXPENSE_LOAD,
        profit_load=PROFIT_LOAD,
        cost_of_capital=coc,
    )
    pricing = pricer.price(treaty_limit=treaty_limit)

    return {
        "ecl":             round(rm.expected_ceded_loss, 0),
        "std":             round(rm.std_deviation, 0),
        "var_95":          round(rm.var_95, 0),
        "var_99":          round(rm.var_99, 0),
        "var_995":         round(rm.var_995, 0),
        "tvar_99":         round(rm.tvar_99, 0),
        "prob_attachment": round(rm.prob_attachment, 4),
        "prob_exhaustion": round(rm.prob_exhaustion, 4),
        "cv":              round(rm.coefficient_of_variation, 4),
        "skewness":        round(rm.skewness, 4),
        "expense_loading": round(pricing.expense_loading, 0),
        "profit_loading":  round(pricing.profit_loading, 0),
        "capital_load":    round(pricing.capital_load, 0),
        "premium":         round(pricing.technical_premium, 0),
        "rol":             round(pricing.rate_on_line, 6),
        "distribution":    build_histogram(results),
    }


def main():
    scenarios = {"xl": [], "sl": [], "meta": {
        "mu": MU,
        "expense_load": EXPENSE_LOAD,
        "profit_load": PROFIT_LOAD,
        "n_sims": N_SIMS,
        "hist_bins": HIST_BINS,
        "xl_retentions": XL_RETENTIONS,
        "xl_limits": XL_LIMITS,
        "sl_attachments": SL_ATTACHMENTS,
        "sl_caps": SL_CAPS,
        "sigmas": SIGMAS,
        "lambdas": LAMBDAS,
        "cocs": COCS,
    }}

    # ─────────────────────────────────────────────
    # XL scenarios
    # ─────────────────────────────────────────────
    xl_combos = list(itertools.product(
        XL_RETENTIONS, XL_LIMITS, SIGMAS, LAMBDAS, COCS
    ))
    total_xl = len(xl_combos)
    print(f"Running {total_xl} XL scenarios...")
    t0 = time.time()

    for i, (ret, lim, sigma, lam, coc) in enumerate(xl_combos):

        # skip invalid: retention >= limit
        if ret >= lim:
            continue

        freq   = PoissonFrequency(lambda_=lam)
        sev    = LognormalSeverity(mu=MU, sigma=sigma)
        treaty = ExcessOfLoss(retention=ret, limit=lim)

        result = run_scenario(freq, sev, treaty, lim, coc)
        result.update({
            "retention": ret,
            "limit":     lim,
            "sigma":     sigma,
            "lambda":    lam,
            "coc":       coc,
        })
        scenarios["xl"].append(result)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            remaining = (total_xl - i - 1) / rate
            print(f"  XL: {i+1}/{total_xl} "
                  f"({(i+1)/total_xl:.0%}) — "
                  f"~{remaining:.0f}s remaining")

    print(f"XL done — {len(scenarios['xl'])} scenarios in "
          f"{time.time()-t0:.0f}s")

    # ─────────────────────────────────────────────
    # Stop-Loss scenarios
    # ─────────────────────────────────────────────
    sl_combos = list(itertools.product(
        SL_ATTACHMENTS, SL_CAPS, SIGMAS, LAMBDAS, COCS
    ))
    total_sl = len(sl_combos)
    print(f"\nRunning {total_sl} Stop-Loss scenarios...")
    t1 = time.time()

    for i, (att, cap, sigma, lam, coc) in enumerate(sl_combos):

        freq   = PoissonFrequency(lambda_=lam)
        sev    = LognormalSeverity(mu=MU, sigma=sigma)
        treaty = StopLoss(attachment=att, cap=cap)

        result = run_scenario(freq, sev, treaty, cap, coc)
        result.update({
            "attachment": att,
            "cap":        cap,
            "sigma":      sigma,
            "lambda":     lam,
            "coc":        coc,
        })
        scenarios["sl"].append(result)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t1
            rate    = (i + 1) / elapsed
            remaining = (total_sl - i - 1) / rate
            print(f"  SL: {i+1}/{total_sl} "
                  f"({(i+1)/total_sl:.0%}) — "
                  f"~{remaining:.0f}s remaining")

    print(f"SL done — {len(scenarios['sl'])} scenarios in "
          f"{time.time()-t1:.0f}s")

    # ─────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────
    output_path = "app/scenarios.json"
    print(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(scenarios, f, separators=(",", ":"))

    size_mb = len(json.dumps(scenarios)) / 1e6
    print(f"Done — {len(scenarios['xl'])} XL + {len(scenarios['sl'])} SL "
          f"scenarios saved ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()