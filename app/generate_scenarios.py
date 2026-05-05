"""
generate_scenarios.py
---------------------
Pre-computes all pricing scenarios and saves them to app/scenarios.json.
The JSON is loaded by dashboard.html for instant client-side rendering.

Phase 4 additions:
    - AAL and AAD scenarios for XL treaties
    - Reinstatement premium impact scenarios

Run with:
    python app/generate_scenarios.py

This will take approximately 30-35 minutes depending on your machine.
Progress is printed to the console.
"""

import json
import time
import itertools
import numpy as np

from reinsure_pricing.frequency import PoissonFrequency
from reinsure_pricing.severity import LognormalSeverity
from reinsure_pricing.treaties import ExcessOfLoss, StopLoss, ReinstatementProvision
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
# XL parameter grid
# ─────────────────────────────────────────────
XL_RETENTIONS    = [500_000, 750_000, 1_000_000, 1_500_000, 2_000_000]
XL_LIMITS        = [2_000_000, 3_000_000, 5_000_000, 7_000_000, 10_000_000]
SIGMAS           = [0.8, 1.0, 1.2, 1.4, 1.6]
LAMBDAS          = [60, 90, 120, 150, 180]
COCS             = [0.05, 0.15]

# AAL as a multiplier of the per-occurrence limit.
# None = no AAL, 2 = 2x limit, 4 = 4x limit
XL_AAL_MULTIPLES = [None, 2, 4]

# AAD expressed as absolute values in euros.
# None = no AAD active for that scenario.
XL_AADS          = [None, 200_000, 500_000]

# ─────────────────────────────────────────────
# Stop-Loss parameter grid
# ─────────────────────────────────────────────
SL_ATTACHMENTS = [10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000]
SL_CAPS        = [10_000_000, 20_000_000, 30_000_000]


def run_scenario(freq, sev, treaty, treaty_limit, coc,
                 reinstatement=None):
    """Run one simulation and return a result dict."""
    engine  = MonteCarloEngine(freq, sev, treaty,
                               n_simulations=N_SIMS, random_state=SEED)
    results = engine.run(reinstatement=reinstatement)
    rm      = compute_risk_measures(results, treaty_limit=treaty_limit)
    pricer  = TechnicalPricer(
        expected_ceded_loss=results.expected_ceded_loss,
        tvar_99=results.tvar_99,
        expense_load=EXPENSE_LOAD,
        profit_load=PROFIT_LOAD,
        cost_of_capital=coc,
        expected_reinstatement_premium=results.expected_reinstatement_premium,
    )
    pricing = pricer.price(treaty_limit=treaty_limit)

    return {
        "ecl":                    round(rm.expected_ceded_loss, 0),
        "std":                    round(rm.std_deviation, 0),
        "var_95":                 round(rm.var_95, 0),
        "var_99":                 round(rm.var_99, 0),
        "var_995":                round(rm.var_995, 0),
        "tvar_99":                round(rm.tvar_99, 0),
        "prob_attachment":        round(rm.prob_attachment, 4),
        "prob_exhaustion":        round(rm.prob_exhaustion, 4),
        "cv":                     round(rm.coefficient_of_variation, 4),
        "skewness":               round(rm.skewness, 4),
        "exp_reinst_premium":     round(results.expected_reinstatement_premium, 0),
        "net_ecl":                round(pricing.net_expected_ceded_loss, 0),
        "expense_loading":        round(pricing.expense_loading, 0),
        "profit_loading":         round(pricing.profit_loading, 0),
        "capital_load":           round(pricing.capital_load, 0),
        "premium":                round(pricing.technical_premium, 0),
        "rol":                    round(pricing.rate_on_line, 6),
    }


def main():
    scenarios = {
        "xl":   [],
        "sl":   [],
        "meta": {
            "mu":               MU,
            "expense_load":     EXPENSE_LOAD,
            "profit_load":      PROFIT_LOAD,
            "n_sims":           N_SIMS,
            "xl_retentions":    XL_RETENTIONS,
            "xl_limits":        XL_LIMITS,
            "xl_aal_multiples": XL_AAL_MULTIPLES,
            "xl_aads":          XL_AADS,
            "sl_attachments":   SL_ATTACHMENTS,
            "sl_caps":          SL_CAPS,
            "sigmas":           SIGMAS,
            "lambdas":          LAMBDAS,
            "cocs":             COCS,
        }
    }

    # ─────────────────────────────────────────────
    # XL scenarios
    # ─────────────────────────────────────────────
    xl_combos = list(itertools.product(
        XL_RETENTIONS, XL_LIMITS, SIGMAS, LAMBDAS, COCS,
        XL_AAL_MULTIPLES, XL_AADS
    ))
    total_xl = len(xl_combos)
    print(f"Running {total_xl} XL combinations (invalid ones skipped)...")
    t0 = time.time()

    for i, (ret, lim, sigma, lam, coc, aal_mult, aad) in enumerate(xl_combos):

        # skip invalid combinations
        if ret >= lim:
            continue
        if aad is not None and aad >= lim:
            continue

        # compute AAL from multiplier
        aal = aal_mult * lim if aal_mult is not None else None

        freq   = PoissonFrequency(lambda_=lam)
        sev    = LognormalSeverity(mu=MU, sigma=sigma)
        treaty = ExcessOfLoss(
            retention=ret,
            limit=lim,
            aggregate_limit=aal,
            aggregate_deductible=aad,
        )

        # Reinstatement provision — 1 free + 1 paid
        # original_premium approximated as 10% of limit
        rp = ReinstatementProvision(
            n_free=1,
            n_paid=1,
            original_premium=lim * 0.10,
        )

        result = run_scenario(freq, sev, treaty, lim, coc,
                              reinstatement=rp)
        result.update({
            "retention":  ret,
            "limit":      lim,
            "sigma":      sigma,
            "lambda":     lam,
            "coc":        coc,
            "aal":        aal if aal is not None else 0,
            "aal_mult":   aal_mult if aal_mult is not None else 0,
            "aad":        aad if aad is not None else 0,
        })
        scenarios["xl"].append(result)

        if (i + 1) % 500 == 0:
            elapsed   = time.time() - t0
            rate      = (i + 1) / elapsed
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
            elapsed   = time.time() - t1
            rate      = (i + 1) / elapsed
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
