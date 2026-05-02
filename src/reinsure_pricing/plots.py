"""
plots.py
--------
Visualisation functions for reinsurance layer pricing analysis.

Provides two core plots:

    1. plot_ceded_loss_distribution()
       Histogram of simulated annual ceded losses with VaR, TVaR,
       and treaty limit annotated as vertical lines. Years with zero
       ceded loss are excluded from the histogram but noted in the
       subtitle — they would otherwise dominate the x-axis and make
       the tail invisible.

    2. plot_sensitivity()
       Line chart showing how the technical premium and ECL respond
       to changes in a single input parameter (e.g. retention, lambda).
       Used to understand which assumptions drive the price most.

All functions return a matplotlib Figure object so the caller can
either display it interactively (plt.show()) or save it to disk
(fig.savefig(...)) without the function making that decision.
"""

import numpy as np
import matplotlib.pyplot as plt
from reinsure_pricing.simulation import SimulationResults
from reinsure_pricing.risk_measures import RiskMeasures


def plot_ceded_loss_distribution(
        results: SimulationResults,
        risk_measures: RiskMeasures,
        treaty_limit: float,
        title: str = "Ceded Loss Distribution",
        figsize: tuple = (12, 6),
        bins: int = 100,
        save_path: str = None,
) -> plt.Figure:
    """
    Histogram of simulated ceded losses with risk measure annotations.

    Years with zero ceded loss (layer not triggered) are excluded from
    the histogram body but their proportion is noted in the subtitle.
    Including zeros would compress the visible distribution and make
    the tail hard to read.

    Vertical lines mark the key risk thresholds:
        - VaR 95%   : orange dashed
        - VaR 99%   : red dashed
        - VaR 99.5% : dark red dashed
        - TVaR 99%  : purple dotted
        - Treaty limit : green dash-dot

    Parameters
    ----------
    results : SimulationResults
        Output from MonteCarloEngine.run(). Provides the raw ceded
        loss array for the histogram.
    risk_measures : RiskMeasures
        Output from compute_risk_measures(). Provides the VaR and
        TVaR values for the vertical line annotations.
    treaty_limit : float
        Maximum ceded loss under the treaty — drawn as a green line
        to show where the layer exhausts.
    title : str
        Plot title. Default is 'Ceded Loss Distribution'.
    figsize : tuple
        Figure dimensions as (width, height) in inches. Default (12, 6).
    bins : int
        Number of histogram bins. Default 100. Increase for smoother
        appearance with very large simulation counts.
    save_path : str, optional
        If provided, saves the figure to this path at 150 dpi.
        Example: 'outputs/ceded_loss_hist.png'

    Returns
    -------
    plt.Figure
        The matplotlib Figure object. Call plt.show() to display
        or fig.savefig(...) to save after calling this function.
    """

    losses  = results.ceded_losses
    nonzero = losses[losses > 0]       # exclude zero years from histogram
    pct_zero = (losses == 0).mean()    # fraction of years with no cession

    fig, ax = plt.subplots(figsize=figsize)

    # --- Histogram of non-zero ceded losses ---
    ax.hist(
        nonzero,
        bins=bins,
        color="#4C72B0",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
        density=True,               # normalise to density so y-axis is comparable
        label="Simulated ceded losses",
    )

    # --- VaR vertical lines at three confidence levels ---
    for alpha, label, color in [
        (0.95,  "VaR 95%",   "#E8A838"),   # orange
        (0.99,  "VaR 99%",   "#D94F3D"),   # red
        (0.995, "VaR 99.5%", "#8B1A1A"),   # dark red
    ]:
        val = results.var(alpha)
        ax.axvline(
            val,
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"{label}: {val:,.0f}",
        )

    # --- TVaR 99% vertical line ---
    # TVaR sits to the right of VaR 99% — shows average severity in the tail
    tvar = risk_measures.tvar_99
    ax.axvline(
        tvar,
        color="#6A0DAD",            # purple
        linewidth=1.8,
        linestyle=":",
        label=f"TVaR 99%: {tvar:,.0f}",
    )

    # --- Treaty limit vertical line ---
    # Shows where the layer exhausts — losses beyond this are retained by cedant
    ax.axvline(
        treaty_limit,
        color="#2CA02C",            # green
        linewidth=1.5,
        linestyle="-.",
        label=f"Treaty limit: {treaty_limit:,.0f}",
    )

    # --- Axis labels and title ---
    ax.set_xlabel("Ceded Loss (€)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"{title}\n"
        f"ECL: {risk_measures.expected_ceded_loss:,.0f}  |  "
        f"n = {len(losses):,}  |  "
        f"Years with no ceded loss: {pct_zero:.1%}",
        fontsize=12,
    )

    # --- Format x-axis in millions for readability ---
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
    )

    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_sensitivity(
        parameter_values: list,
        premiums: list,
        ecls: list,
        parameter_name: str = "Parameter",
        figsize: tuple = (10, 5),
        save_path: str = None,
) -> plt.Figure:
    """
    Line chart showing premium and ECL sensitivity to a single parameter.

    Plots both the technical premium and ECL on the same axes as the
    chosen parameter varies. The gap between the two lines represents
    the total loading (expense + profit + capital) and how it changes
    with the parameter.

    Typical use cases:
        - Vary retention to show how pricing changes with attachment
        - Vary lambda to show sensitivity to claim frequency assumptions
        - Vary sigma to show sensitivity to severity tail assumptions

    Parameters
    ----------
    parameter_values : list
        X-axis values — the range of the parameter being varied.
        For example, retentions in millions: [0.5, 0.75, 1.0, 1.5, 2.0]
    premiums : list
        Technical premium for each parameter value. Same length as
        parameter_values.
    ecls : list
        Expected Ceded Loss for each parameter value. Same length as
        parameter_values.
    parameter_name : str
        X-axis label and plot title suffix. Default is 'Parameter'.
        Example: 'Retention (M€)'
    figsize : tuple
        Figure dimensions as (width, height) in inches. Default (10, 5).
    save_path : str, optional
        If provided, saves the figure to this path at 150 dpi.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # --- Technical premium line ---
    ax.plot(
        parameter_values,
        premiums,
        marker="o",
        color="#D94F3D",            # red — premium is the key output
        linewidth=2,
        label="Technical Premium",
    )

    # --- ECL line ---
    # Dashed to distinguish from premium — ECL is the floor the premium is built on
    ax.plot(
        parameter_values,
        ecls,
        marker="s",
        color="#4C72B0",            # blue
        linewidth=2,
        linestyle="--",
        label="Expected Ceded Loss",
    )

    # --- Labels and formatting ---
    ax.set_xlabel(parameter_name, fontsize=12)
    ax.set_ylabel("Amount (€)", fontsize=12)
    ax.set_title(f"Sensitivity: {parameter_name}", fontsize=13)

    # Format y-axis in millions for readability
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M")
    )

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)       # light grid for easier reading
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig