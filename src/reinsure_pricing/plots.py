import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    Histogram of simulated ceded losses with VaR and TVaR annotations.
    Zeros (years with no ceded loss) are excluded from the plot
    but noted in the subtitle.
    """

    losses = results.ceded_losses
    nonzero = losses[losses > 0]
    pct_zero = (losses == 0).mean()

    fig, ax = plt.subplots(figsize=figsize)

    # --- Histogram ---
    ax.hist(nonzero, bins=bins, color="#4C72B0", alpha=0.75,
            edgecolor="white", linewidth=0.4, density=True,
            label="Simulated ceded losses")

    # --- VaR lines ---
    for alpha, label, color in [
        (0.95,  "VaR 95%",   "#E8A838"),
        (0.99,  "VaR 99%",   "#D94F3D"),
        (0.995, "VaR 99.5%", "#8B1A1A"),
    ]:
        val = results.var(alpha)
        ax.axvline(val, color=color, linewidth=1.8, linestyle="--",
                   label=f"{label}: {val:,.0f}")

    # --- TVaR 99% line ---
    tvar = risk_measures.tvar_99
    ax.axvline(tvar, color="#6A0DAD", linewidth=1.8, linestyle=":",
               label=f"TVaR 99%: {tvar:,.0f}")

    # --- Treaty limit line ---
    ax.axvline(treaty_limit, color="#2CA02C", linewidth=1.5, linestyle="-.",
               label=f"Treaty limit: {treaty_limit:,.0f}")

    # --- Labels and formatting ---
    ax.set_xlabel("Ceded Loss (€)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"{title}\n"
        f"ECL: {risk_measures.expected_ceded_loss:,.0f}  |  "
        f"n = {len(losses):,}  |  "
        f"Years with no ceded loss: {pct_zero:.1%}",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
    )
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
    Line chart showing how technical premium and ECL vary
    as a single input parameter changes.
    """

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(parameter_values, premiums, marker="o", color="#D94F3D",
            linewidth=2, label="Technical Premium")
    ax.plot(parameter_values, ecls, marker="s", color="#4C72B0",
            linewidth=2, linestyle="--", label="Expected Ceded Loss")

    ax.set_xlabel(parameter_name, fontsize=12)
    ax.set_ylabel("Amount (€)", fontsize=12)
    ax.set_title(f"Sensitivity: {parameter_name}", fontsize=13)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M")
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig