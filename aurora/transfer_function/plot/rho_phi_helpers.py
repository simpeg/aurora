"""
helper functions being factored out of plotter to support mulitple plots on a single
axis.
"""


def plot_rho(
    ax, periods, rho, marker="o", color="k", linestyle="None", label="", markersize=10
):
    ax.loglog(
        periods,
        rho,
        marker=marker,
        color=color,
        linestyle=linestyle,
        label=label,
        markersize=markersize,
    )
    ax.tick_params(axis="both", which="major", labelsize=16)
    return
