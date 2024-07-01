"""
helper functions being factored out of plotter to support mulitple plots on a single
axis.
"""


def plot_rho(
    ax,
    periods,
    rho,
    marker="o",
    color="k",
    linestyle="None",
    label="",
    markersize=10,
    ax_label_size=16,
):
    """

    Parameters
    ----------
    ax
    periods
    rho
    marker
    color
    linestyle
    label
    markersize
    ax_label_size

    Returns
    -------

    """
    ax.loglog(
        periods,
        rho,
        marker=marker,
        color=color,
        linestyle=linestyle,
        label=label,
        markersize=markersize,
    )
    ax.tick_params(axis="both", which="major", labelsize=ax_label_size)
    ax.tick_params(axis="x", which="minor", bottom=True)
    return


def plot_phi(
    ax,
    periods,
    phi,
    marker="o",
    color="k",
    linestyle="None",
    label="",
    markersize=10,
    ax_label_size=16,
):
    ax.semilogx(
        periods,
        phi,
        marker=marker,
        color=color,
        linestyle=linestyle,
        label=label,
        markersize=markersize,
    )
    ax.tick_params(axis="both", which="major", labelsize=ax_label_size)
    ax.minorticks_on()  # (axis="x", which="minor", bottom=True)
    return
