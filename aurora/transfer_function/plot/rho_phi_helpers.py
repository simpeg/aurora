"""
This module contains functions for plotting appararent resistivity and phase.

They are based on the original matlab codes.
They support multiple plots on a single axis.

TODO: replace these with calls to MTpy
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

    Plots apparent resistivity on the given axis

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
    """
    Plots the phase on the given axis.

    Parameters
    ----------
    ax
    periods
    phi
    marker
    color
    linestyle
    label
    markersize
    ax_label_size

    Returns
    -------

    """
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
