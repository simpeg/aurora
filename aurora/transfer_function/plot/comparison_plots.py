"""
    This module contains a function to for comparing legacy "z-file"
     transfer function files.

"""
import pathlib

from aurora.sandbox.io_helpers.zfile_murphy import read_z_file
from aurora.transfer_function.plot.rho_phi_helpers import plot_phi
from aurora.transfer_function.plot.rho_phi_helpers import plot_rho
from loguru import logger
from matplotlib import pyplot as plt
from typing import Optional, Union


def compare_two_z_files(
    z_path1: Union[pathlib.Path, str],
    z_path2: Union[pathlib.Path, str],
    angle1: Optional[float] = 0.0,
    angle2: Optional[float] = 0.0,
    label1: Optional[str] = "",
    label2: Optional[str] = "",
    scale_factor1: Optional[float] = 1.0,
    scale_factor2: Optional[float] = 1.0,
    out_file: Optional[Union[pathlib.Path, str]] = "",
    show_plot: Optional[bool] = True,
    use_ylims: Optional[bool] = True,
    use_xlims: Optional[bool] = True,
    rho_ax_label_size: Optional[float] = 16,
    phi_ax_label_size: Optional[float] = 16,
    markersize: Optional[float] = 3,
    rho_ylims: Optional[tuple] = (1, 1e3),
    phi_ylims: Optional[tuple] = (0, 90),
    xlims: Optional[tuple] = (1e-3, 1e3),
    title_string: Optional[str] = "",
    subtitle_string: Optional[str] = "",
):
    """
    Takes as input two z-files and plots them both on the same axis

    TODO: Replace with a method from MTpy

    Parameters
    ----------
    z_path1: Union[pathlib.Path, str]
        The first z-file to compare
    z_path2: Union[pathlib.Path, str]
        The second z-file to compare
    angle1: Optional[float] = 0.0
        The angle to rotate the first TF
    angle2: Optional[float] = 0.0
        The angle to rotate the second TF
    label1: Optional[str] = "",
        A legend label for the first TF
    label2: Optional[str] = "",
        A legend label for the second TF
    scale_factor1: Optional[float] = 1.0
        A scale factor to shift rho of TF1
    scale_factor2: Optional[float] =1.0
        A scale factor to shift rho of TF2
    out_file: Optional[Union[pathlib.Path, str]] = ""
        A file to save the plot
    show_plot: Optional[bool] = True
        If True, show an interactive plot
    use_ylims: Optional[bool] = True
        If True, explicitly set y-axis limits to rho_ylims
    use_xlims: Optional[bool] = True
        If True, explicitly set x-axis limits to xlims
    rho_ax_label_size: Optional[float] = 16
        Set the y-axis label size for rho
    phi_ax_label_size: Optional[float] = 16,
        Set the y-axis label size for phi
    markersize: Optional[float] = 3
        Set the markersize (for both rho and phi)
    rho_ylims: Optional[tuple] = (1, 1e3)
        The Y-axis limits to apply on rho (if use_ylims is True)
    phi_ylims: Optional[tuple] = (0, 90),
        The Y-axis limits to apply on phi
    xlims: Optional[tuple] = (1e-3, 1e3)
        The Z-axis limits to apply (if use_xlims is True)

    """
    zfile1 = read_z_file(z_path1, angle=angle1)
    zfile2 = read_z_file(z_path2, angle=angle2)

    logger.info(f"Sacling TF scale_factor1: {scale_factor1}")
    fig, axs = plt.subplots(nrows=2, dpi=300, sharex=True)  # figsize=(8, 6.),

    # Make LaTeX symbol strings
    rho_phi_strings = {}
    rho_phi_strings["rho"] = {}
    rho_phi_strings["phi"] = {}
    for xy_or_yx in ["xy", "yx"]:
        rho_phi_strings["rho"][xy_or_yx] = f"$\\rho_{{{xy_or_yx}}}$"
        rho_phi_strings["phi"][xy_or_yx] = f"$\phi_{{{xy_or_yx}}}$"

    markers = {}
    markers["xy"] = "^"
    markers["yx"] = "o"
    file1_colors = {}
    file2_colors = {}
    file1_colors["xy"] = "black"
    file1_colors["yx"] = "black"
    file2_colors["xy"] = "red"
    file2_colors["yx"] = "blue"

    rho_or_phi = "rho"
    for xy_or_yx in ["xy", "yx"]:
        plot_rho(
            axs[0],
            zfile1.periods,
            zfile1.rho(xy_or_yx) * scale_factor1,
            label=f"{label1} {rho_phi_strings[rho_or_phi][xy_or_yx]}",
            markersize=markersize,
            marker=markers[xy_or_yx],
            color=file1_colors[xy_or_yx],
            ax_label_size=rho_ax_label_size,
        )
        plot_rho(
            axs[0],
            zfile2.periods,
            zfile2.rho(xy_or_yx) * scale_factor2,
            label=f"{label2} {rho_phi_strings[rho_or_phi][xy_or_yx]}",
            markersize=markersize,
            marker=markers[xy_or_yx],
            color=file2_colors[xy_or_yx],
            ax_label_size=rho_ax_label_size,
        )

    axs[0].legend(prop={"size": 6})
    # axs[0].set_ylabel("$\\rho_a$")
    axs[0].set_ylabel("Apparent Resistivity $\Omega$-m", fontsize=12)
    if use_ylims:
        axs[0].set_ylim(rho_ylims[0], rho_ylims[1])
    if use_xlims:
        axs[0].set_xlim(xlims[0], xlims[1])

    rho_or_phi = "phi"
    for xy_or_yx in ["xy", "yx"]:
        plot_phi(
            axs[1],
            zfile1.periods,
            zfile1.phi(xy_or_yx) * scale_factor1,
            label=f"{label1} {rho_phi_strings[rho_or_phi][xy_or_yx]}",
            markersize=markersize,
            marker=markers[xy_or_yx],
            color=file1_colors[xy_or_yx],
            ax_label_size=phi_ax_label_size,
        )
        plot_phi(
            axs[1],
            zfile2.periods,
            zfile2.phi(xy_or_yx) * scale_factor2,
            label=f"{label2} {rho_phi_strings[rho_or_phi][xy_or_yx]}",
            markersize=markersize,
            marker=markers[xy_or_yx],
            color=file2_colors[xy_or_yx],
            ax_label_size=phi_ax_label_size,
        )

    axs[1].legend(prop={"size": 6})
    axs[1].set_xlabel("Period (s)", fontsize=12)
    axs[1].set_ylabel("Phase (degrees)", fontsize=12)
    axs[1].set_ylim(phi_ylims[0], phi_ylims[1])

    axs[0].grid(
        which="both",
        axis="both",
    )
    axs[1].grid(
        which="both",
        axis="both",
    )
    if title_string:
        plt.suptitle(title_string, fontsize=15)
    if subtitle_string:
        axs[0].set_title(subtitle_string, fontsize=8)
    if out_file:
        plt.savefig(f"{out_file}")

    if show_plot:
        plt.show()
