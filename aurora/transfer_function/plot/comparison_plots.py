from matplotlib import pyplot as plt

from aurora.sandbox.io_helpers.zfile_murphy import read_z_file
from aurora.transfer_function.plot.rho_phi_helpers import plot_phi
from aurora.transfer_function.plot.rho_phi_helpers import plot_rho


def compare_two_z_files(
    z_path1,
    z_path2,
    angle1 = 0.0,
    angle2 = 0.0,
    label1="",
    label2="",
    scale_factor1=1.0,
    scale_factor2=1.0,
    out_file="",
    show_plot=True,
    use_ylims=True,
    use_xlims=True,
    **kwargs,
):
    """
    TODO: Put this in plot_helpers

    Parameters
    ----------
    z_path1: str or pathlib.Path
    z_path2: str or pathlib.Path
    angle1: float
    angle2: float
    label1: str
    label2: str
    scale_factor1
    scale_factor2
    out_file
    show_plot
    use_ylims
    use_xlims

    kwargs
    rho_ylims
    xlims


    Returns
    -------

    """
    zfile1 = read_z_file(z_path1, angle=angle1)
    zfile2 = read_z_file(z_path2, angle=angle2)
    print(f"scale_factor1: {scale_factor1}")
    fig, axs = plt.subplots(nrows=2, dpi=300, sharex=True)  # figsize=(8, 6.),
    markersize = kwargs.get("markersize", 3)

    plot_rho(
        axs[0],
        zfile1.periods,
        zfile1.rxy * scale_factor1,
        label=f"{label1} rxy",
        markersize=markersize,
        marker="^",
        color="red",
    )
    plot_rho(
        axs[0],
        zfile2.periods,
        zfile2.rxy * scale_factor2,
        label=f"{label2} rxy",
        markersize=markersize,
        marker="^",
        color="black",
    )
    plot_rho(
        axs[0],
        zfile1.periods,
        zfile1.ryx * scale_factor1,
        label=f"{label1} ryx",
        markersize=markersize,
        color="blue",
    )
    plot_rho(
        axs[0],
        zfile2.periods,
        zfile2.ryx,
        label=f"{label2} ryx",
        markersize=markersize,
        color="black",
    )
    axs[0].legend(prop={"size": 6})
    # axs[0].set_ylabel("$\rho_a$")
    axs[0].set_ylabel("Apparent Resistivity $\Omega$-m")
    rho_ylims = kwargs.get("rho_ylims", [1, 1e3])
    if use_ylims:
        axs[0].set_ylim(rho_ylims[0], rho_ylims[1])
    xlims = kwargs.get("xlims", [1e-3, 1e3])
    if use_xlims:
        axs[0].set_xlim(xlims[0], xlims[1])

    plot_phi(
        axs[1],
        zfile1.periods,
        zfile1.pxy,
        label=f"{label1} pxy",
        markersize=markersize,
        marker="^",
        color="red",
    )
    plot_phi(
        axs[1],
        zfile2.periods,
        zfile2.pxy,
        label=f"{label2} pxy",
        markersize=markersize,
        marker="^",
        color="black",
    )
    plot_phi(
        axs[1],
        zfile1.periods,
        zfile1.pyx,
        label=f"{label1} pyx",
        markersize=markersize,
        color="blue",
    )
    plot_phi(
        axs[1],
        zfile2.periods,
        zfile2.pyx,
        label=f"{label1} pyx",
        markersize=markersize,
        color="black",
    )
    axs[1].set_xlabel("Period (s)")
    axs[1].set_ylabel("Phase (degrees)")
    phi_ylims = kwargs.get("phi_ylims", [0, 90])
    axs[1].set_ylim(phi_ylims[0], phi_ylims[1])
    if out_file:
        # if out_file[-3:] != ".png":
        #     out_file+=".png"
        plt.savefig(f"{out_file}")
    if show_plot:
        plt.show()
