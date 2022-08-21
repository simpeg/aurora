import matplotlib.pyplot as plt
import numpy as np


def plot_complex_response(frequency, complex_response, **kwargs):
    """
    ToDo: add methods for suporting instrument object but for now take as kwargs
    :param frequency: numpy array of frequencies at which complex response is defined
    :param complex_response: numpy array of full complex valued response function
    :param kwargs:
    :return:
    """

    show = kwargs.get("show", True)
    linewidth = kwargs.get("linewidth", 3)

    make = kwargs.get("make", None)
    model = kwargs.get("model", None)
    y_amp_string = kwargs.get("yamp", None)

    amplitude = np.abs(complex_response)
    phase = np.angle(complex_response)

    plt.figure(1)
    # plt.clf()
    ax1 = plt.subplot(2, 1, 1)
    ax1.loglog(frequency, amplitude, linewidth=linewidth)
    ax1.set_title("{}-{}    Amplitude Response".format(make, model))
    ax1.grid(True, which="both", ls="-")
    ax1.set_ylabel("{}".format(y_amp_string))
    y_lim = ax1.get_ylim()
    ax1.set_ylim((y_lim[0], 1.1 * y_lim[1]))
    ax2 = plt.subplot(2, 1, 2)
    ax2.semilogx(frequency, phase, linewidth=linewidth)
    ax2.set_title("{}-{}    Phase Response".format(make, model))
    ax2.grid(True, which="both", ls="-")
    if show:
        plt.show()


def plot_tf_obj(tf_obj, out_filename=None, show=True):
    """
    To Do: Get plotter from MTpy or elsewhere.
    See Issue #209
    https://github.com/simpeg/aurora/issues/209
    Parameters
    ----------
    tf_obj: aurora.transfer_function.TTFZ.TTFZ
        The transfer function values packed into an object
    out_filename: string
        Where to save the file.  No png is saved if this is False

    """
    from aurora.transfer_function.plot.rho_plot import RhoPlot
    import matplotlib.pyplot as plt

    plotter = RhoPlot(tf_obj)
    fig, axs = plt.subplots(nrows=2)
    ttl_str = tf_obj.tf_header.local_station_id
    plotter.rho_sub_plot(axs[0], ttl_str=ttl_str)
    plotter.phase_sub_plot(axs[1], ttl_str=ttl_str)
    if out_filename:
        plt.savefig(f"{out_filename}.png")
    if show:
        plt.show()
