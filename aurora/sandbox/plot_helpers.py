"""
    This module contains some plotting helper functions.

    Most of these were used in initial development and should be replaced by methods in MTpy.
    TODO: review which of these can be replaced with methods in MTpy-v2

"""
from matplotlib.gridspec import GridSpec
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ssig


def _is_flat_amplitude(array) -> bool:
    """
    Check of an amplitude response is basically flat.

    If so, it is best to tune the y-axis lims to make numeric noise invisible.

    Parameters
    ----------
    array: the response of some stage as a function of frequency.
    Returns
    -------
    bool:
        True is response is flat, False otherwise
    """
    differences = np.diff(np.abs(array))
    if np.isclose(differences, 0.0).all():
        return True
    else:
        return False


def cast_angular_frequency_to_period_or_hertz(w, units):
    """
        Converts angular frequency to period or Hertz

    Parameters
    ----------
    w: numpy array
        Angular frequencies (radians per second)
    units: str
        Requested output units ("period" or "frequency")

    Returns
    -------
    x_axis: np.ndarray
        Same as input but units are now the requested ones.

    """
    if units.lower() == "period":
        x_axis = (2.0 * np.pi) / w
    elif units.lower() == "frequency":
        x_axis = w / (2.0 * np.pi)
    return x_axis


def plot_complex_response(
    frequency: np.ndarray,
    complex_response: np.ndarray,
    show: Optional[bool] = True,
    linewidth: Optional[float] = 3,
    make: Optional[Union[str, None]] = None,
    model: Optional[Union[str, None]] = None,
    yamp: Optional[Union[str, None]] = None,
):
    """
    Plots amplitude and phase of a complex response as a function of frequency.

    Development Notes:
    ToDo: add methods for supporting instrument object but for now take as kwargs

    :param frequency: numpy array of frequencies at which complex response is defined
    :param complex_response: numpy array of full complex valued response function
    :return:
    """
    y_amp_string = yamp
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


def plot_response_pz(
    w_obs=None,
    resp_obs=None,
    zpk_obs=None,
    zpk_pred=None,
    w_values=None,
    xlim=None,
    title=None,
    x_units="Period",
):
    """
    Plots the pole zero response.

    This function was contributed by Ben Murphy at USGS
    2021-03-17: there are some issues encountered when using this function to plot
    generic resposnes, looks like the x-axis gets out of order when using frequency
    as the axis ...

    Parameters
    ----------
    w_obs : numpy array
        Angular frequencies from lab observations, for example a fap calibration table
    resp_obs: numpy array
        Complex valued response associated with lab observation
    zpk_obs: scipy.signal.ltisys.ZerosPolesGainContinuous
        Pole-Zero object representing the lab observation
    zpk_pred: scipy.signal.ltisys.ZerosPolesGainContinuous
        Pole-Zero object representing the model
    w_values: numpy array
        Angular frequencies at which to evaluate zpk_pred and zpk_obs
    xlim:
    title: string
        Title for the plot
    x_units: string
        One of ["Period" , "frequency"], case insensitve

    """
    # set up the plotting axes
    fig = plt.figure(figsize=(14, 4))
    if title is not None:
        fig.suptitle(title)
    gs = GridSpec(2, 3)
    ax_amp = fig.add_subplot(gs[0, :2])
    ax_phs = fig.add_subplot(gs[1, :2])
    ax_pz = fig.add_subplot(gs[:, 2], aspect="equal")

    # plot observed (lab) response as amplitude and phase
    if w_obs is not None and resp_obs is not None:

        response_amplitude = np.absolute(resp_obs)
        if _is_flat_amplitude(resp_obs):
            response_amplitude[:] = response_amplitude[0]
            ax_amp.set_ylim([0.9 * response_amplitude[0], 1.1 * response_amplitude[0]])
        x_values = cast_angular_frequency_to_period_or_hertz(w_obs, x_units)
        ax_amp.plot(
            x_values,
            response_amplitude,
            color="tab:blue",
            linewidth=1.5,
            linestyle="-",
            label="True",
        )
        ax_phs.plot(
            x_values,
            np.angle(resp_obs, deg=True),
            color="tab:blue",
            linewidth=1.5,
            linestyle="-",
        )
    elif zpk_obs is not None:
        w_obs, resp_obs = ssig.freqresp(zpk_obs, w=w_values)
        x_values = cast_angular_frequency_to_period_or_hertz(w_obs, x_units)
        ax_amp.plot(
            x_values,
            np.absolute(resp_obs),
            color="tab:blue",
            linewidth=1.5,
            linestyle="-",
            label="True",
        )
        ax_phs.plot(
            x_values,
            np.angle(resp_obs, deg=True),
            color="tab:blue",
            linewidth=1.5,
            linestyle="-",
        )
        ax_pz.scatter(
            np.real(zpk_obs.zeros),
            np.imag(zpk_obs.zeros),
            s=75,
            marker="o",
            ec="tab:blue",
            fc="w",
            label="True Zeros",
        )
        ax_pz.scatter(
            np.real(zpk_obs.poles),
            np.imag(zpk_obs.poles),
            s=75,
            marker="x",
            ec="tab:blue",
            fc="tab:blue",
            label="True Poles",
        )

    # plot predicted response (model) as amplitude and phase
    if zpk_pred is not None:
        w_pred, resp_pred = ssig.freqresp(zpk_pred, w=w_values)
        x_values = cast_angular_frequency_to_period_or_hertz(w_values, x_units)

        ax_amp.plot(
            x_values,
            np.absolute(resp_pred),
            color="tab:red",
            linewidth=3,
            linestyle=":",
            label="Fit",
        )
        # print(np.angle(resp_pred, deg=True))
        ax_phs.plot(
            x_values,
            np.angle(resp_pred, deg=True),
            color="tab:red",
            linewidth=3,
            linestyle=":",
        )
        ax_pz.scatter(
            np.real(zpk_pred.zeros),
            np.imag(zpk_pred.zeros),
            s=35,
            marker="o",
            ec="tab:red",
            fc="w",
            label="Fit Zeros",
        )
        ax_pz.scatter(
            np.real(zpk_pred.poles),
            np.imag(zpk_pred.poles),
            s=35,
            marker="x",
            ec="tab:red",
            fc="tab:blue",
            label="Fit Poles",
        )

    if xlim is not None:
        ax_amp.set_xlim(xlim)
        ax_phs.set_xlim(xlim)

    ax_amp.set_xscale("log")
    ax_amp.set_yscale("log")
    ax_amp.set_ylabel("Amplitude Response")
    ax_amp.grid()
    ax_amp.legend()

    ax_phs.set_ylim([-200.0, 200.0])
    ax_phs.set_xscale("log")
    ax_phs.set_ylabel("Phase Response")
    if x_units.lower() == "period":
        x_label = "Period (s)"
    elif x_units.lower() == "frequency":
        x_label = "Frequency (Hz)"
    ax_phs.set_xlabel(x_label)
    ax_phs.grid()

    ax_pz.set_xlabel("Re(z)")
    ax_pz.set_ylabel("Im(z)")
    max_lim = max(
        [
            abs(ax_pz.get_ylim()[0]),
            abs(ax_pz.get_ylim()[1]),
            abs(ax_pz.get_xlim()[0]),
            abs(ax_pz.get_xlim()[0]),
        ]
    )
    ax_pz.set_ylim([-1.25 * max_lim, 1.25 * max_lim])
    ax_pz.set_xlim([-1.25 * max_lim, 1.25 * max_lim])
    ax_pz.grid()
    ax_pz.legend()

    plt.show()


def plot_tf_obj(tf_obj, out_filename=None, show=True):
    """
    Plot the transfer function object in terms of apparent resistivity and phase.

    Development Notes:
        This function is only used in the processing pipeline to give some QC plots

    TODO: Get plotter from MTpy or elsewhere.
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
    ttl_str = tf_obj.tf_header.local_station.id
    plotter.rho_sub_plot(axs[0], ttl_str=ttl_str)
    plotter.phase_sub_plot(axs[1], ttl_str=ttl_str)
    if out_filename:
        plt.savefig(f"{out_filename}.png")
    if show:
        plt.show()
