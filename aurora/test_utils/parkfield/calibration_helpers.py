import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from scipy.signal import medfilt

plt.ion()


def load_bf4_fap_for_parkfield_test_using_mt_metadata(frequencies):
    """
    The hardware repsonses (AAF and digitizer) are not included in this response,
    but these do not make any significant difference away from the Nyquist frequecny.

    Near the Nyquist calibration is inadequate anyhow.  Looking at the output plots,
    which show the "full calibration" vs "response table (EMI)", neither one is
    realistic at high frequency.  The fap ("response table (EMI)") curve does not
    compensate for AAF and plunges down at high frequency.  The full calibration
    from the PZ response on the other hand rises unrealistically.  The PZ rising
    signal amplitude at high frequency is an artefact of calibrating noise.

    Parameters
    ----------
    frequencies : numpy array
        Array of frequencies at which to evaluate the bf response function
    Returns
    -------

    """
    from aurora.time_series.filters.filter_helpers import (
        make_frequency_response_table_filter,
    )

    bf4_obj = make_frequency_response_table_filter(case="bf4")
    bf4_resp = bf4_obj.complex_response(frequencies)
    bf4_resp *= 421721.0  # counts-per-volt compensation for PKD
    return bf4_resp


def plot_responses(
    key,
    frequencies,
    pz_calibration_response,
    bf4_resp,
    figures_path,
    show_response_curves,
):
    """

    Parameters
    ----------
    key : str
        The channel name, "hx", "hy", "ex", "ey", "hz"
    frequencies : numpy array
        The frequencies at which the response will be plotted
    pz_calibration_response : numpy.ndarray
        The complex-values resposne function from the pole-zero response
    bf4_resp : None or numpy.ndarray
        The complex-values resposne function from the BF-4 coil only.
    figures_path : str or Path
        Where the figures will be saved
    show_response_curves : bool
        If True, plots flash to screen - for debugging

    Returns
    -------

    """

    if key[0].lower() == "h":
        response_units = "counts/nT"
    else:
        response_units = "counts/mV/km"

    plt.figure(1)
    plt.clf()
    plt.loglog(frequencies, np.abs(pz_calibration_response), label="pole-zero")
    if bf4_resp is not None:
        plt.loglog(frequencies, np.abs(bf4_resp), label="fap EMI")
    plt.legend()
    plt.title(f"Calibration Response Functions {key}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(f"Response {response_units}")
    png_name = f"{key}_response_function.png"
    plt.savefig(figures_path.joinpath(png_name))
    if show_response_curves:
        plt.show()


def parkfield_sanity_check(
    fft_obj,
    run_obj,
    show_response_curves=False,
    show_spectra=True,
    figures_path=Path(""),
    include_decimation=True,
):
    """
    loop over channels in fft obj and make calibrated spectral plots

    Parameters
    ----------
    fft_obj : xarray.core.dataset.Dataset
        The FFT of the data.  This is actually an STFT but with only one time window.

    Returns
    -------

    """

    frequencies = fft_obj.frequency.data[1:]  # drop DC, add flag for dropping DC
    figures_path.mkdir(parents=True, exist_ok=True)
    channel_keys = list(fft_obj.data_vars.keys())
    print(f"channel_keys: {channel_keys}")

    for key in channel_keys:
        print(f"calibrating channel {key}")
        if key[0].lower() == "h":
            bf4 = True
            spectral_units = "nT/sqrt(Hz)"
        else:
            bf4 = False
            spectral_units = "mv/km/sqrt(Hz)"
        channel = run_obj.get_channel(key)

        # pole-zero calibration response
        pz_calibration_response = channel.channel_response_filter.complex_response(
            frequencies, include_decimation=include_decimation
        )

        if channel.channel_response_filter.units_in.lower() in ["t", "tesla"]:
            print("WARNING: Expecting nT but got T")

        # Frequency response table response
        bf4_resp = None
        if bf4:
            bf4_resp = load_bf4_fap_for_parkfield_test_using_mt_metadata(frequencies)
        # compare responses graphically
        plot_responses(
            key,
            frequencies,
            pz_calibration_response,
            bf4_resp,
            figures_path,
            show_response_curves,
        )

        # Add assert test issue #156 here:
        if bf4_resp is not None:
            response_ratio = np.abs(pz_calibration_response) / np.abs(bf4_resp)
            if np.median(response_ratio) > 1.1:
                print("ERROR in response calculation")
                print("See issue #156")
                raise Exception
        # create smoothed amplitude spectra
        n_smooth = 131  # use 1 for no smoothing
        show_raw = 0
        raw_spectral_density = fft_obj[key].data[:, 1:]
        raw_spectral_density = raw_spectral_density.squeeze()  # only 1 FFT window
        calibrated_data_pz = raw_spectral_density / pz_calibration_response
        smooth_calibrated_data_pz = medfilt(np.abs(calibrated_data_pz), n_smooth)
        if bf4:
            calibrated_data_fap = raw_spectral_density / np.abs(bf4_resp)
            smooth_calibrated_data_fap = medfilt(np.abs(calibrated_data_fap), n_smooth)

        if bf4 & (key == "hx"):
            schumann_cond = (frequencies > 7.6) & (frequencies < 8.0)
            schumann_amplitude_fap = np.median(
                smooth_calibrated_data_fap[schumann_cond]
            )
            schumann_amplitude_pz = np.median(smooth_calibrated_data_pz[schumann_cond])
            schumann_ratio = schumann_amplitude_pz / schumann_amplitude_fap
            if (schumann_ratio > 10) | (schumann_ratio < 0.1):
                print("ERROR in response calculation")
                print("See issue #156")
                print("Amplitude of field around Schumann band incorrect")
                raise Exception

        # Do Plotting (can factor this out)
        plt.figure(2)
        plt.clf()
        bf4_colour = "red"
        pz_color = "blue"

        if show_raw:
            plt.loglog(
                frequencies, calibrated_data_pz, color=pz_color, label="pole-zero"
            )
            if bf4:
                plt.loglog(
                    frequencies,
                    calibrated_data_fap,
                    color=bf4_colour,
                    label="response table (EMI)",
                )
        if n_smooth:
            plt.loglog(
                frequencies,
                smooth_calibrated_data_pz,
                color=pz_color,
                label="smooth pole-zero",
            )
            if bf4:
                plt.loglog(
                    frequencies,
                    smooth_calibrated_data_fap,
                    color=bf4_colour,
                    label="response table (EMI)",
                )
        plt.legend()
        plt.grid(True, which="both", ls="-")
        plt.title(f"Calibrated Spectra {key}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(f"{spectral_units}")
        png_name = f"{key}_calibrated_spectra.png"
        plt.savefig(figures_path.joinpath(png_name))
        if show_spectra:
            plt.show()
