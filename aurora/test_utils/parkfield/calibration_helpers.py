"""
    This module contains methods that are used in the Parkfield calibration tests.
"""
import matplotlib.pyplot as plt
import mth5.groups.run
import numpy as np
import pathlib

import xarray
from scipy.signal import medfilt
from loguru import logger
from typing import Optional, Union

plt.ion()


def load_bf4_fap_for_parkfield_test_using_mt_metadata(frequencies: np.ndarray):
    """
    Loads a csv format response file for a BF4 coil and return the calibration function.
    Uses an mt_metadata filter object.

    - Anti-alias filter and digitizer responses are not included in the csv -- it is coil only.
    - We ignore the AAF, and hard-code a counts-per-volt value for now

    Development Notes:
    TODO: Add doc showing where counts per volt is accessing in FDSN metadata.

    Parameters
    ----------
    frequencies: np.ndarray
        Frequencies at which to evaluate the bf response function

    Returns
    -------
    bf4_resp:  np.ndarray
        Complex response of the filter at the input frequencies
    """
    from aurora.general_helper_functions import DATA_PATH
    from mt_metadata.timeseries.filters.helper_functions import (
        make_frequency_response_table_filter,
    )

    bf4_file_path = DATA_PATH.joinpath("parkfield", "bf4_9819.csv")
    bf4_obj = make_frequency_response_table_filter(bf4_file_path, case="bf4")
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
    Makes a sanity check plot to show the response of the calibration curves

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
    figures_path : str or pathlib.Path
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
    fft_obj: xarray.Dataset,
    run_obj: mth5.groups.run.RunGroup,
    show_response_curves: Optional[bool] = False,
    show_spectra: Optional[bool] = False,
    figures_path: Optional[Union[str, pathlib.Path]] = pathlib.Path(""),
    include_decimation: Optional[bool] = False,
):
    """
    Loop over channels in fft obj and make calibrated spectral plots

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
    logger.info(f"channel_keys: {channel_keys}")

    for key in channel_keys:
        logger.info(f"calibrating channel {key}")
        if key[0].lower() == "h":
            bf4 = True
            spectral_units = "nT/sqrt(Hz)"
        else:
            bf4 = False
            spectral_units = "mv/km/sqrt(Hz)"
        channel = run_obj.get_channel(key)

        # pole-zero calibration response

        pz_calibration_response = channel.channel_response.complex_response(
            frequencies, include_decimation=include_decimation
        )

        if channel.channel_response.units_in.lower() in ["t", "tesla"]:
            logger.warning("WARNING: Expecting nT but got T")

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
                logger.error("ERROR in response calculation")
                logger.error("See issue #156")
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
                logger.error("ERROR in response calculation")
                logger.error("See issue #156")
                logger.error("Amplitude of field around Schumann band incorrect")
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
