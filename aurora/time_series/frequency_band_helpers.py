"""
    This module contains functions that are associated with time series of Fourier coefficients
    TODO: Move these methods to mth5.processing.spectre.frequency_band_helpers
"""
from loguru import logger
from mt_metadata.transfer_functions.processing.aurora import (
    DecimationLevel as AuroraDecimationLevel,
)
from mt_metadata.transfer_functions.processing.aurora import Band
from mth5.timeseries.spectre.spectrogram import extract_band
from typing import Optional, Tuple
import xarray as xr


def get_band_for_tf_estimate(
    band: Band,
    dec_level_config: AuroraDecimationLevel,
    local_stft_obj: xr.Dataset,
    remote_stft_obj: Optional[xr.Dataset],
) -> Tuple[xr.Dataset, xr.Dataset, Optional[xr.Dataset]]:
    """
    Returns spectrograms X, Y, RR for harmonics within the given band

    Parameters
    ----------
    band : mt_metadata.transfer_functions.processing.aurora.Band
        object with lower_bound and upper_bound to tell stft object which
        subarray to return
    config : AuroraDecimationLevel
        information about the input and output channels needed for TF
        estimation problem setup
    local_stft_obj : xarray.core.dataset.Dataset or None
        Time series of Fourier coefficients for the station whose TF is to be
        estimated
    remote_stft_obj : xarray.core.dataset.Dataset or None
        Time series of Fourier coefficients for the remote reference station

    Returns
    -------
    X, Y, RR : xarray.core.dataset.Dataset or None
        data structures as local_stft_object and remote_stft_object, but
        restricted only to input_channels, output_channels,
        reference_channels and also the frequency axes are restricted to
        being within the frequency band given as an input argument.
    """
    logger.info(
        f"Accessing band {band.center_period:.6f}s  ({1./band.center_period:.6f}Hz)"
    )
    band_dataset = extract_band(band, local_stft_obj)
    X = band_dataset[dec_level_config.input_channels]
    Y = band_dataset[dec_level_config.output_channels]
    check_time_axes_synched(X, Y)
    if dec_level_config.reference_channels:
        band_dataset = extract_band(band, remote_stft_obj)
        RR = band_dataset[dec_level_config.reference_channels]
        check_time_axes_synched(Y, RR)
    else:
        RR = None

    return X, Y, RR


def check_time_axes_synched(X, Y):
    """
    Utility function for checking that time axes agree.
    Raises ValueError if axes do not agree.

    It is critical that X, Y, RR have the same time axes for aurora processing.

    Parameters
    ----------
    X : xarray
    Y : xarray


    """
    if (X.time == Y.time).all():
        pass
    else:
        msg = "Time axes of arrays not identical"
        #  "NAN Handling could fail if X,Y dont share time axes"
        logger.warning(msg)
        raise ValueError(msg)
    return


def adjust_band_for_coherence_sorting(frequency_band, spectrogram, rule="min3"):
    """

    WIP: Intended to broaden band to allow more FCs for spectral features
    - used in coherence sorting and general feature extraction

    Parameters
    ----------
    frequency_band
    spectrogram: Spectrogram
    rule

    Returns
    -------

    """
    band = frequency_band.copy()
    if spectrogram.num_harmonics_in_band(band) == 1:
        logger.warning("Cant evaluate coherence with only 1 harmonic")
        logger.info(f"Widening band according to {rule} rule")
        if rule == "min3":
            band.frequency_min -= spectrogram.frequency_increment
            band.frequency_max += spectrogram.frequency_increment
        else:
            msg = f"Band adjustment rule {rule} not recognized"
            logger.error(msg)
            raise NotImplementedError(msg)
    return band


# TODO: move this to mth5.processing.spectre
def get_band_for_coherence_sorting(
    frequency_band,
    dec_level_config: AuroraDecimationLevel,
    local_stft_obj,
    remote_stft_obj,
    widening_rule="min3",
):
    """
    Just like get_band_for_tf_estimate, but here we enforce some rules so that the band is not one FC wide
    - it is possible that this method will get merged with get_band_for_tf_estimate
    - this is a placeholder until the appropriate rules are sorted out.

    Parameters
    ----------
    band : mt_metadata.transfer_functions.processing.aurora.FrequencyBands
        object with lower_bound and upper_bound to tell stft object which
        subarray to return
    config : AuroraDecimationLevel
        information about the input and output channels needed for TF
        estimation problem setup
    local_stft_obj : xarray.core.dataset.Dataset or None
        Time series of Fourier coefficients for the station whose TF is to be
        estimated
    remote_stft_obj : xarray.core.dataset.Dataset or None
        Time series of Fourier coefficients for the remote reference station

    Returns
    -------
    X, Y, RR : xarray.core.dataset.Dataset or None
        data structures as local_stft_object and remote_stft_object, but
        restricted only to input_channels, output_channels,
        reference_channels and also the frequency axes are restricted to
        being within the frequency band given as an input argument.
    """
    from mth5.timeseries.spectre import Spectrogram

    band = frequency_band.copy()
    logger.info(
        f"Processing band {band.center_period:.6f}s  ({1./band.center_period:.6f}Hz)"
    )
    stft = Spectrogram(local_stft_obj)
    if stft.num_harmonics_in_band(band) == 1:
        logger.warning("Cant evaluate coherence with only 1 harmonic")
        logger.info(f"Widening band according to {widening_rule} rule")
        if widening_rule == "min3":
            band.frequency_min -= stft.frequency_increment
            band.frequency_max += stft.frequency_increment
        else:
            msg = f"Widening rule {widening_rule} not recognized"
            logger.error(msg)
            raise NotImplementedError(msg)
    # return band TODO: Make this return the band only, then get later
    # proceed as in
    x, y, rr = get_band_for_tf_estimate(
        band, dec_level_config, local_stft_obj, remote_stft_obj
    )
    return x, y, rr, band


def cross_spectra(X, Y):
    """WIP: Returns the cross power spectra between two arrays"""
    return X.conj() * Y
