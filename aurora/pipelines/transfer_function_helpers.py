"""
This module contains helper methods that are used during transfer function processing.

Development Notes:
 Note #1: repeatedly applying edf_weights seems to have no effect at all.
 tested 20240118 and found that test_compare in synthetic passed whether this was commented
 or not.  TODO confirm this is a one-and-done add doc about why this is so.

"""

from aurora.time_series.frequency_band_helpers import get_band_for_tf_estimate
from aurora.time_series.xarray_helpers import handle_nan
from aurora.transfer_function.regression.base import RegressionEstimator
from aurora.transfer_function.regression.iter_control import IterControl
from aurora.transfer_function.regression.RME import RME
from aurora.transfer_function.regression.RME_RR import RME_RR

from aurora.transfer_function.weights.edf_weights import (
    effective_degrees_of_freedom_weights,
)
from mt_metadata.transfer_functions.processing.aurora.decimation_level import (
    DecimationLevel as AuroraDecimationLevel,
)
from loguru import logger
from typing import Literal, Optional, Union

import numpy as np
import xarray as xr

ESTIMATOR_LIBRARY = {"OLS": RegressionEstimator, "RME": RME, "RME_RR": RME_RR}
SUPPORTED_REGRESSION_ESTIMATOR = Literal["OLS", "RME", "RME_RR"]


def get_estimator_class(
    estimation_engine: SUPPORTED_REGRESSION_ESTIMATOR,
) -> RegressionEstimator:
    """

    Parameters
    ----------
    estimation_engine: Literal["OLS", "RME", "RME_RR"]
        One of the keys in the ESTIMATOR_LIBRARY, designates the method that will be
        used to estimate the transfer function

    Returns
    -------
    estimator_class: aurora.transfer_function.regression.base.RegressionEstimator
        The class that will do the TF estimation
    """
    try:
        estimator_class = ESTIMATOR_LIBRARY[estimation_engine]
    except KeyError:
        logger.error(f"processing_scheme {estimation_engine} not supported")
        logger.error(
            f"processing_scheme must be one of {list(ESTIMATOR_LIBRARY.keys())}"
        )
        raise Exception
    return estimator_class


def set_up_iter_control(config: AuroraDecimationLevel):
    """
    Initializes an IterControl object based on values in the processing config.

    Development Notes:
    TODO: Review: maybe better to just make this the __init__ method of the IterControl object, iter_control = IterControl(config)


    Parameters
    ----------
    config: AuroraDecimationLevel
        metadata about the decimation level processing.

    Returns
    -------
    iter_control: aurora.transfer_function.regression.iter_control.IterControl
        Object with parameters about iteration control in regression
    """
    if config.estimator.engine in ["RME", "RME_RR"]:
        iter_control = IterControl(
            max_number_of_iterations=config.regression.max_iterations,
            max_number_of_redescending_iterations=config.regression.max_redescending_iterations,
            r0=config.regression.r0,
            u0=config.regression.u0,
            tolerance=config.regression.tolerance,
        )
    elif config.estimator.engine in [
        "OLS",
    ]:
        iter_control = None

    return iter_control


def drop_nans(X: xr.Dataset, Y: xr.Dataset, RR: Union[xr.Dataset, None]) -> tuple:
    """
    Drops any observation where any variable in X, Y, or RR is NaN.
    """
    import numpy as np

    def get_obs_mask(ds):
        """
        Generate a boolean mask indicating which 'observation' entries are finite across all data variables in an xarray Dataset.

        This function iterates over all data variables in the provided xarray Dataset `ds`, checks for finite values (i.e., not NaN or infinite)
        along all axes except the 'observation' axis, and combines the results to produce a single boolean mask. The resulting mask is True
        for each 'observation' index where all data variables are finite, and False otherwise.


        Parameters
        ds : xarray.Dataset
            The input dataset containing data variables with an 'observation' dimension.

        Returns
        numpy.ndarray
            A boolean array with shape matching the 'observation' dimension, where True indicates all data variables are finite
        """
        mask = None
        for v in ds.data_vars.values():
            # Reduce all axes except 'observation'
            axes = tuple(i for i, d in enumerate(v.dims) if d != "observation")
            this_mask = np.isfinite(v)
            if axes:
                this_mask = this_mask.all(axis=axes)
            mask = this_mask if mask is None else mask & this_mask
        return mask

    mask = get_obs_mask(X)
    mask = mask & get_obs_mask(Y)
    if RR is not None:
        mask = mask & get_obs_mask(RR)

    X = X.isel(observation=mask)
    Y = Y.isel(observation=mask)
    if RR is not None:
        RR = RR.isel(observation=mask)
    return X, Y, RR


def stack_fcs(X, Y, RR):
    """
    Reshape 2D arrays of frequency and time to 1D.

    Notes: When the data for a frequency band are extracted from the Spectrogram, each
    channel is a 2D array, one axis is time (the time of the window that was FFT-ed) and the
    other axis is frequency. However if we make no distinction between the harmonics
    (bins) within a band in regression, then all the FCs for each channel can be
    put into a 1D array.  This method performs that reshaping (ravelling) operation.
    It is not important how we unravel the FCs but it is important that the same indexing
    scheme is used for X, Y and RR.

    TODO: Consider this take a list and return a list rather than X,Y,RR

    TODO: Consider decorate this with @dataset_or_dataarray

    Parameters
    ----------
    X: xarray.core.dataset.Dataset
    Y: xarray.core.dataset.Dataset
    RR: xarray.core.dataset.Dataset or None

    Returns
    -------
    X, Y, RR: Same as input but with stacked time and frequency dimensions
    """
    X = X.stack(observation=("frequency", "time"))
    Y = Y.stack(observation=("frequency", "time"))
    if RR is not None:
        RR = RR.stack(observation=("frequency", "time"))
    return X, Y, RR


def apply_weights(
    X: xr.Dataset,
    Y: xr.Dataset,
    RR: xr.Dataset,
    W: np.ndarray,
    segment: bool = False,
    dropna: bool = False,
) -> tuple:
    """
    Applies data weights (W) to each of X, Y, RR.
    If weight is zero, we set to nan and optionally dropna.

    Parameters
    ----------
    X: xarray.core.dataset.Dataset
    Y: xarray.core.dataset.Dataset
    RR: xarray.core.dataset.Dataset or None
    W: numpy array
        The Weights to apply to the data
    segment: bool
        If True the weights may need to be reshaped.
    dropna: bool
        Whether or not to drop zero-weighted data.  If true, we drop the nans.

    Returns
    -------
    X, Y, RR: tuple
        Same as input but with weights applied and (optionally) nan dropped.
    """
    W[W == 0] = np.nan
    if segment:
        W = np.atleast_2d(W).T
    X *= W
    Y *= W
    if RR is not None:
        RR *= W

    if dropna:
        X, Y, RR = drop_nans(X, Y, RR)

    return X, Y, RR


def process_transfer_functions(
    dec_level_config: AuroraDecimationLevel,
    local_stft_obj: xr.Dataset,
    remote_stft_obj: xr.Dataset,
    transfer_function_obj,
):
    """
    This is the main tf_processing method.  It is based on the Matlab legacy code TTFestBand.m.

    Note #1: Although it is advantageous to execute the regression channel-by-channel
    vs. all-at-once, we need to keep the all-at-once to get residual covariances (see aurora issue #87)

    TODO: Consider push the nan-handling into the band extraction as a kwarg.

    Parameters
    ----------
    dec_level_config: AuroraDecimationLevel
        Processing parameters for the active decimation level.
    local_stft_obj: xarray.core.dataset.Dataset
    remote_stft_obj: xarray.core.dataset.Dataset or None
    transfer_function_obj: aurora.transfer_function.TTFZ.TTFZ
        The transfer function container ready to receive values in this method.

    Returns
    -------
    transfer_function_obj: aurora.transfer_function.TTFZ.TTFZ
    """
    estimator_class: RegressionEstimator = get_estimator_class(
        dec_level_config.estimator.engine
    )
    iter_control = set_up_iter_control(dec_level_config)
    for band in transfer_function_obj.frequency_bands.bands():

        X, Y, RR = get_band_for_tf_estimate(
            band, dec_level_config, local_stft_obj, remote_stft_obj
        )

        # Reshape to 2d
        X, Y, RR = stack_fcs(X, Y, RR)

        # Should only be needed if weights were applied
        X, Y, RR = drop_nans(X, Y, RR)

        W = effective_degrees_of_freedom_weights(X, RR, edf_obj=None)
        X, Y, RR = apply_weights(X, Y, RR, W, segment=False, dropna=True)

        if dec_level_config.estimator.estimate_per_channel:
            for ch in dec_level_config.output_channels:

                Y_ch = Y[ch].to_dataset()  # keep as a dataset, maybe not needed

                X_, Y_, RR_ = handle_nan(X, Y_ch, RR, drop_dim="observation")

                # see note #1
                # if RR is not None:
                #     W = effective_degrees_of_freedom_weights(X_, RR_, edf_obj=None)
                #     X_, Y_, RR_ = apply_weights(X_, Y_, RR_, W, segment=False)

                regression_estimator = estimator_class(
                    X=X_, Y=Y_, Z=RR_, iter_control=iter_control
                )
                regression_estimator.estimate()
                transfer_function_obj.set_tf(regression_estimator, band.center_period)
        else:
            X, Y, RR = handle_nan(X, Y, RR, drop_dim="observation")
            regression_estimator = estimator_class(
                X=X, Y=Y, Z=RR, iter_control=iter_control
            )
            regression_estimator.estimate()
            transfer_function_obj.set_tf(regression_estimator, band.center_period)

    return transfer_function_obj


def process_transfer_functions_with_weights(
    dec_level_config: AuroraDecimationLevel,
    local_stft_obj: xr.Dataset,
    remote_stft_obj: xr.Dataset,
    transfer_function_obj,
):
    """
    This is version of process_transfer_functions applies weights to the data.

    Development Notes:
    Note #1: This is only for per-channel estimation, so it does not support the
    dec_level_config.estimator.estimate_per_channel = False
    Note #2: This was adapted from the process_transfer_functions method but the core loop
    is inverted to loop over channels first, then bands.

    Parameters
    ----------
    dec_level_config: AuroraDecimationLevel
        Processing parameters for the active decimation level.
    local_stft_obj: xarray.core.dataset.Dataset
    remote_stft_obj: xarray.core.dataset.Dataset or None
    transfer_function_obj: aurora.transfer_function.TTFZ.TTFZ
        The transfer function container ready to receive values in this method.

    Returns
    -------
    transfer_function_obj: aurora.transfer_function.TTFZ.TTFZ
    """
    if not dec_level_config.estimator.estimate_per_channel:
        msg = (
            "process_transfer_functions_with_weights is only for per-channel estimation"
        )
        logger.error(msg)
        raise ValueError(msg)

    estimator_class: RegressionEstimator = get_estimator_class(
        dec_level_config.estimator.engine
    )
    iter_control = set_up_iter_control(dec_level_config)
    for ch in dec_level_config.output_channels:

        # check if there are channel weights for this channel
        weights = None
        for chws in dec_level_config.channel_weight_specs:
            if ch in chws.output_channels:
                weights = chws.weights

        for band in transfer_function_obj.frequency_bands.bands():

            X, Y, RR = get_band_for_tf_estimate(
                band, dec_level_config, local_stft_obj, remote_stft_obj
            )
            Y_ch = Y[ch].to_dataset()  # keep as a dataset, maybe not needed

            # extract the weights for this band
            if weights is not None:
                # TODO: Investigate best way to extract the weights for band
                #  This may involve finding the nearest frequency bin to the band center period
                #  and then applying the weights for that bin, or some tapered region around it.
                #  For now, we will just use the mean of the weights for the band.
                #  This is a temporary solution and should be replaced with a more robust method.
                # band_weights = chws.get_weights_for_band(band)
                band_weights = weights.mean(axis=1)  # chws.get_weights_for_band(band)

                apply_weights(
                    X, Y_ch, RR, band_weights.squeeze(), segment=True, dropna=False
                )

            X, Y_ch, RR = stack_fcs(X, Y_ch, RR)  # Reshape to 2d

            # Should only be needed if weights were applied
            X, Y_ch, RR = drop_nans(X, Y_ch, RR)

            W = effective_degrees_of_freedom_weights(X, RR, edf_obj=None)

            X, Y_ch, RR = apply_weights(X, Y_ch, RR, W, segment=False, dropna=True)
            X_, Y_, RR_ = handle_nan(X, Y_ch, RR, drop_dim="observation")
            regression_estimator = estimator_class(
                X=X_, Y=Y_, Z=RR_, iter_control=iter_control
            )
            regression_estimator.estimate()
            transfer_function_obj.set_tf(regression_estimator, band.center_period)

    return transfer_function_obj
