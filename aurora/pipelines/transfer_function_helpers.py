import numpy as np

from aurora.time_series.frequency_band_helpers import get_band_for_tf_estimate
from aurora.time_series.xarray_helpers import handle_nan
from aurora.transfer_function.regression.base import RegressionEstimator
from aurora.transfer_function.regression.iter_control import IterControl
from aurora.transfer_function.regression.TRME import TRME
from aurora.transfer_function.regression.TRME_RR import TRME_RR

# from aurora.transfer_function.weights.coherence_weights import compute_multiple_coherence_weights
from aurora.transfer_function.weights.edf_weights import (
    effective_degrees_of_freedom_weights,
)
from loguru import logger


ESTIMATOR_LIBRARY = {"OLS": RegressionEstimator, "RME": TRME, "RME_RR": TRME_RR}


def get_estimator_class(estimation_engine):
    """

    Parameters
    ----------
    estimation_engine: str
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


def set_up_iter_control(config):
    """
    TODO: Review: maybe better to just make this the __init__ method of the
    IterControl object, iter_control = IterControl(config)


    Parameters
    ----------
    config: mt_metadata.transfer_functions.processing.aurora.decimation_level.DecimationLevel

    Returns
    -------
    iter_control: aurora.transfer_function.regression.iter_control.IterControl
        Object with parameters about iteration control in regression
    """
    if config.estimator.engine in ["RME", "RME_RR"]:
        iter_control = IterControl(
            max_number_of_iterations=config.regression.max_iterations,
            max_number_of_redescending_iterations=config.regression.max_redescending_iterations,
        )
    elif config.estimator.engine in [
        "OLS",
    ]:
        iter_control = None
    return iter_control


def select_channel(xrda, channel_label):
    """
    Extra helper function to make process_transfer_functions more readable without
    black forcing multiline
    Parameters
    ----------
    xrda
    channel_label

    Returns
    -------

    """
    ch = xrda.sel(
        channel=[
            channel_label,
        ]
    )
    return ch


def dropna(X, Y, RR):
    """
    Just a helper intended to enhance readability
    TODO: document the implications of dropna on index of xarray for other weights
    """
    X = X.dropna(dim="observation")
    Y = Y.dropna(dim="observation")
    if RR is not None:
        RR = RR.dropna(dim="observation")
    return X, Y, RR


def stack_fcs(X, Y, RR):
    """Reshape 2D arrays of frequency and time to 1D"""
    X = X.stack(observation=("frequency", "time"))
    Y = Y.stack(observation=("frequency", "time"))
    if RR is not None:
        RR = RR.stack(observation=("frequency", "time"))
    return X, Y, RR


def apply_weights(X, Y, RR, W, segment=False, dropna=False):
    W[W == 0] = np.nan
    if segment:
        W = np.atleast_2d(W).T
    X *= W
    Y *= W
    if RR is not None:
        RR *= W

    if dropna:
        X, Y, RR = dropna(X, Y, RR)

    return X, Y, RR


def process_transfer_functions(
    dec_level_config,
    local_stft_obj,
    remote_stft_obj,
    transfer_function_obj,
    # segment_weights=["jj84_coherence_weights",],
    segment_weights=[],
    channel_weights=None,
):
    """
    This method based on TTFestBand.m

    Parameters
    ----------
    dec_level_config
    local_stft_obj
    remote_stft_obj
    transfer_function_obj: aurora.transfer_function.TTFZ.TTFZ
        The transfer function container ready to receive values in this method.
    segment_weights : numpy array or None
        1D array which should be of the same length as the time axis of the STFT objects
        If these weights are zero anywhere, we drop all that segment across all channels
    channel_weights : numpy array or None


    TODO:
    1. Review the advantages of executing the regression all at once vs
    channel-by-channel.  If there is not disadvantage to always
    using a channel-by-channel approach we can modify this to only support that
    method.  However, we still need a way to get residual covariances (see issue #87)
    2. Consider push the nan-handling into the band extraction as a
    kwarg.
    3. The reindexing of the band may be done in the extraction as well.  This would
    result in an "edf-weighting-scheme-ready" format.

    Returns
    -------

    """
    estimator_class = get_estimator_class(dec_level_config.estimator.engine)
    iter_control = set_up_iter_control(dec_level_config)
    for band in transfer_function_obj.frequency_bands.bands():

        X, Y, RR = get_band_for_tf_estimate(
            band, dec_level_config, local_stft_obj, remote_stft_obj
        )

        # Apply segment weights first
        # This could be replaced by a method that combines (product) all segment weights in a dict
        # weights = {}
        if "jj84_coherence_weights" in segment_weights:
            from aurora.transfer_function.weights.coherence_weights import (
                coherence_weights_jj84,
            )

            Wjj84 = coherence_weights_jj84(band, local_stft_obj, remote_stft_obj)
            apply_weights(X, Y, RR, Wjj84, segment=True, dropna=False)

        # if multiple_coherence_weights in segment_weights:
        #     from aurora.transfer_function.weights.coherence_weights import compute_multiple_coherence_weights
        #     Wmc = compute_multiple_coherence_weights(band, local_stft_obj, remote_stft_obj)
        #     apply_segment_weights(X, Y, RR, Wmc)

        # if there are channel weights apply them here

        # Reshape to 2d
        X, Y, RR = stack_fcs(X, Y, RR)

        # Should only be needed if weights were applied
        X, Y, RR = dropna(X, Y, RR)

        W = effective_degrees_of_freedom_weights(X, RR, edf_obj=None)
        X, Y, RR = apply_weights(X, Y, RR, W, segment=False, dropna=True)

        if dec_level_config.estimator.estimate_per_channel:
            for ch in dec_level_config.output_channels:
                Y_ch = Y[ch].to_dataset()  # keep as a dataset, maybe not needed

                X_, Y_, RR_ = handle_nan(X, Y_ch, RR, drop_dim="observation")

                W = effective_degrees_of_freedom_weights(X_, RR_, edf_obj=None)
                X_, Y_, RR_ = apply_weights(X_, Y_, RR_, W, segment=False)

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
