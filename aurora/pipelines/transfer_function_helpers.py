import numpy as np

from aurora.time_series.frequency_band_helpers import extract_band
from aurora.time_series.xarray_helpers import handle_nan
from aurora.transfer_function.regression.iter_control import IterControl
from aurora.transfer_function.regression.TRME import TRME
from aurora.transfer_function.regression.TRME_RR import TRME_RR

from aurora.transfer_function.regression.base import RegressionEstimator
from aurora.transfer_function.weights.edf_weights import (
    effective_degrees_of_freedom_weights,
)

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
        print(f"processing_scheme {estimation_engine} not supported")
        print(f"processing_scheme must be one of {list(ESTIMATOR_LIBRARY.keys())}")
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


def check_time_axes_synched(X, Y):
    """
    Utility function for checking that time axes agree

    Parameters
    ----------
    X : xarray
    Y : xarray

    Returns
    -------

    """
    """
    It is critical that X, Y, RR have the same time axes here

    Returns
    -------

    """
    if (X.time == Y.time).all():
        pass
    else:
        print("WARNING - NAN Handling could fail if X,Y dont share time axes")
        raise Exception
    return


def get_band_for_tf_estimate(
    band, config, i_dec_level, local_stft_obj, remote_stft_obj
):
    """
    Get data for TF estimation for a particular band.

    Parameters
    ----------
    band : mt_metadata.transfer_functions.processing.aurora.FrequencyBands
        object with lower_bound and upper_bound to tell stft object which
        subarray to return
    config : mt_metadata.transfer_functions.processing.aurora.decimation_level.DecimationLevel
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
    dec_level_config = config.decimations[0]
    print(f"Processing band {band.center_period:.6f}s")
    band_dataset = extract_band(band, local_stft_obj)
    X = band_dataset[dec_level_config.input_channels]
    Y = band_dataset[dec_level_config.output_channels]
    check_time_axes_synched(X, Y)
    if config.stations.remote:
        band_dataset = extract_band(band, remote_stft_obj)
        RR = band_dataset[dec_level_config.reference_channels]
        check_time_axes_synched(Y, RR)
    else:
        RR = None

    return X, Y, RR


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


def process_transfer_functions(
    config,
    i_dec_level,
    local_stft_obj,
    remote_stft_obj,
    transfer_function_obj,
    segment_weights=None,
    channel_weights=None,
):
    """
    This method based on TTFestBand.m

    Parameters
    ----------
    config
    local_stft_obj
    remote_stft_obj
    transfer_function_obj: aurora.transfer_function.TTFZ.TTFZ
        The transfer function container ready to receive values in this method.
    segment_weights : numpy array or None
        1D array which should be of the same length as the time axis of the STFT objects
        If these weights are zero anywhere, we drop all that segment across all channels
    channel_weights : numpy array or None


    TODO:
    1. Review the advantages of excuting the regression all at once vs
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
    # PUT COHERENCE SORTING HERE IF WIDE BAND?
    dec_level_config = config.decimations[i_dec_level]
    estimator_class = get_estimator_class(dec_level_config.estimator.engine)
    for band in transfer_function_obj.frequency_bands.bands():
        iter_control = set_up_iter_control(dec_level_config)
        X, Y, RR = get_band_for_tf_estimate(
            band, config, i_dec_level, local_stft_obj, remote_stft_obj
        )
        # if there are segment weights apply them here
        # if there are channel weights apply them here
        # Reshape to 2d - maybe push this into extract band method
        X = X.stack(observation=("frequency", "time"))
        Y = Y.stack(observation=("frequency", "time"))
        if RR is not None:
            RR = RR.stack(observation=("frequency", "time"))

        W = effective_degrees_of_freedom_weights(X, RR, edf_obj=None)
        W[W == 0] = np.nan  # use this to drop values in the handle_nan
        # apply weights
        X *= W
        Y *= W
        if RR is not None:
            RR *= W
        X = X.dropna(dim="observation")
        Y = Y.dropna(dim="observation")
        if RR is not None:
            RR = RR.dropna(dim="observation")

        # INSERT COHERENCE SORTING HERE>
        # coh_type = "local"
        # if i_dec_level == 0:
        #     from aurora.transfer_function.weights.coherence_weights import compute_coherence_weights
        #     X, Y, RR = compute_coherence_weights(X,Y,RR, coh_type=coh_type)

        if dec_level_config.estimator.estimate_per_channel:
            for ch in dec_level_config.output_channels:
                Y_ch = Y[ch].to_dataset()  # keep as a dataset, maybe not needed

                X_, Y_, RR_ = handle_nan(X, Y_ch, RR, drop_dim="observation")

                # W = effective_degrees_of_freedom_weights(X_, RR_, edf_obj=None)
                # X_ *= W
                # Y_ *= W
                # if RR is not None:
                #     RR_ *= W

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
