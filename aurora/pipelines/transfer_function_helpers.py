"""
Split this into a module for time_domain processing helpers,
and transfer_function_processing helpers.


"""
from aurora.time_series.frequency_band_helpers import extract_band

# from aurora.time_series.xarray_helpers import cast_3d_stft_to_2d_observations
from aurora.time_series.xarray_helpers import handle_nan
from aurora.transfer_function.iter_control import IterControl
from aurora.transfer_function.transfer_function_header import TransferFunctionHeader
from aurora.transfer_function.regression.TRME import TRME
from aurora.transfer_function.regression.TRME_RR import TRME_RR

# <TF PROCESSING HELPERS>
# TODO: Make all these regression methods accept kwargs on init so that
# none of them choke if we pass them Z=None or iter_control=None
from aurora.transfer_function.regression.base import RegressionEstimator
from aurora.transfer_function.weights.edf_weights import (
    effective_degrees_of_freedom_weights,
)

REGRESSION_LIBRARY = {"OLS": RegressionEstimator, "RME": TRME, "TRME_RR": TRME_RR}


def get_regression_class(config):
    try:
        regression_class = REGRESSION_LIBRARY[config.estimation_engine]
    except KeyError:
        print(f"processing_scheme {config.estimation_engine} not supported")
        print(f"processing_scheme must be one of {list(REGRESSION_LIBRARY.keys())}")
        raise Exception
    return regression_class


def set_up_iter_control(config):
    """
    TODO: Review: maybe better to just make this the __init__ method of the
    IterControl object, iter_control = IterControl(config)


    Parameters
    ----------
    config

    Returns
    -------

    """
    if config.estimation_engine in ["RME", "TRME_RR"]:
        iter_control = IterControl(
            max_number_of_iterations=config.max_number_of_iterations
        )
    elif config.estimation_engine in [
        "OLS",
    ]:
        iter_control = None
    return iter_control


def transfer_function_header_from_config(config):
    transfer_function_header = TransferFunctionHeader(
        processing_scheme=config.estimation_engine,
        local_station_id=config.local_station_id,
        reference_station_id=config.reference_station_id,
        input_channels=config.input_channels,
        output_channels=config.output_channels,
        reference_channels=config.reference_channels,
    )
    return transfer_function_header


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


def get_band_for_tf_estimate(band, config, local_stft_obj, remote_stft_obj):
    """
    Get data for TF estimation for a particular band.

    Parameters
    ----------
    band : aurora.time_series.frequency_band.FrequencyBand
        object with lower_bound and upper_bound to tell stft object which
        subarray to return
    config : aurora.config.decimation_level_config.DecimationLevelConfig
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
    print(f"Processing band {band.center_period}s")
    band_dataset = extract_band(band, local_stft_obj)
    X = band_dataset[config.input_channels]
    Y = band_dataset[config.output_channels]
    check_time_axes_synched(X, Y)
    if config.reference_station_id:
        band_dataset = extract_band(band, remote_stft_obj)
        RR = band_dataset[config.reference_channels]
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
    local_stft_obj,
    remote_stft_obj,
    transfer_function_obj,
    segment_weights=None,
    channel_weights=None,
):
    """
    This method is very similar to TTFestBand.m

    Parameters
    ----------
    config
    local_stft_obj
    remote_stft_obj
    transfer_function_obj
    segment_weights : numpy array or None
        1D array which should be of the same length as the time axis of the STFT objects
        If these weights are zero anywhere, we drop all that segment across all channels
    channel_weights : numpy array or None


    TODO: Review the advantages of excuting the regression all at once vs
    channel-by-channel.  If there is not disadvantage to always
    using a channel-by-channel approach we can modify this to only support that
    method.  We also may want to push the nan-handling into the band extraction as a
    kwarg.  Finally, the reindexing of the band may be done in the extraction as
    well.  This would result in an "edf-weighting-scheme-ready" format.
    Returns
    -------

    """
    iter_control = set_up_iter_control(config)
    regression_class = get_regression_class(config)
    for band in transfer_function_obj.frequency_bands.bands():
        X, Y, RR = get_band_for_tf_estimate(
            band, config, local_stft_obj, remote_stft_obj
        )
        # if there are segment weights apply them here
        # if there are channel weights apply them here
        # Reshape to 2d - maybe push this into extract band method
        X = X.stack(observation=("frequency", "time"))
        Y = Y.stack(observation=("frequency", "time"))
        if RR is not None:
            RR = RR.stack(observation=("frequency", "time"))

        W = effective_degrees_of_freedom_weights(X, RR, edf_obj=None)
        # apply weights
        X *= W
        Y *= W
        if RR is not None:
            RR *= W

        if config.estimate_per_channel:
            for ch in config.output_channels:
                Y_ch = Y[ch].to_dataset()  # keep as a dataset, maybe not needed

                X_, Y_, RR_ = handle_nan(X, Y_ch, RR, drop_dim="observation")
                regression_estimator = regression_class(
                    X=X_, Y=Y_, Z=RR_, iter_control=iter_control
                )
                regression_estimator.estimate()
                transfer_function_obj.set_tf(regression_estimator, band.center_period)
            print("Add method for compute residuals and noise covariance")
        else:
            X, Y, RR = handle_nan(X, Y, RR, drop_dim="observation")
            regression_estimator = regression_class(
                X=X, Y=Y, Z=RR, iter_control=iter_control
            )
            regression_estimator.estimate()
            transfer_function_obj.set_tf(regression_estimator, band.center_period)

    return transfer_function_obj
