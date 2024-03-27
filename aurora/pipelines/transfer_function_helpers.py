"""
Note #1: repeatedly applying edf_weights seems to have no effect at all.
tested 20240118 and found that test_compare in synthetic passed whether this was commented
or not.  TODO confirm this is a one-and-done add doc about why this is so.
"""
import numpy as np

from aurora.time_series.frequency_band_helpers import adjust_band_for_coherence_sorting

# from aurora.time_series.frequency_band_helpers import get_band_for_tf_estimate
from aurora.time_series.spectrogram import Spectrogram
from aurora.time_series.xarray_helpers import handle_nan
from aurora.transfer_function.regression import get_regression_estimator
from aurora.transfer_function.regression.iter_control import IterControl

# from aurora.transfer_function.weights.coherence_weights import compute_multiple_coherence_weights
from aurora.transfer_function.weights.edf_weights import (
    effective_degrees_of_freedom_weights,
)
from loguru import logger


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


def drop_nans(X, Y, RR):
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
    """
    Reshape 2D arrays of frequency and time to 1D

    Context:
    When the data for a frequency band are extracted from the Spectrogram, each channel
    is a 2D array, one axis is time (the time of the window that was FFT-ed) and the
    other axis is frequency.  However if we make no distinction between the harmonics
    (bins) within a band in regression, then all the FCs for each channel can be
    put into a 1D array.  This method performs that reshaping (ravelling) operation.
    **It is not important how we unravel the FCs but it is important that
    we use the same scheme for X and Y.

    TODO: Make this take a list and return a list rather than X,Y,RR
    TODO: Decorate this with @dataset_or_dataarray
        if isinstance(X, xr.Dataset):
        tmp = X.to_array("channel")
        tmp = tmp.stack()
        or similar

    """
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
        X, Y, RR = drop_nans(X, Y, RR)

    return X, Y, RR


def process_transfer_functions(
    dec_level_config,
    merged_stft_object,
    transfer_function_obj,
    # segment_weights=["multiple_coherence",],#["simple_coherence",],#["multiple_coherence",],#jj84_coherence_weights",],
    segment_weights=[],
    channel_weights=None,
):
    """
    This method based on TTFestBand.m

    Parameters
    ----------
    dec_level_config
    merged_stft_object: xarray.core.dataset.Dataset
        The time series of Fourier coefficients from the local station, with optional
        keys ["rx", "ry"] for remote reference channels

    transfer_function_obj: aurora.transfer_function.TTFZ.TTFZ
        The transfer function container ready to receive values in this method.
    segment_weights : numpy array or list of strings
        1D array which should be of the same length as the time axis of the STFT objects
        If these weights are zero anywhere, we drop all that segment across all channels
        If it is a list of strings, each string corresponds to a weighting
        algorithm to be applied.
        ["jackknife_jj84", "multiple_coherence", "simple_coherence"]
    channel_weights : numpy array or None

    Note #1: Although it is advantageous to executing the regression channel-by-channel
    vs. all-at-once, we need to keep the all-at-once to get residual covariances (see issue #87)

    Note #2:
    Consider placing the segment weight logic in its own module with the various functions in a dictionary.
    Possibly can combines (product) all segment weights, like the following pseudocode:

        W = zeros
        for wt_style in  segment_weights:
            fcn = wt_fucntions[style]
            w = fcn(X, Y, RR, )
            W *= w
        return W


    TODO: Consider push the nan-handling into the band extraction as a kwarg.
    TODO: Since RR officially becomes part of the nomenclature here, we can
    Returns
    -------

    """
    # Also consider applying channel nomenlclature map to standard channels for regression, map back when done.

    estimator_class = get_regression_estimator(dec_level_config.estimator.engine)
    iter_control = set_up_iter_control(dec_level_config)
    spectrogram = Spectrogram(dataset=merged_stft_object)
    for band in transfer_function_obj.frequency_bands.bands():

        # Uncomment for Testing -- this will make a 3xFC-wide spectrogram if there is only 1 FC
        rule = "min3"  # TODO: Put band-adjustment rule into processing config
        # adjusted_band = adjust_band_for_coherence_sorting(band, spectrogram, rule=rule)
        # band_spectrogram = spectrogram.extract_band(adjusted_band)

        band_spectrogram = spectrogram.extract_band(band)
        X = band_spectrogram.dataset[dec_level_config.input_channels]
        Y = band_spectrogram.dataset[dec_level_config.output_channels]
        if dec_level_config.reference_channels:
            RR = band_spectrogram.dataset[["rx", "ry"]]
        else:
            RR = None

        # Apply segment weights first -- see Note #2

        if "jackknife_jj84" in segment_weights:
            from aurora.transfer_function.weights.coherence_weights import (
                coherence_weights_jj84,
            )

            rule = "min3"  # TODO: Put band-adjustment rule into processing config
            adjusted_band = adjust_band_for_coherence_sorting(
                band, spectrogram, rule=rule
            )
            weight_band_spectrogram = spectrogram.extract_band(adjusted_band)

            Wjj84 = coherence_weights_jj84(weight_band_spectrogram)
            apply_weights(X, Y, RR, Wjj84, segment=True, dropna=False)
        if "simple_coherence" in segment_weights:
            # Note that these weights might be better applied within the loop over channel as the weights
            # may be differ depending on the output channel being estimated.
            # We can compute them here, but
            from aurora.transfer_function.weights.spectral_features import (
                estimate_simple_coherence,
            )
            from aurora.transfer_function.weights.coherence_weights import (
                simple_coherence_weights,
            )

            # def merge_simple_coherence_weights(weights):
            #     """
            #
            #     Parameters
            #     ----------
            #     weights dict
            #
            #     Returns
            #     -------
            #
            #     """
            #     keys = list(weights.keys())[0]
            #     n_obs = len(weights[keys[0]])
            #     cumulative_weights np.ones(n_obs)
            #     for k,v in weights.items():
            #         cumulative_weights *= v
            rule = "min3"  # TODO: Put band-adjustment rule into processing config
            adjusted_band = adjust_band_for_coherence_sorting(
                band, spectrogram, rule=rule
            )
            weight_band_spectrogram = spectrogram.extract_band(adjusted_band)

            # Optionally add rx, ry to spectrogram here
            channel_pairs = (
                ("ex", "hy"),
                ("ey", "hx"),
            )
            simple_coherences = estimate_simple_coherence(
                weight_band_spectrogram, channel_pairs=channel_pairs
            )
            # TODO: Put cutoffs in the procesing config:
            cutoffs = {}
            cutoffs["ex_hy"] = {}
            cutoffs["ex_hy"]["lower"] = 0.3
            cutoffs["ex_hy"]["upper"] = 0.999
            cutoffs["ey_hx"] = {}
            cutoffs["ey_hx"]["lower"] = 0.3
            cutoffs["ey_hx"]["upper"] = 0.999

            W = simple_coherence_weights(simple_coherences, channel_pair=("ex", "hy"))
            # try:
            #     W = merge_simple_coherence_weights(simple_coh_weights)
            # except:
            #     pass
            #     # # Get cumulative rankings
            #     # for key in cutoffs.keys():
            #     #     simple_coherence = simple_coherences[key]
            #     #     cumulative_rankings += np.argsort(
            #     #         simple_coherence)  # keep track of lowest and highest values per time window
            # apply_weights(X, Y, RR, W, segment=True, dropna=False)

        if "multiple_coherence" in segment_weights:
            from aurora.transfer_function.weights.coherence_weights import (
                multiple_coherence_weights,
            )

            rule = "min3"  # TODO: Put band-adjustment rule into processing config
            adjusted_band = adjust_band_for_coherence_sorting(
                band, spectrogram, rule=rule
            )
            weight_band_spectrogram = spectrogram.extract_band(adjusted_band)
            W = multiple_coherence_weights(
                weight_band_spectrogram
            )  # band, local_stft_obj, remote_stft_obj)
            apply_weights(X, Y, RR, W, segment=True, dropna=False)

        # if there are channel weights apply them here

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
