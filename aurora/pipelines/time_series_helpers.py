"""
    Collection of modules used in processing pipeline that operate on time series
"""
import mt_metadata
import numpy as np
import pandas as pd
import scipy.signal as ssig
import xarray as xr

from loguru import logger

from aurora.time_series.windowed_time_series import WindowedTimeSeries
from aurora.time_series.windowing_scheme import window_scheme_from_decimation
from mt_metadata.transfer_functions.processing import TimeSeriesDecimation
from mt_metadata.transfer_functions.processing.aurora.decimation_level import (
    DecimationLevel as AuroraDecimationLevel,
)
from mt_metadata.transfer_functions.processing.fourier_coefficients import (
    Decimation as FCDecimation,
)
from mth5.groups import RunGroup
from mth5.timeseries.spectre.prewhitening import apply_prewhitening
from mth5.timeseries.spectre.prewhitening import apply_recoloring
import mth5.timeseries.spectre as spectre
from typing import Literal, Optional, Union


def truncate_to_clock_zero(
    decimation_obj: Union[AuroraDecimationLevel, FCDecimation],
    run_xrds: RunGroup,
):
    """
    Compute the time interval between the first data sample and the clock zero
    Identify the first sample in the xarray time series that corresponds to a
    window start sample.

    Parameters
    ----------
    decimation_obj: Union[AuroraDecimationLevel, FCDecimation]
        Information about how the decimation level is to be processed
    run_xrds : xarray.core.dataset.Dataset
        normally extracted from mth5.RunTS


    Returns
    -------
    run_xrds : xarray.core.dataset.Dataset
        same as the input time series, but possibly slightly shortened
    """
    if decimation_obj.stft.window.clock_zero_type == "ignore":
        pass
    else:
        clock_zero = pd.Timestamp(decimation_obj.stft.window.clock_zero)
        clock_zero = clock_zero.to_datetime64()
        delta_t = clock_zero - run_xrds.time[0]
        assert delta_t.dtype == "<m8[ns]"  # expected in nanoseconds
        delta_t_seconds = int(delta_t) / 1e9
        if delta_t_seconds == 0:
            pass  # time series start is already clock zero
        else:
            windowing_scheme = window_scheme_from_decimation(decimation_obj)
            number_of_steps = delta_t_seconds / windowing_scheme.duration_advance
            n_partial_steps = number_of_steps - np.floor(number_of_steps)
            n_clip = n_partial_steps * windowing_scheme.num_samples_advance
            n_clip = int(np.round(n_clip))
            t_clip = run_xrds.time[n_clip]
            cond1 = run_xrds.time >= t_clip
            msg = (
                f"dropping {n_clip} samples to agree with "
                f"{decimation_obj.stft.window.clock_zero_type} clock zero {clock_zero}"
            )
            logger.info(msg)
            run_xrds = run_xrds.where(cond1, drop=True)
    return run_xrds


def nan_to_mean(xrds: xr.Dataset) -> xr.Dataset:
    """
    Set Nan values in xr.Dataset to the mean value per channel.

    xrds: xr.Dataset
        Time series data

    Returns
    -------
    run_xrds : xr.Dataset
        The same as the input time series but NaN values are replaced by the mean of the time series (per channel).

    """
    for ch in xrds.keys():
        null_values_present = xrds[ch].isnull().any()
        if null_values_present:
            msg = "Unexpected Null values in xrds -- this should be examined"
            logger.warning(msg)
            value = np.nan_to_num(np.nanmean(xrds[ch].data))
            xrds[ch] = xrds[ch].fillna(value)
    return xrds


def run_ts_to_stft(
    decimation_obj: AuroraDecimationLevel, run_xrds_orig: xr.Dataset
) -> spectre.Spectrogram:
    """
    Converts a runts object into a time series of Fourier coefficients.
    Similar to run_ts_to_stft_scipy, but in this implementation operations on individual
    windows are possible (for example pre-whitening per time window via ARMA filtering).

    TODO: Make the output of this function a Spectrogram object

    Parameters
    ----------
    decimation_obj : AuroraDecimationLevel
        Information about how the decimation level is to be processed
    run_xrds_orig: xarray.core.dataset.Dataset
        normally extracted from mth5.RunTS

    Returns
    -------
    stft_obj: xarray.core.dataset.Dataset
        Note that the STFT object may have inf/nan in the DC harmonic, introduced by
        recoloring. This really doesn't matter since we don't use the DC harmonic for
        anything.
    """
    # need to remove any nans before windowing, or else if there is a single
    # nan then the whole channel becomes nan.
    run_xrds = nan_to_mean(run_xrds_orig)
    run_xrds = apply_prewhitening(decimation_obj.stft.prewhitening_type, run_xrds)
    run_xrds = truncate_to_clock_zero(decimation_obj, run_xrds)
    windowing_scheme = window_scheme_from_decimation(decimation_obj)
    windowed_obj = windowing_scheme.apply_sliding_window(
        run_xrds, dt=1.0 / decimation_obj.decimation.sample_rate
    )
    if not np.prod(windowed_obj.to_array().data.shape):
        raise ValueError

    windowed_obj = WindowedTimeSeries.detrend(data=windowed_obj, detrend_type="linear")
    tapered_obj = WindowedTimeSeries.apply_taper(
        data=windowed_obj, taper=windowing_scheme.taper
    )
    stft_obj = WindowedTimeSeries.apply_fft(
        data=tapered_obj,
        sample_rate=windowing_scheme.sample_rate,
        spectral_density_correction=windowing_scheme.linear_spectral_density_calibration_factor,
        detrend_type=decimation_obj.stft.per_window_detrend_type,
    )

    if decimation_obj.stft.recoloring:
        stft_obj = apply_recoloring(decimation_obj.stft.prewhitening_type, stft_obj)

    spectrogram = spectre.Spectrogram(dataset=stft_obj)
    return spectrogram


def calibrate_stft_obj(
    stft_obj: xr.Dataset,
    run_obj: RunGroup,
    units: Literal["MT", "SI"] = "MT",
    channel_scale_factors: Optional[dict] = None,
) -> xr.Dataset:
    """
    Calibrates frequency domain data into MT units.

    Development Notes:
     The calibration often raises a runtime warning due to DC term in calibration response = 0.
     TODO: It would be nice to suppress this, maybe by only calibrating the non-dc terms and directly assigning np.nan to the dc component when DC-response is zero.

    Parameters
    ----------
    stft_obj : xarray.core.dataset.Dataset
        Time series of Fourier coefficients to be calibrated
    run_obj : mth5.groups.master_station_run_channel.RunGroup
        Provides information about filters for calibration
    units : string
        usually "MT", contemplating supporting "SI"
    scale_factors : dict or None
        keyed by channel, supports a single scalar to apply to that channels data
        Useful for debugging.  Should not be used in production and should throw a
        warning if it is not None

    Returns
    -------
    stft_obj : xarray.core.dataset.Dataset
        Time series of calibrated Fourier coefficients
    """
    for channel_id in stft_obj.keys():

        channel = run_obj.get_channel(channel_id)
        channel_response = channel.channel_response
        if not channel_response.filters_list:
            msg = f"Channel {channel_id} with empty filters list detected"
            logger.warning(msg)
            if channel_id == "hy":
                msg = "Channel hy has no filters, try using filters from hx"
                logger.warning(msg)
                channel_response = run_obj.get_channel("hx").channel_response

        indices_to_flip = channel_response.get_indices_of_filters_to_remove(
            include_decimation=False, include_delay=False
        )
        indices_to_flip = [
            i for i in indices_to_flip if channel.metadata.filter.applied[i]
        ]
        filters_to_remove = [channel_response.filters_list[i] for i in indices_to_flip]
        if not filters_to_remove:
            logger.warning("No filters to remove")

        calibration_response = channel_response.complex_response(
            stft_obj.frequency.data, filters_list=filters_to_remove
        )

        if channel_scale_factors:
            try:
                channel_scale_factor = channel_scale_factors[channel_id]
            except KeyError:
                channel_scale_factor = 1.0
            calibration_response /= channel_scale_factor

        if units == "SI":
            logger.warning("Warning: SI Units are not robustly supported issue #36")

        # TODO: FIXME Sometimes raises a runtime warning due to DC term in calibration response = 0
        stft_obj[channel_id].data /= calibration_response
    return stft_obj


def prototype_decimate(
    ts_decimation: TimeSeriesDecimation,
    run_xrds: xr.Dataset,
) -> xr.Dataset:
    """
    Basically a wrapper for scipy.signal.decimate.  Takes input timeseries (as xarray
     Dataset) and a TimeSeriesDecimation object and returns a decimated version of the
     input time series.

    TODO: Consider moving this function into time_series/decimate.py
    TODO: Consider Replacing the downsampled_time_axis with rolling mean, or somthing that takes the average value of the time, not the window start
    TODO: Compare outputs with scipy resample_poly, which also has an FIR AAF and appears faster
    TODO: Add handling for case that could occur when sliced time axis has a different length than the decimated data -- see mth5 issue #217 https://github.com/kujaku11/mth5/issues/217

    Parameters
    ----------
    ts_decimation : AuroraDecimationLevel
    run_xrds: xr.Dataset
        Originally from mth5.timeseries.run_ts.RunTS.dataset, but possibly decimated
        multiple times

    Returns
    -------
    xr_ds: xr.Dataset
        Decimated version of the input run_xrds
    """
    # downsample the time axis
    slicer = slice(None, None, int(ts_decimation.factor))  # decimation.factor
    downsampled_time_axis = run_xrds.time.data[slicer]

    # decimate the time series
    num_observations = len(downsampled_time_axis)
    channel_labels = list(run_xrds.data_vars.keys())  # run_ts.channels
    num_channels = len(channel_labels)
    new_data = np.full((num_observations, num_channels), np.nan)
    for i_ch, ch_label in enumerate(channel_labels):
        # TODO: add check here for ts_decimation.anti_alias_filter
        new_data[:, i_ch] = ssig.decimate(run_xrds[ch_label], int(ts_decimation.factor))

    xr_da = xr.DataArray(
        new_data,
        dims=["time", "channel"],
        coords={"time": downsampled_time_axis, "channel": channel_labels},
    )
    attr_dict = run_xrds.attrs
    attr_dict["sample_rate"] = ts_decimation.sample_rate
    xr_da.attrs = attr_dict
    xr_ds = xr_da.to_dataset("channel")
    return xr_ds


# Here are some other ways to downsample/decimate that were experimented with.
# TODO: Jupyter notebook tutorial showing how these differ (cf standard method).
# def prototype_decimate_2(config, run_xrds):
#     """
#     Uses the built-in xarray coarsen method.   Not clear what AAF effects are.
#     Method is fast.  Might be non-linear.  Seems to give similar performance to
#     prototype_decimate for synthetic data.
#
#     N.B. config.decimation.factor must be integer valued
#
#     Parameters
#     ----------
#     config : mt_metadata.transfer_functions.processing.aurora.Decimation
#     run_xrds: xr.Dataset
#         Originally from mth5.timeseries.run_ts.RunTS.dataset, but possibly decimated
#         multiple times
#
#     Returns
#     -------
#     xr_ds: xr.Dataset
#         Decimated version of the input run_xrds
#     """
#     new_xr_ds = run_xrds.coarsen(time=int(config.decimation.factor), boundary="trim").mean()
#     attr_dict = run_xrds.attrs
#     attr_dict["sample_rate"] = config.sample_rate
#     new_xr_ds.attrs = attr_dict
#     return new_xr_ds
#
#
# def prototype_decimate_3(config, run_xrds):
#     """
#     Uses scipy's resample method.   Not clear what AAF effects are.
#     Method is fast.
#
#     Parameters
#     ----------
#     config : mt_metadata.transfer_functions.processing.aurora.Decimation
#     run_xrds: xr.Dataset
#         Originally from mth5.timeseries.run_ts.RunTS.dataset, but possibly decimated
#         multiple times
#
#     Returns
#     -------
#     xr_ds: xr.Dataset
#         Decimated version of the input run_xrds
#     """
#     dt = run_xrds.time.diff(dim="time").median().values
#     dt_new = config.decimation.factor * dt
#     dt_new = dt_new.__str__().replace("nanoseconds", "ns")
#     new_xr_ds = run_xrds.resample(time=dt_new).mean(dim="time")
#     attr_dict = run_xrds.attrs
#     attr_dict["sample_rate"] = config.sample_rate
#     new_xr_ds.attrs = attr_dict
#     return new_xr_ds
