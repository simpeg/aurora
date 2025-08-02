"""
    Collection of modules used in processing pipeline that operate on time series
"""
import mt_metadata
import numpy as np
import pandas as pd
import scipy.signal as ssig
import xarray as xr

from loguru import logger
from aurora.time_series.windowing_scheme import window_scheme_from_decimation
from mt_metadata.transfer_functions.processing import TimeSeriesDecimation
from mt_metadata.transfer_functions.processing.aurora.decimation_level import (
    DecimationLevel as AuroraDecimationLevel,
)
from mt_metadata.transfer_functions.processing.fourier_coefficients import (
    Decimation as FCDecimation,
)
from mth5.groups import RunGroup
from typing import Union


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
