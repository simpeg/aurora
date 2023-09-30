import numpy as np
import pandas as pd
import scipy.signal as ssig
import xarray as xr

from aurora.time_series.windowed_time_series import WindowedTimeSeries
from aurora.time_series.windowing_scheme import window_scheme_from_decimation

def validate_sample_rate(run_ts, expected_sample_rate, tol=1e-4):
    """

    Parameters
    ----------
    run_ts: mth5.timeseries.run_ts.RunTS
        Time series object
    expected_sample_rate: float
        The samepling rate the time series is expected to have. Normally taken from
        the processing config

    """
    if run_ts.sample_rate != expected_sample_rate:
        print(
            f"sample rate in run time series {run_ts.sample_rate} and "
            f"processing decimation_obj {expected_sample_rate} do not match"
        )
        delta = run_ts.sample_rate - expected_sample_rate
        if np.abs(delta) > tol:
            print(f"Delta sample rate {delta} > {tol} tolerance")
            print("TOL should be a percentage")
            raise Exception


def apply_prewhitening(decimation_obj, run_xrds_input):
    """
    Applys prewhitening to time series to avoid spectral leakage when FFT is applied.

    If "first difference", may want to consider clipping first and last sample from
    the differentiated time series.

    Parameters
    ----------
    decimation_obj : mt_metadata.transfer_functions.processing.aurora.DecimationLevel
        Information about how the decimation level is to be processed
    run_xrds_input : xarray.core.dataset.Dataset
        Time series to be prewhitened

    Returns
    -------
    run_xrds : xarray.core.dataset.Dataset
        prewhitened time series

    """
    if not decimation_obj.prewhitening_type:
        return run_xrds_input

    if decimation_obj.prewhitening_type == "first difference":
        run_xrds = run_xrds_input.differentiate("time")

    else:
        print(f"{decimation_obj.prewhitening_type} pre-whitening not implemented")
        raise NotImplementedError
    return run_xrds


def apply_recoloring(decimation_obj, stft_obj):
    """
    Parameters
    ----------
    decimation_obj : mt_metadata.transfer_functions.processing.fourier_coefficients.decimation.Decimation
        Information about how the decimation level is to be processed
    stft_obj : xarray.core.dataset.Dataset
        Time series of Fourier coefficients to be recoloured


    Returns
    -------
    stft_obj : xarray.core.dataset.Dataset
        Recolored time series of Fourier coefficients
    """
    # No recoloring needed if prewhitening not appiled, or recoloring set to False
    if not decimation_obj.prewhitening_type:
        return stft_obj
    if not decimation_obj.recoloring:
        return stft_obj

    if decimation_obj.prewhitening_type == "first difference":
        freqs = decimation_obj.fft_frequencies
        prewhitening_correction = 1.0j * 2 * np.pi * freqs  # jw

        stft_obj /= prewhitening_correction

        # suppress nan and inf to mute later warnings
        if prewhitening_correction[0] == 0.0:
            cond = stft_obj.frequency != 0.0
            stft_obj = stft_obj.where(cond, complex(0.0))
    # elif decimation_obj.prewhitening_type == "ARMA":
    #     from statsmodels.tsa.arima.model import ARIMA
    #     AR = 3 # add this to processing config
    #     MA = 4 # add this to processing config

    else:
        print(f"{decimation_obj.prewhitening_type} recoloring not yet implemented")
        raise NotImplementedError

    return stft_obj


def run_ts_to_stft_scipy(decimation_obj, run_xrds_orig):
    """
    Parameters
    ----------
    decimation_obj : mt_metadata.transfer_functions.processing.aurora.DecimationLevel
        Information about how the decimation level is to be processed
    run_xrds_orig : : xarray.core.dataset.Dataset
        Time series to be processed

    Returns
    -------
    stft_obj : xarray.core.dataset.Dataset
        Time series of Fourier coefficients
    """
    run_xrds = apply_prewhitening(decimation_obj, run_xrds_orig)
    windowing_scheme = window_scheme_from_decimation(decimation_obj)

    stft_obj = xr.Dataset()
    for channel_id in run_xrds.data_vars:
        ff, tt, specgm = ssig.spectrogram(
            run_xrds[channel_id].data,
            fs=decimation_obj.sample_rate_decimation,
            window=windowing_scheme.taper,
            nperseg=decimation_obj.window.num_samples,
            noverlap=decimation_obj.window.overlap,
            detrend="linear",
            scaling="density",
            mode="complex",
        )

        # drop Nyquist>
        ff = ff[:-1]
        specgm = specgm[:-1, :]
        specgm *= np.sqrt(2)

        # make time_axis
        tt = tt - tt[0]
        tt *= decimation_obj.sample_rate_decimation
        time_axis = run_xrds.time.data[tt.astype(int)]

        xrd = xr.DataArray(
            specgm.T,
            dims=["time", "frequency"],
            coords={"frequency": ff, "time": time_axis},
        )
        stft_obj.update({channel_id: xrd})

    stft_obj = apply_recoloring(decimation_obj, stft_obj)

    return stft_obj


def truncate_to_clock_zero(decimation_obj, run_xrds):
    """
    Compute the time interval between the first data sample and the clockzero
    Identify the first sample in the xarray time series that corresponds to a
    window start sample.

    Parameters
    ----------
    decimation_obj: mt_metadata.transfer_functions.processing.aurora.DecimationLevel
        Information about how the decimation level is to be processed
    run_xrds : xarray.core.dataset.Dataset
        normally extracted from mth5.RunTS


    Returns
    -------
    run_xrds : xarray.core.dataset.Dataset
        same as the input time series, but possibly slightly shortened
    """
    if decimation_obj.window.clock_zero_type == "ignore":
        pass
    else:
        clock_zero = pd.Timestamp(decimation_obj.window.clock_zero)
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
            print(
                f"dropping {n_clip} samples to agree with "
                f"{decimation_obj.window.clock_zero_type} clock zero {clock_zero}"
            )
            run_xrds = run_xrds.where(cond1, drop=True)
    return run_xrds


def nan_to_mean(xrds):
    """
    Set Nan values to mean value

    :param xrds: DESCRIPTION
    :type xrds: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    for ch in xrds.keys():
        null_values_present = xrds[ch].isnull().any()
        if null_values_present:
            print(
                "Null values detected in xrds -- this is not expected and should be examined"
            )
            value = np.nan_to_num(np.nanmean(xrds[ch].data))
            xrds[ch] = xrds[ch].fillna(value)
    return xrds


def run_ts_to_stft(decimation_obj, run_xrds_orig):
    """

    Parameters
    ----------
    decimation_obj : mt_metadata.transfer_functions.processing.aurora.DecimationLevel
        Information about how the decimation level is to be processed
    run_ts : xarray.core.dataset.Dataset
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
    run_xrds = apply_prewhitening(decimation_obj, run_xrds_orig)
    run_xrds = truncate_to_clock_zero(decimation_obj, run_xrds)
    windowing_scheme = window_scheme_from_decimation(decimation_obj)
    windowed_obj = windowing_scheme.apply_sliding_window(
        run_xrds, dt=1.0 / decimation_obj.sample_rate_decimation
    )
    if not np.prod(windowed_obj.to_array().data.shape):
        raise ValueError

    windowed_obj = WindowedTimeSeries.detrend(data=windowed_obj, detrend_type="linear")
    tapered_obj = windowed_obj * windowing_scheme.taper
    stft_obj = windowing_scheme.apply_fft(
        tapered_obj, detrend_type=decimation_obj.extra_pre_fft_detrend_type
    )
    stft_obj = apply_recoloring(decimation_obj, stft_obj)

    return stft_obj


def calibrate_stft_obj(stft_obj, run_obj, units="MT", channel_scale_factors=None):
    """

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
        mth5_channel = run_obj.get_channel(channel_id)
        channel_filter = mth5_channel.channel_response_filter
        if not channel_filter.filters_list:
            print("WARNING UNEXPECTED CHANNEL WITH NO FILTERS")
            if channel_id == "hy":
                print("Channel HY has no filters, try using filters from HX")
                channel_filter = run_obj.get_channel("hx").channel_response_filter
        calibration_response = channel_filter.complex_response(stft_obj.frequency.data)
        if channel_scale_factors:
            try:
                channel_scale_factor = channel_scale_factors[channel_id]
            except KeyError:
                channel_scale_factor = 1.0
            calibration_response /= channel_scale_factor
        if units == "SI":
            print("Warning: SI Units are not robustly supported issue #36")

        stft_obj[channel_id].data /= calibration_response
    return stft_obj


def prototype_decimate(config, run_xrds):
    """
    Consider:
    1. Moving this function into time_series/decimate.py
    2. Replacing the downsampled_time_axis with rolling mean, or somthing that takes
    the average value of the time, not the window start

    Parameters
    ----------
    config : mt_metadata.transfer_functions.processing.aurora.Decimation
    run_xrds: xr.Dataset
        Originally from mth5.timeseries.run_ts.RunTS.dataset, but possibly decimated
        multiple times

    Returns
    -------
    xr_ds: xr.Dataset
        Decimated version of the input run_xrds
    """
    slicer = slice(None, None, int(config.factor))  # decimation.factor
    downsampled_time_axis = run_xrds.time.data[slicer]

    num_observations = len(downsampled_time_axis)
    channel_labels = list(run_xrds.data_vars.keys())  # run_ts.channels
    num_channels = len(channel_labels)
    new_data = np.full((num_observations, num_channels), np.nan)
    for i_ch, ch_label in enumerate(channel_labels):
        new_data[:, i_ch] = ssig.decimate(run_xrds[ch_label], int(config.factor))

    xr_da = xr.DataArray(
        new_data,
        dims=["time", "channel"],
        coords={"time": downsampled_time_axis, "channel": channel_labels},
    )
    attr_dict = run_xrds.attrs
    attr_dict["sample_rate"] = config.sample_rate
    xr_da.attrs = attr_dict
    xr_ds = xr_da.to_dataset("channel")
    return xr_ds


def prototype_decimate_2(config, run_xrds):
    """
    Uses the built-in xarray coarsen method.   Not clear what AAF effects are.
    Method is fast.  Might be non-linear.  Seems to give similar performance to
    prototype_decimate for synthetic data.

    N.B. config.factor must be integer valued

    Parameters
    ----------
    config : mt_metadata.transfer_functions.processing.aurora.Decimation
    run_xrds: xr.Dataset
        Originally from mth5.timeseries.run_ts.RunTS.dataset, but possibly decimated
        multiple times

    Returns
    -------
    xr_ds: xr.Dataset
        Decimated version of the input run_xrds
    """
    new_xr_ds = run_xrds.coarsen(time=int(config.factor), boundary="trim").mean()
    attr_dict = run_xrds.attrs
    attr_dict["sample_rate"] = config.sample_rate
    new_xr_ds.attrs = attr_dict
    return new_xr_ds


def prototype_decimate_3(config, run_xrds):
    """
    Uses the built-in xarray coarsen method.   Not clear what AAF effects are.
    Method is fast.  Might be non-linear.  Seems to give similar performance to
    prototype_decimate for synthetic data.

    N.B. config.factor must be integer valued

    Parameters
    ----------
    config : mt_metadata.transfer_functions.processing.aurora.Decimation
    run_xrds: xr.Dataset
        Originally from mth5.timeseries.run_ts.RunTS.dataset, but possibly decimated
        multiple times

    Returns
    -------
    xr_ds: xr.Dataset
        Decimated version of the input run_xrds
    """
    dt = run_xrds.time.diff(dim="time").median().values
    dt_new = config.factor * dt
    dt_new = dt_new.__str__().replace("nanoseconds", "ns")
    new_xr_ds = run_xrds.resample(time=dt_new).mean(dim="time")
    attr_dict = run_xrds.attrs
    attr_dict["sample_rate"] = config.sample_rate
    new_xr_ds.attrs = attr_dict
    return new_xr_ds
