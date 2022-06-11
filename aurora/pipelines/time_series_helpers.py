import numpy as np
import scipy.signal as ssig
import xarray as xr

from aurora.time_series.frequency_domain_helpers import get_fft_harmonics
from aurora.time_series.windowed_time_series import WindowedTimeSeries
from aurora.time_series.windowing_scheme import WindowingScheme

def validate_sample_rate(run_ts, expected_sample_rate):
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
            f"processing decimation_obj {decimation_obj.sample_rate} do not match"
        )
        raise Exception


def apply_prewhitening(decimation_obj, run_xrts_input):
    """

    Parameters
    ----------
    decimation_obj : aurora.config.metadata.decimation_level.DecimationLevel
        Information about how the decimation level is to be processed
    run_xrts_input : xarray.core.dataset.Dataset
        Time series to be prewhitened

    Returns
    -------
    run_xrts : xarray.core.dataset.Dataset
        prewhitened time series

    """
    if decimation_obj.prewhitening_type == "first difference":
        run_xrts = run_xrts_input.diff("time")
    else:
        print(f"{decimation_obj.prewhitening_type} prehitening not yet implemented")
        print(f"returning original time series")
        run_xrts = run_xrts_input
    return run_xrts


def apply_recoloring(decimation_obj, stft_obj):
    """
    Parameters
    ----------
    decimation_obj : aurora.config.metadata.decimation_level.DecimationLevel
        Information about how the decimation level is to be processed
    stft_obj : xarray.core.dataset.Dataset
        Time series of Fourier coefficients to be recoloured


    Returns
    -------
    stft_obj : xarray.core.dataset.Dataset
        Recolored time series of Fourier coefficients
    """
    if decimation_obj.prewhitening_type == "first difference":
        freqs = get_fft_harmonics(decimation_obj.window.num_samples, 
                                  decimation_obj.decimation.sample_rate)
        prewhitening_correction = 1.0j * 2 * np.pi * freqs  # jw
        stft_obj /= prewhitening_correction
    return stft_obj


def run_ts_to_stft_scipy(decimation_obj, run_xrts_orig):
    """
    Parameters
    ----------
    decimation_obj : aurora.config.metadata.decimation_level.DecimationLevel
        Information about how the decimation level is to be processed
    run_xrts : : xarray.core.dataset.Dataset
        Time series to be processed

    Returns
    -------
    stft_obj : xarray.core.dataset.Dataset
        Time series of Fourier coefficients
    """
    run_xrts = apply_prewhitening(decimation_obj, run_xrts_orig)

    windowing_scheme = WindowingScheme(
        taper_family=decimation_obj.window.type,
        num_samples_window=decimation_obj.window.num_samples,
        num_samples_overlap=decimation_obj.window.overlap,
        taper_additional_args=decimation_obj.window.additional_args,
        sample_rate=decimation_obj.decimation.sample_rate,
    )

    stft_obj = xr.Dataset()
    for channel_id in run_xrts.data_vars:
        ff, tt, specgm = ssig.spectrogram(
            run_xrts[channel_id].data,
            fs=decimation_obj.decimation.sample_rate,
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
        tt *= decimation_obj.decimation.sample_rate
        time_axis = run_xrts.time.data[tt.astype(int)]

        xrd = xr.DataArray(
            specgm.T,
            dims=["time", "frequency"],
            coords={"frequency": ff, "time": time_axis},
        )
        stft_obj.update({channel_id: xrd})

    stft_obj = apply_recoloring(decimation_obj, stft_obj)

    return stft_obj


def run_ts_to_stft(decimation_obj, run_xrts_orig):
    """

    Parameters
    ----------
    decimation_obj : aurora.config.metadata.decimation_level.DecimationLevel
        Information about how the decimation level is to be processed
    run_ts ; xarray.core.dataset.Dataset
        normally extracted from mth5.RunTS

    Returns
    -------
    stft_obj: xarray.core.dataset.Dataset
        Note that the STFT object may have inf/nan in the DC harmonic, introduced by
        recoloring. This really doesn't matter since we don't use the DC harmonic for
        anything.
    """
    try:
        windowing_scheme = WindowingScheme(
            taper_family=decimation_obj.window.type,
            num_samples_window=decimation_obj.window.num_samples,
            num_samples_overlap=decimation_obj.window.overlap,
            taper_additional_args=decimation_obj.window.additional_args,
            sample_rate=decimation_obj.decimation.sample_rate,
        )
    except AttributeError:
        print("AttributeError --- run_ts_to_stft ?")


    run_xrts = apply_prewhitening(decimation_obj, run_xrts_orig)

    #optionally clip data based on clock zero
    if decimation_obj.window.clock_zero_type == "ignore":
        pass #ignore clock zero
    elif decimation_obj.window.clock_zero_type == "data zero":
        raise NotImplementedError
    elif decimation_obj.window.clock_zero_type == "user defined":
        raise NotImplementedError
        #clock_zero = decimation_obj.window.clock_zero

    windowed_obj = windowing_scheme.apply_sliding_window(
        run_xrts, dt=1.0 / decimation_obj.decimation.sample_rate
    )
    windowed_obj = WindowedTimeSeries.detrend(data=windowed_obj, detrend_type="linear")

    tapered_obj = windowed_obj * windowing_scheme.taper
    # stft_obj = WindowedTimeSeries.apply_stft(data=tapered_obj,
    #                                          sample_rate=windowing_scheme.sample_rate,
    #                                          detrend_type="linear",
    # scale_factor=windowing_scheme.linear_spectral_density_calibration_factor)

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


def get_run_run_ts_from_mth5(mth5_obj, station_id, run_id, expected_sample_rate,
                             start=None, end=None):
    """
    ToDo: Review if this method should be moved into mth5.

    Simple implementation of what could eventually become a somewhat complicated
    function -- see issue #13.  In a future version this could also take a decimation
    level as an argument.  It could then be merged with prototype decimate, depending
    on the decimation level.

    Future version should be able to
    1. accept a config and an mth5_obj and return decimation_level_0,
    2. Accept data from a given decimation level, and decimation
    instrucntions and return it
    3. If we decide to house decimated data in an mth5 should return time
    series for the run at the perscribed decimation level

    Thus args would be
    decimation_level_config, mth5,
    decimation_level_config, runs and run_ts'
    decimation_level_config, mth5

    Parameters
    ----------
    mth5_obj: mth5.mth5.MTH5
        The data container with run and time series'
    station_id: str
        The name of the station to get data from
    run_id: str
        The name of the run to get data from
    sample_rate : float (may choose to also support  None)
        expected sample rate of data in the mth5

    Returns
    -------
    run_run_ts : dict
        Dictionary associated with a station-run. Has keys "run" and "mvts".
        "run" maps to mth5.groups.master_station_run_channel.RunGroup
        "mvts" maps to xarray.core.dataset.Dataset

    """
    run_obj = mth5_obj.get_run(station_id, run_id)
    run_ts = run_obj.to_runts(start=start, end=end)
    validate_sample_rate(run_ts, expected_sample_rate)
    run_run_ts = {"run": run_obj, "mvts": run_ts.dataset}
    return run_run_ts


def prototype_decimate(config, run_run_ts):
    """
    TODO: ?Move this function into time_series/decimate.py?
    Parameters
    ----------
    config : aurora.config.metadata.decimation.Decimation
    run_run_ts: dict keyed by "run" and "mvts"
    out_dict["run"] is mth5.groups.master_station_run_channel.RunGroup
    out_dict["mvts"] is mth5.timeseries.run_ts.RunTS

    Returns
    -------
    dict: same structure as run_run_ts
    """
    run_obj = run_run_ts["run"]
    run_xrts = run_run_ts["mvts"]
    run_obj.metadata.sample_rate = config.sample_rate

    # <Replace with rolling mean, somethng that works with time>
    # and preferably takes the average time, not the start of the window
    slicer = slice(None, None, int(config.factor))#decimation.factor
    downsampled_time_axis = run_xrts.time.data[slicer]
    # </Replace with rolling mean, somethng that works with time>

    num_observations = len(downsampled_time_axis)
    channel_labels = list(run_xrts.data_vars.keys())  # run_ts.channels
    num_channels = len(channel_labels)
    new_data = np.full((num_observations, num_channels), np.nan)
    for i_ch, ch_label in enumerate(channel_labels):
        new_data[:, i_ch] = ssig.decimate(run_xrts[ch_label], int(config.factor))

    xr_da = xr.DataArray(
        new_data,
        dims=["time", "channel"],
        coords={"time": downsampled_time_axis, "channel": channel_labels},
    )

    xr_ds = xr_da.to_dataset("channel")
    result = {"run": run_obj, "mvts": xr_ds}

    return result
