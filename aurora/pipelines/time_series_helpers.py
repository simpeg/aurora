from deprecated import deprecated
import numpy as np
import pandas as pd
import scipy.signal as ssig
import xarray as xr

from aurora.time_series.frequency_domain_helpers import get_fft_harmonics
from aurora.time_series.windowed_time_series import WindowedTimeSeries


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


def apply_prewhitening(decimation_obj, run_xrts_input):
    """
    Applys prewhitening to time series to avoid spectral leakage when FFT is applied.

    If "first difference", may want to consider clipping first and last sample from
    the differentiated time series.

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
        run_xrts = run_xrts_input.differentiate("time")
    else:
        print(f"{decimation_obj.prewhitening_type} prehitening not yet implemented")
        raise NotImplementedError
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
        # replace below with decimation_obj.get_fft_harmonics() ?
        freqs = get_fft_harmonics(
            decimation_obj.window.num_samples, decimation_obj.decimation.sample_rate
        )
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
    windowing_scheme = decimation_obj.windowing_scheme

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


def truncate_to_clock_zero(decimation_obj, run_xrts):
    """
    Compute the time interval between the first data sample and the clockzero
    Identify the first sample in the xarray time series that corresponds to a
    window start sample.

    Parameters
    ----------
    decimation_obj: aurora.config.metadata.decimation_level.DecimationLevel
        Information about how the decimation level is to be processed
    run_xrts : xarray.core.dataset.Dataset
        normally extracted from mth5.RunTS


    Returns
    -------
    run_xrts : xarray.core.dataset.Dataset
        same as the input time series, but possibly slightly shortened
    """
    if decimation_obj.window.clock_zero_type == "ignore":
        pass
    else:
        clock_zero = pd.Timestamp(decimation_obj.window.clock_zero)
        clock_zero = clock_zero.to_datetime64()
        delta_t = clock_zero - run_xrts.time[0]
        assert delta_t.dtype == "<m8[ns]"  # expected in nanoseconds
        delta_t_seconds = int(delta_t) / 1e9
        if delta_t_seconds == 0:
            pass  # time series start is already clock zero
        else:
            windowing_scheme = decimation_obj.windowing_scheme
            number_of_steps = delta_t_seconds / windowing_scheme.duration_advance
            n_partial_steps = number_of_steps - np.floor(number_of_steps)
            n_clip = n_partial_steps * windowing_scheme.num_samples_advance
            n_clip = int(np.round(n_clip))
            t_clip = run_xrts.time[n_clip]
            cond1 = run_xrts.time >= t_clip
            print(
                f"dropping {n_clip} samples to agree with "
                f"{decimation_obj.window.clock_zero_type} clock zero {clock_zero}"
            )
            run_xrts = run_xrts.where(cond1, drop=True)
    return run_xrts


def run_ts_to_stft(decimation_obj, run_xrts_orig):
    """

    Parameters
    ----------
    decimation_obj : aurora.config.metadata.decimation_level.DecimationLevel
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
    run_xrts = apply_prewhitening(decimation_obj, run_xrts_orig)
    run_xrts = truncate_to_clock_zero(decimation_obj, run_xrts)
    windowing_scheme = decimation_obj.windowing_scheme
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


@deprecated
def get_run_run_ts_from_mth5(
    mth5_obj,
    station_id,
    run_id,
    expected_sample_rate,
    start=None,
    end=None,
    survey=None,
):
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
    run_obj = mth5_obj.get_run(station_id, run_id, survey=survey)
    run_ts = run_obj.to_runts(start=start, end=end)
    validate_sample_rate(run_ts, expected_sample_rate)
    run_run_ts = {"run": run_obj, "mvts": run_ts.dataset}
    return run_run_ts


def prototype_decimate(config, run_xrts):
    """
    Consider:
    1. Moving this function into time_series/decimate.py
    2. Replacing the downsampled_time_axis with rolling mean, or somthing that takes
    the average value of the time, not the window start

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
    slicer = slice(None, None, int(config.factor))  # decimation.factor
    downsampled_time_axis = run_xrts.time.data[slicer]

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
    attr_dict = run_xrts.attrs
    attr_dict["sample_rate"] = config.sample_rate
    xr_da.attrs = attr_dict
    print("DONT FORGET TO RESET THE SAMPLE RATE")
    print("DONT FORGET TO RESET THE SAMPLE RATE")
    print("!!!Sort of correct usage of sample_rate and decimated_sample_rate also!!!")
    print("XARRAY RESAMPLE")
    xr_ds = xr_da.to_dataset("channel")
    return xr_ds
