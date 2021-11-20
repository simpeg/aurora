import scipy.signal as ssig

from aurora.time_series.windowing_scheme import WindowingScheme


def validate_sample_rate(run_ts, config):
    if run_ts.sample_rate != config.sample_rate:
        print(
            f"sample rate in run time series {run_ts.sample_rate} and "
            f"processing config {config.sample_rate} do not match"
        )
        raise Exception
    return


def apply_prewhitening(config, run_xrts_input):
    if config["prewhitening_type"] == "first difference":
        run_xrts = run_xrts_input.diff("time")
    else:
        run_xrts = run_xrts_input
    return run_xrts


def apply_recoloring(config, stft_obj):
    if config["prewhitening_type"] == "first difference":
        from aurora.time_series.frequency_domain_helpers import get_fft_harmonics
        from numpy import pi

        freqs = get_fft_harmonics(config.num_samples_window, config.sample_rate)
        prewhitening_correction = 1.0j * 2 * pi * freqs  # jw
        stft_obj /= prewhitening_correction
    return stft_obj


def run_ts_to_stft_scipy(config, run_xrts_orig):
    """
    Parameters
    ----------
    config
    run_xrts

    Returns
    -------

    """
    import xarray as xr

    run_xrts = apply_prewhitening(config, run_xrts_orig)

    windowing_scheme = WindowingScheme(
        taper_family=config.taper_family,
        num_samples_window=config.num_samples_window,
        num_samples_overlap=config.num_samples_overlap,
        taper_additional_args=config.taper_additional_args,
        sample_rate=config.sample_rate,
    )
    # stft_obj = run_xrts.copy(deep=True)
    stft_obj = xr.Dataset()
    for channel_id in run_xrts.data_vars:
        ff, tt, specgm = ssig.spectrogram(
            run_xrts[channel_id].data,
            fs=config.sample_rate,
            window=windowing_scheme.taper,
            nperseg=config.num_samples_window,
            noverlap=config.num_samples_overlap,
            detrend="linear",
            scaling="density",
            mode="complex",
        )

        # drop Nyquist>
        ff = ff[:-1]
        specgm = specgm[:-1, :]

        import numpy as np

        specgm *= np.sqrt(2)

        # make time_axis
        tt = tt - tt[0]
        tt *= config.sample_rate
        time_axis = run_xrts.time.data[tt.astype(int)]

        xrd = xr.DataArray(
            specgm.T,
            dims=["time", "frequency"],
            coords={"frequency": ff, "time": time_axis},
        )
        stft_obj.update({channel_id: xrd})

    stft_obj = apply_recoloring(config, stft_obj)

    return stft_obj


def run_ts_to_stft(config, run_xrts_orig):
    """

    Parameters
    ----------
    config : ShortTimeFourierTransformConfig object
    run_ts ; xarray.core.dataset.Dataset, normally extracted from mth5.RunTS

    Returns
    -------

    """
    from aurora.time_series.windowed_time_series import WindowedTimeSeries

    windowing_scheme = WindowingScheme(
        taper_family=config.taper_family,
        num_samples_window=config.num_samples_window,
        num_samples_overlap=config.num_samples_overlap,
        taper_additional_args=config.taper_additional_args,
        sample_rate=config.sample_rate,
    )

    run_xrts = apply_prewhitening(config, run_xrts_orig)

    windowed_obj = windowing_scheme.apply_sliding_window(
        run_xrts, dt=1.0 / config.sample_rate
    )
    windowed_obj = WindowedTimeSeries.detrend(data=windowed_obj, detrend_type="linear")

    tapered_obj = windowed_obj * windowing_scheme.taper
    # stft_obj = WindowedTimeSeries.apply_stft(data=tapered_obj,
    #                                          sample_rate=windowing_scheme.sample_rate,
    #                                          detrend_type="linear",
    # scale_factor=windowing_scheme.linear_spectral_density_calibration_factor)

    stft_obj = windowing_scheme.apply_fft(
        tapered_obj, detrend_type=config.extra_pre_fft_detrend_type
    )
    stft_obj = apply_recoloring(config, stft_obj)

    return stft_obj


def run_ts_to_calibrated_stft(run_ts, run_obj, config, units="MT"):
    """
    Parameters
    ----------
    run_ts
    run_obj
    config
    units

    Returns
    -------

    """
    stft_obj = run_ts_to_stft(config, run_ts.dataset)
    stft_obj = calibrate_stft_obj(stft_obj, run_obj, units=units)

    return stft_obj


def calibrate_stft_obj(stft_obj, run_obj, units="MT", channel_scale_factors=None):
    """

    Parameters
    ----------
    stft_obj
    run_obj
    units
    scale_factors : dict
        keyed by channel, supports a single scalar to apply to that channels data
        Useful for debugging.  Should not be used in production and should throw a
        warning if it is not None

    Returns
    -------

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
