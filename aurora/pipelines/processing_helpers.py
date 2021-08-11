"""
Split this into a module for time_domain processing helpers,
and transfer_function_processing helpers.


"""

from aurora.time_series.frequency_band_helpers import extract_band
from aurora.time_series.frequency_band import FrequencyBands
from aurora.time_series.windowing_scheme import WindowingScheme
from aurora.transfer_function.iter_control import IterControl
from aurora.transfer_function.transfer_function_header import \
    TransferFunctionHeader
from aurora.transfer_function.regression.TRME import TRME
from aurora.transfer_function.regression.TRME_RR import TRME_RR

import scipy.signal as ssig



#<TIME SERIES PIPELINE HELPERS>
def validate_sample_rate(run_ts, config):
    if run_ts.sample_rate != config.sample_rate:
        print(f"sample rate in run time series {local_run_ts.sample_rate} and "
              f"processing config {config.sample_rate} do not match")
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
        from aurora.time_series.frequency_domain_helpers import \
            get_fft_harmonics
        from numpy import pi
        freqs = get_fft_harmonics(config.num_samples_window, config.sample_rate)
        prewhitening_correction = 1.j * 2 * pi * freqs #jw
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
        sampling_rate=config.sample_rate)
    #stft_obj = run_xrts.copy(deep=True)
    stft_obj = xr.Dataset()
    for channel_id in run_xrts.data_vars:
        ff, tt, specgm = ssig.spectrogram(run_xrts[channel_id].data,
                                        fs=config.sample_rate,
                                  window=windowing_scheme.taper,
                                  nperseg=config.num_samples_window,
                                  noverlap=config.num_samples_overlap,
                                  detrend="linear", scaling='density',
                                  mode="complex")

        #drop Nyquist>
        ff = ff[:-1]
        specgm = specgm[:-1,:]

        import numpy as np
        specgm *= np.sqrt(2)

        #make time_axis
        tt = tt - tt[0]
        tt *= config.sample_rate
        time_axis = run_xrts.time.data[tt.astype(int)]

        xrd = xr.DataArray(specgm.T, dims=["time", "frequency"],
                           coords={"frequency": ff,
                                   "time": time_axis})
        stft_obj.update({channel_id: xrd})

    stft_obj = apply_recoloring(config, stft_obj)
    
    return  stft_obj

def run_ts_to_stft(config, run_xrts_orig):
    """

    Parameters
    ----------
    config : ShortTimeFourierTransformConfig object
    run_ts ; mth5.RunTS (but could be replaced by the xr.dataset....)

    Returns
    -------

    """
    from aurora.time_series.windowed_time_series import WindowedTimeSeries
    windowing_scheme = WindowingScheme(
    taper_family = config.taper_family,
    num_samples_window = config.num_samples_window,
    num_samples_overlap = config.num_samples_overlap,
    taper_additional_args=config.taper_additional_args,
    sampling_rate = config.sample_rate)

    run_xrts = apply_prewhitening(config, run_xrts_orig)

    windowed_obj = windowing_scheme.apply_sliding_window(run_xrts)
    windowed_obj = WindowedTimeSeries.detrend(data=windowed_obj,
                                              detrend_type="linear")

    tapered_obj = windowed_obj * windowing_scheme.taper
    # stft_obj = WindowedTimeSeries.apply_stft(data=tapered_obj,
    #                                          sampling_rate=windowing_scheme.sampling_rate,
    #                                          detrend_type="linear",
    # scale_factor=windowing_scheme.linear_spectral_density_calibration_factor)

    stft_obj = windowing_scheme.apply_fft(tapered_obj,
                                          detrend_type=config.extra_pre_fft_detrend_type)
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

def calibrate_stft_obj(stft_obj, run_obj, units="MT"):
    """

    Parameters
    ----------
    stft_obj
    run_obj
    units

    Returns
    -------

    """
    for channel_id in stft_obj.keys():
        mth5_channel = run_obj.get_channel(channel_id)
        channel_filter = mth5_channel.channel_response_filter
        calibration_response = channel_filter.complex_response(
            stft_obj.frequency.data)

        if units == "SI":
            print("Warning: SI Units are not robustly supported issue #36")
            #This is not robust, and is really only here for the parkfield test
            #We should add units support as a general fix and handle the
            # parkfield case by converting to "MT" units in calibration filters
            if channel_id[0].lower() == 'h':
                calibration_response /= 1e-9  # SI Units
        stft_obj[channel_id].data /= calibration_response
    return stft_obj
#</TIME SERIES PIPELINE HELPERS>




#<TF PROCESSING HELPERS>
#TODO: Make all these regression methods accept kwargs on init so that
#none of them choke if we pass them Z=None or iter_control=None
from aurora.transfer_function.regression.base import RegressionEstimator
REGRESSION_LIBRARY = {
    "OLS" : RegressionEstimator,
    "RME" : TRME,
    "TRME_RR": TRME_RR
}

def get_regression_class(config):
    try:
        regression_class = REGRESSION_LIBRARY[config.estimation_engine]
    except:
        print(f"processing_scheme {config.estimation_engine} not supported")
        print(f"processing_scheme must be one of OLS, RME ")
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
        iter_control = IterControl(max_number_of_iterations=config.max_number_of_iterations)
    elif config.estimation_engine in ["OLS", ]:
        iter_control = None
    return iter_control

def transfer_function_header_from_config(config):
    transfer_function_header = TransferFunctionHeader(
        processing_scheme=config.estimation_engine,
        local_station_id=config.local_station_id,
        reference_station_id=config.reference_station_id,
        input_channels=config.input_channels,
        output_channels=config.output_channels,
        reference_channels=config.reference_channels)
    return transfer_function_header

def get_band_for_tf_estimate(band, config, local_stft_obj, remote_stft_obj):
    """
    Get data for TF estimation for a particular band.

    Parameters
    ----------
    band : aurora.time_series.frequency_band.FrequencyBand
        object with lower_bound and upper_bound to tell stft object which 
        subarray to return 
    config : aurora.sandbox.processing_config.ProcessingConfig
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
    print("TODO: Handle nan of input channels HERE!")
    if config.reference_station_id:
        band_dataset = extract_band(band, remote_stft_obj)
        RR = band_dataset[config.reference_channels]
    else:
        RR = None

    return X, Y, RR

def process_transfer_functions(config, local_stft_obj,
                               remote_stft_obj, transfer_function_obj,
                               estimate_per_channel=True):
    """
    This method is very similar to TTFestBand.
    20210810: This method currently estimates all the TF coefficients in a
    single call to estimate().  That is OK, especially if there is no missing
    data but EMTF estimates the TF for each output channel independently
    
    Parameters
    ----------
    config
    local_stft_obj
    remote_stft_obj
    transfer_function_obj

    Returns
    -------

    """
    iter_control = set_up_iter_control(config)
    regression_class = get_regression_class(config)
    for band in transfer_function_obj.frequency_bands.bands():
        X, Y, RR = get_band_for_tf_estimate(band,
                                            config,
                                            local_stft_obj,
                                            remote_stft_obj)

        regression_estimator = regression_class(X=X, Y=Y, Z=RR,
                                                iter_control=iter_control)

        Z = regression_estimator.estimate()

        transfer_function_obj.set_tf(regression_estimator, band.center_period)
    return transfer_function_obj
#</TF PROCESSING HELPERS>