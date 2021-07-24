
from aurora.time_series.frequency_band_helpers import extract_band
from aurora.time_series.frequency_band import FrequencyBands
from aurora.time_series.windowing_scheme import WindowingScheme
from aurora.transfer_function.iter_control import IterControl
from aurora.transfer_function.transfer_function_header import \
    TransferFunctionHeader
from aurora.transfer_function.TRME import TRME
from aurora.transfer_function.TRME_RR import TRME_RR



def configure_frequency_bands(config):
    frequency_bands = FrequencyBands()
    if config["band_setup_style"] == "EMTF":
        frequency_bands.from_emtf_band_setup(
            filepath=config.emtf_band_setup_file,
            sampling_rate=config.sample_rate,
            decimation_level=1,
            num_samples_window=config.num_samples_window)
    else:
        print("TODO:Write a method to choose lower and upper bounds, "
              "and number of bands to split it into")
    return frequency_bands


def transfer_function_header_from_config(config):
    transfer_function_header = TransferFunctionHeader(
        processing_scheme=config.estimation_engine,
        local_station_id=config.local_station_id,
        reference_station_id=config.reference_station_id,
        input_channels=config.input_channels,
        output_channels=config.output_channels,
        reference_channels=config.reference_channels)
    return transfer_function_header


def validate_sample_rate(run_ts, config):
    if run_ts.sample_rate != config.sample_rate:
        print(f"sample rate in run time series {local_run_ts.sample_rate} and "
              f"processing config {config.sample_rate} do not match")
        raise Exception
    return


def run_ts_to_stft(config, run_xrts):
    """

    Parameters
    ----------
    config : ShortTimeFourierTransformConfig object
    run_ts ; mth5.RunTS (but could be replaced by the xr.dataset....)

    Returns
    -------

    """
    windowing_scheme = WindowingScheme(
    taper_family = config.taper_family,
    num_samples_window = config.num_samples_window,
    num_samples_overlap = config.num_samples_overlap,
    sampling_rate = config.sample_rate)

    windowed_obj = windowing_scheme.apply_sliding_window(run_xrts)
    tapered_obj = windowing_scheme.apply_taper(windowed_obj)
    stft_obj = windowing_scheme.apply_fft(tapered_obj)
    return stft_obj


def run_ts_to_calibrated_stft(run_ts, run_obj, config, units="MT"):

    stft_obj = run_ts_to_stft(config, run_ts.dataset)
    stft_obj = calibrate_stft_obj(stft_obj, run_obj, units=units)

    stft_obj_xrda = stft_obj.to_array("channel")
    print("why bother making this an array here?")
    return stft_obj_xrda

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


def process_transfer_functions(config, local_stft_obj,
                               remote_stft_obj, transfer_function_obj):
    for band in transfer_function_obj.frequency_bands.bands():
        print(f"Processing band {band.center_period}s")
        #expecitng dataArray as local_stft_obj
        band_dataarray = extract_band(band, local_stft_obj)
        band_dataset = band_dataarray.to_dataset("channel")
        X = band_dataset[config.input_channels]
        Y = band_dataset[config.output_channels]
        if config.reference_station_id:
            band_dataarray = extract_band(band, remote_stft_obj)
            band_dataset = band_dataarray.to_dataset("channel")
            RR = band_dataset[config.reference_channels]

        if config.estimation_engine == "OLS":
            regression_estimator = RegressionEstimator(X=X, Y=Y)
            Z = regression_estimator.estimate_ols()
        elif config.estimation_engine=="RME":
            iter_control = IterControl(max_number_of_iterations=config.max_number_of_iterations)
            regression_estimator = TRME(X=X, Y=Y, iter_control=iter_control)
            Z = regression_estimator.estimate()
        elif config.estimation_engine=="TRME_RR":
            iter_control = IterControl(max_number_of_iterations=config.max_number_of_iterations)
            regression_estimator = TRME_RR(X=X, Y=Y, Z=RR,
                                           iter_control=iter_control)
            Z = regression_estimator.estimate()
        else:
            print(f"processing_scheme {config.estimation_engine} not supported")
            print(f"processing_scheme must be one of OLS, RME "
            f"not supported")
            raise Exception
        print(f"{band.center_period} {config.estimation_engine}, \n {Z}")
        transfer_function_obj.set_tf(regression_estimator, band.center_period)
    return transfer_function_obj