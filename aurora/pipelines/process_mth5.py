from pathlib import Path

from aurora.sandbox.processing_config import ProcessingConfig
from aurora.time_series.frequency_band import FrequencyBands
from aurora.time_series.frequency_band_helpers import extract_band
from aurora.time_series.windowing_scheme import WindowingScheme
from aurora.transfer_function.iter_control import IterControl
from aurora.transfer_function.transfer_function_header import \
    TransferFunctionHeader
from aurora.transfer_function.TRME import TRME
from aurora.transfer_function.TRME_RR import TRME_RR
from aurora.transfer_function.TTFZ import TTFZ


from mth5.mth5 import MTH5

def run_ts_to_calibrated_stft(run_ts, run_obj, config):
    windowing_scheme = WindowingScheme(
        taper_family=config.taper_family,
        num_samples_window=config.num_samples_window,
        num_samples_overlap=config.num_samples_overlap,
        sampling_rate=run_ts.sample_rate)

    windowed_obj = windowing_scheme.apply_sliding_window(run_ts.dataset)
    tapered_obj = windowing_scheme.apply_taper(windowed_obj)
    stft_obj = windowing_scheme.apply_fft(tapered_obj)
    # <CALIBRATE>
    for channel_id in stft_obj.keys():
        mth5_channel = run_obj.get_channel(channel_id)
        channel_filter = mth5_channel.channel_response_filter
        calibration_response = channel_filter.complex_response(
            stft_obj.frequency.data)
    #
    # # if UNITS == "SI":
    # #     if channel_id[0].lower() == 'h':
    # #         calibration_response /= 1e-9  # SI Units
    #     stft_obj[channel_id].data /= calibration_response
    # </CALIBRATE>

    stft_obj_xrda = stft_obj.to_array("channel")
    return stft_obj_xrda

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
        local_site=config.local_station_id,
        remote_site=config.remote_reference_station_id,
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


def process_mth5_decimation_level(processing_cfg, run_id):
    """
    Processing pipeline for a single decimation_level
    Note that we will need a check that the processing config sample rates agree
    with the data sampling rates otherwise raise Exception
    This method can be single station or remote based on the process cfg
    :param processing_cfg:
    :return:
    """
    if isinstance(processing_cfg, Path) or isinstance(processing_cfg, str):
        config = ProcessingConfig()
        config.from_json(processing_cfg)
    else:
        config = processing_cfg


    m = MTH5()
    m.open_mth5(config["mth5_path"], mode="r")

    local_run_obj = m.get_run(config["local_station_id"], run_id)
    local_run_ts = local_run_obj.to_runts()
    validate_sample_rate(local_run_ts, config)
    local_stft_obj = run_ts_to_calibrated_stft(local_run_ts, local_run_obj,
                                             config)

    if config.remote_reference_station_id:
        remote_run_obj = m.get_run(config["remote_reference_station_id"],
                                   run_id)
        remote_run_ts = remote_run_obj.to_runts()
        remote_stft_obj = run_ts_to_calibrated_stft(remote_run_ts,
                                                    remote_run_obj,
                                                    config)

    frequency_bands = configure_frequency_bands(config)
    transfer_function_header = transfer_function_header_from_config(config)
    transfer_function_obj = TTFZ(transfer_function_header,
                                 frequency_bands.number_of_bands)

    for i_band in range(frequency_bands.number_of_bands):
        band = frequency_bands.band(i_band)
        band_dataarray = extract_band(band, local_stft_obj)
        band_dataset = band_dataarray.to_dataset("channel")
        X = band_dataset[config.input_channels]
        Y = band_dataset[config.output_channels]
        if config.remote_reference_station_id:
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
        transfer_function_obj.set_tf(i_band, regression_estimator, band.center_period)

    transfer_function_obj.apparent_resistivity(units="MT")
    print(transfer_function_obj.rho.shape)
    print(transfer_function_obj.rho[0])
    print(transfer_function_obj.rho[-1])
    return transfer_function_obj

