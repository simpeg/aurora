from aurora.time_series.frequency_band_helpers import extract_band
from aurora.transfer_function.iter_control import IterControl
from aurora.transfer_function.TRME import TRME
from aurora.transfer_function.TRME_RR import TRME_RR


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


def process_transfer_functions(config, frequency_bands, local_stft_obj,
                               remote_stft_obj, transfer_function_obj):
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
    return transfer_function_obj