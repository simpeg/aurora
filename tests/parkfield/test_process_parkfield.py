from pathlib import Path
from aurora.sandbox.plot_helpers import plot_tf_obj
from aurora.time_series.frequency_band_helpers import configure_frequency_bands

processing_cfg = Path("config", "pkd_processing_config.json")
processing_cfg = Path("config", "ascii_pkd_processing_config.json")

def process_mth5_decimation_level(processing_cfg, run_id, units="MT"):
    """
    20210718: Moved this code out of process_mth5 to keep this test passing.
    This will be replaced with process_mth5_run as soon as the configs have
    been updated
    TODO : DEPRECATE THIS METHOS
    Processing pipeline for a single decimation_level
    Note that we will need a check that the processing config sample rates agree
    with the data sampling rates otherwise raise Exception
    This method can be single station or remote based on the process cfg
    :param processing_cfg:
    :return:
    """
    from pathlib import Path
    from aurora.sandbox.processing_config import ProcessingConfig
    from mth5.mth5 import MTH5
    from aurora.pipelines.processing_helpers import process_transfer_functions
    from aurora.pipelines.processing_helpers import run_ts_to_calibrated_stft
    from aurora.pipelines.processing_helpers import transfer_function_header_from_config
    from aurora.pipelines.processing_helpers import validate_sample_rate
    from aurora.transfer_function.TTFZ import TTFZ
    if isinstance(processing_cfg, Path) or isinstance(processing_cfg, str):
        config = ProcessingConfig()
        config.from_json(processing_cfg)
    elif isinstance(processing_cfg, ProcessingConfig):
        config = processing_cfg
    else:
        print(f"Unrecognized config of type {type(ProcessingConfig)}")
        raise Exception


    mth5_obj = MTH5()
    mth5_obj.open_mth5(config["mth5_path"], mode="r")

    local_run_obj = mth5_obj.get_run(config["local_station_id"], run_id)
    local_run_ts = local_run_obj.to_runts()
    validate_sample_rate(local_run_ts, config)
    local_stft_obj = run_ts_to_calibrated_stft(local_run_ts, local_run_obj,
                                             config, units=units)

    remote_stft_obj = None#get_remote_stft(config, mth5_obj, run_id)

    frequency_bands = configure_frequency_bands(config)
    transfer_function_header = transfer_function_header_from_config(config)
    transfer_function_obj = TTFZ(transfer_function_header,
                                 frequency_bands)

    transfer_function_obj = process_transfer_functions(config,
                                                       frequency_bands,
                                                       local_stft_obj,
                                                       remote_stft_obj,
                                                       transfer_function_obj)

    transfer_function_obj.apparent_resistivity(units=units)
    print(transfer_function_obj.rho.shape)
    print(transfer_function_obj.rho[0])
    print(transfer_function_obj.rho[-1])
    return transfer_function_obj

tf_obj = process_mth5_decimation_level(processing_cfg, "001", units="SI")
plot_tf_obj(tf_obj)
