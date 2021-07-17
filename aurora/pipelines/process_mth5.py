from pathlib import Path

from aurora.pipelines.processing_helpers import calibrate_stft_obj
from aurora.pipelines.processing_helpers import configure_frequency_bands
from aurora.pipelines.processing_helpers import process_transfer_functions
from aurora.pipelines.processing_helpers import run_ts_to_calibrated_stft
from aurora.pipelines.processing_helpers import transfer_function_header_from_config
from aurora.pipelines.processing_helpers import validate_sample_rate
from aurora.sandbox.processing_config import ProcessingConfig
from aurora.transfer_function.TTFZ import TTFZ

from mth5.mth5 import MTH5




def process_mth5_decimation_level(processing_cfg, run_id, units="MT"):
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
    elif isinstance(processing_cfg, ProcessingConfig):
        config = processing_cfg
    else:
        print(f"Unrecognized config of type {type(ProcessingConfig)}")
        raise Exception


    m = MTH5()
    m.open_mth5(config["mth5_path"], mode="r")

    local_run_obj = m.get_run(config["local_station_id"], run_id)
    local_run_ts = local_run_obj.to_runts()
    validate_sample_rate(local_run_ts, config)
    local_stft_obj = run_ts_to_calibrated_stft(local_run_ts, local_run_obj,
                                             config, units=units)

    if config.remote_reference_station_id:
        remote_run_obj = m.get_run(config["remote_reference_station_id"],
                                   run_id)
        remote_run_ts = remote_run_obj.to_runts()
        remote_stft_obj = run_ts_to_calibrated_stft(remote_run_ts,
                                                    remote_run_obj,
                                                    config)
    else:
        remote_stft_obj = None

    frequency_bands = configure_frequency_bands(config)
    transfer_function_header = transfer_function_header_from_config(config)
    transfer_function_obj = TTFZ(transfer_function_header,
                                 frequency_bands.number_of_bands)

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


def process_mth5_run(run_cfg, run_id, units="MT"):
    from aurora.sandbox.processing_config import RunConfig
    if isinstance(run_cfg, Path) or isinstance(run_cfg, str):
        config = RunConfig()
        config.from_json(run_cfg)
    elif isinstance(run_cfg, RunConfig):
        config = run_cfg
    else:
        print(f"Unrecognized config of type {type(run_cfg)}")
        raise Exception




    m = MTH5()
    m.open_mth5(config["mth5_path"], mode="r")

    local_run_obj = m.get_run(config["local_station_id"], run_id)
    local_run_ts = local_run_obj.to_runts()
    validate_sample_rate(local_run_ts, config)
