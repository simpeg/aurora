from pathlib import Path

from aurora.pipelines.processing_helpers import calibrate_stft_obj
from aurora.pipelines.processing_helpers import process_transfer_functions
from aurora.pipelines.processing_helpers import run_ts_to_calibrated_stft
from aurora.pipelines.processing_helpers import run_ts_to_stft

# from aurora.pipelines.processing_helpers import run_ts_to_stft_scipy
from aurora.pipelines.processing_helpers import transfer_function_header_from_config
from aurora.pipelines.processing_helpers import validate_sample_rate
from aurora.sandbox.processing_config import RunConfig
from aurora.time_series.frequency_band_helpers import configure_frequency_bands
from aurora.transfer_function.transfer_function_collection import (
    TransferFunctionCollection,
)
from aurora.transfer_function.TTFZ import TTFZ


from mth5.mth5 import MTH5


def initialize_pipeline(run_config):
    if isinstance(run_config, Path) or isinstance(run_config, str):
        config = RunConfig()
        config.from_json(run_config)
    elif isinstance(run_config, RunConfig):
        config = run_config
    else:
        print(f"Unrecognized config of type {type(run_config)}")
        raise Exception

    mth5_obj = MTH5()
    mth5_obj.open_mth5(config["mth5_path"], mode="r")
    return config, mth5_obj


def get_remote_stft(config, mth5_obj, run_id):
    if config.reference_station_id:
        remote_run_obj = mth5_obj.get_run(config["reference_station_id"], run_id)
        remote_run_ts = remote_run_obj.to_runts()
        remote_stft_obj = run_ts_to_calibrated_stft(
            remote_run_ts, remote_run_obj, config
        )
    else:
        remote_stft_obj = None
    return remote_stft_obj


def prototype_decimate(config, run_run_ts):
    """

    Parameters
    ----------
    config : DecimationConfig object
    run_run_ts

    Returns
    -------

    """
    import numpy as np
    import scipy.signal as ssig
    import xarray as xr

    run_obj = run_run_ts["run"]
    run_xrts = run_run_ts["mvts"]
    run_obj.metadata.sample_rate = config.sample_rate

    # <Replace with rolling mean, somethng that works with time>
    # and preferably takes the average time, not the start of th
    slicer = slice(None, None, config.decimation_factor)
    downsampled_time_axis = run_xrts.time.data[slicer]
    # <Replace with rolling mean, somethng that works with time>

    num_observations = len(downsampled_time_axis)
    channel_labels = list(run_xrts.data_vars.keys())  # run_ts.channels
    num_channels = len(channel_labels)
    new_data = np.full((num_observations, num_channels), np.nan)
    for i_ch, ch_label in enumerate(channel_labels):
        new_data[:, i_ch] = ssig.decimate(run_xrts[ch_label], config.decimation_factor)

    xr_da = xr.DataArray(
        new_data,
        dims=["time", "channel"],
        coords={"time": downsampled_time_axis, "channel": channel_labels},
    )

    xr_ds = xr_da.to_dataset("channel")
    result = {"run": run_obj, "mvts": xr_ds}

    return result


def process_mth5_decimation_level(config, local, remote, units="MT"):
    """
    Processing pipeline for a single decimation_level
    TODO: Add a check that the processing config sample rates agree with the
    data sampling rates otherwise raise Exception
    This method can be single station or remote based on the process cfg
    :param processing_cfg:
    :return:
    Parameters
    ----------
    config : ProcessingConfig (for a decimation level)
    units

    Returns
    -------


    """
    local_run_obj = local["run"]
    local_run_xrts = local["mvts"]
    local_stft_obj = run_ts_to_stft(config, local_run_xrts)
    # local_stft_obj = run_ts_to_stft_scipy(config, local_run_xrts)
    local_stft_obj = calibrate_stft_obj(local_stft_obj, local_run_obj, units=units)

    remote_run_obj = remote["run"]
    remote_run_xrts = remote["mvts"]
    if config.reference_station_id:
        remote_stft_obj = run_ts_to_stft(config, remote_run_xrts)
        remote_stft_obj = calibrate_stft_obj(
            remote_stft_obj, remote_run_obj, units=units
        )
    else:
        remote_stft_obj = None

    frequency_bands = configure_frequency_bands(config)
    transfer_function_header = transfer_function_header_from_config(config)
    transfer_function_obj = TTFZ(
        transfer_function_header, frequency_bands, processing_config=config
    )

    transfer_function_obj = process_transfer_functions(
        config, local_stft_obj, remote_stft_obj, transfer_function_obj
    )

    transfer_function_obj.apparent_resistivity(units=units)
    print(transfer_function_obj.rho.shape)
    print(transfer_function_obj.rho[0])
    print(transfer_function_obj.rho[-1])
    return transfer_function_obj


def get_data_from_decimation_level_from_mth5(config, mth5_obj, run_id):
    """

    Parameters
    ----------
    config : decimation_level_config
    mth5_obj

    Returns
    -------

    Somewhat complicated function -- see issue #13.

    SHould be able to
    1. accept a config and an mth5_obj and return decimation_level_0,
    2. Accept data from a given decimation level, and decimation
    instrucntions and return it
    3. If we decide to house decimated data in an mth5 should return time
    series for the run at the perscribed decimation level

    Thus args are
    decimation_level_config, mth5,
    decimation_level_config, runs and run_ts'
    decimation_level_config, mth5
    Returns: tuple of dicts
        Each dictionary is associated with a staiton, one for local and one
        for remote at this point
        Each Dict has keys "run" and "mvts" which are the mth5_run and the
        mth5_run_ts objects respectively for the associated station
    -------

    """
    # <LOCAL>
    local_run_obj = mth5_obj.get_run(config["local_station_id"], run_id)
    local_run_ts = local_run_obj.to_runts()
    validate_sample_rate(local_run_ts, config)
    local = {"run": local_run_obj, "mvts": local_run_ts.dataset}
    # </LOCAL>

    # <REMOTE>
    if config.reference_station_id:
        remote_run_obj = mth5_obj.get_run(config["reference_station_id"], run_id)
        remote_run_ts = remote_run_obj.to_runts()
        validate_sample_rate(remote_run_ts, config)
        remote = {"run": remote_run_obj, "mvts": remote_run_ts.dataset}
    else:
        remote = {"run": None, "mvts": None}
    # </REMOTE>
    return local, remote


def process_mth5_run(run_cfg, run_id, units="MT", show_plot=True, z_file_path=None):
    """
    Stages here:
    1. Read in the config and figure out how many decimation levels there are
    Parameters
    ----------
    run_cfg
    run_id
    units

    Returns
    -------

    """
    run_config, mth5_obj = initialize_pipeline(run_cfg)
    print(
        f"config indicates {run_config.number_of_decimation_levels} "
        f"decimation levels to process: {run_config.decimation_level_ids}"
    )
    local_run_obj = mth5_obj.get_run(run_config["local_station_id"], run_id)
    tf_dict = {}

    for dec_level_id in run_config.decimation_level_ids:

        processing_config = run_config.decimation_level_configs[dec_level_id]
        processing_config.local_station_id = run_config.local_station_id
        processing_config.reference_station_id = run_config.reference_station_id

        # <GET DATA>
        # Careful here -- for multiple station processing we will need to load
        # many time series' here.  Will probably have another version of
        # process_mth5_run for MMT

        if dec_level_id == 0:
            local, remote = get_data_from_decimation_level_from_mth5(
                processing_config, mth5_obj, run_id
            )
        else:
            local = prototype_decimate(processing_config, local)
            if processing_config.reference_station_id:
                remote = prototype_decimate(processing_config, remote)

        # </GET DATA>

        tf_obj = process_mth5_decimation_level(
            processing_config, local, remote, units=units
        )
        tf_dict[dec_level_id] = tf_obj

        if show_plot:
            from aurora.sandbox.plot_helpers import plot_tf_obj

            plot_tf_obj(tf_obj, out_filename="out")

    tf_collection = TransferFunctionCollection(header=tf_obj.tf_header, tf_dict=tf_dict)
    if z_file_path:
        tf_collection.write_emtf_z_file(z_file_path, run_obj=local_run_obj)
    return tf_collection
