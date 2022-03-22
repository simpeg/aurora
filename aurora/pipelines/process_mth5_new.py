"""
Process an MTH5 using the metdata config object.
"""
# =============================================================================
# Imports
# =============================================================================
from aurora.pipelines.helpers import initialize_config_object
from aurora.pipelines.time_series_helpers import (
    calibrate_stft_obj, run_ts_to_calibrated_stft, run_ts_to_stft, validate_sample_rate)
from aurora.pipelines.transfer_function_helpers import (
    transfer_function_header_from_config, process_transfer_functions
)
from aurora.time_series.frequency_band_helpers import configure_frequency_bands
from aurora.transfer_function.transfer_function_collection import (
    TransferFunctionCollection,
)
from aurora.transfer_function.TTFZ import TTFZ

from mt_metadata.transfer_functions.core import TF
from mth5.mth5 import MTH5
# =============================================================================

def initialize_pipeline(processing_config, mth5_path=None):
    """
    A place to organize args and kwargs.
    This could be split into initialize_config() and initialize_mth5()

    Parameters
    ----------
    processing_config : str, pathlib.Path, or a RunConfig object
        If str or Path is provided, this will read in the config and return it as a
        RunConfig object.
    mth5_path : string or pathlib.Path
        optional argument.  If it is provided, it overrides the path in the RunConfig
        object

    Returns
    -------
    config : aurora.config.processing_config import RunConfig
    mth5_obj :
    """
    config = initialize_config_object(processing_config)

    # <Initialize mth5 for reading>
    if mth5_path:
        if config.stations.local.mth5_path != str(mth5_path):
            print(
                "Warning - the mth5 path supplied to initialize pipeline differs"
                "from the one in the config file"
            )
            print(f"config path changing from \n{config.stations.local.mth5_path} to \n{mth5_path}")
            config.stations.local.mth5_path = str(mth5_path)
    
    mth5_obj = MTH5(file_version="0.1.0")
    mth5_obj.open_mth5(config.stations.local.mth5_path, mode="r")

    return config, mth5_obj


def get_remote_stft(config, mth5_obj, run_id):
    """
    Compute remote reference stft
    
    :param config: DESCRIPTION
    :type config: aurora.config.Processing
    :param mth5_obj: DESCRIPTION
    :type mth5_obj: TYPE
    :param run_id: DESCRIPTION
    :type run_id: TYPE
    :return: DESCRIPTION
    :rtype: TYPE
    
    TODO: should use the MTH5 file path from
    config.stations.remote_dict[remote].mth5_path

    """
    if len(config.stations.remote) > 0: 
        remote_run_obj = mth5_obj.get_run(config.stations.remote[0].id, 
                                          run_id)
        remote_run_ts = remote_run_obj.to_runts()
        remote_stft_obj = run_ts_to_calibrated_stft(
            remote_run_ts, remote_run_obj, config
        )
    else:
        remote_stft_obj = None
    return remote_stft_obj


def prototype_decimate(decimation_obj, run_run_ts):
    """
    TODO: Move this function into time_series/decimate.py

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
    run_obj.metadata.sample_rate = decimation_obj.decimation.sample_rate

    # <Replace with rolling mean, somethng that works with time>
    # and preferably takes the average time, not the start of th
    slicer = slice(None, None, decimation_obj.decimation.factor)
    downsampled_time_axis = run_xrts.time.data[slicer]
    # <Replace with rolling mean, somethng that works with time>

    num_observations = len(downsampled_time_axis)
    channel_labels = list(run_xrts.data_vars.keys())  # run_ts.channels
    num_channels = len(channel_labels)
    new_data = np.full((num_observations, num_channels), np.nan)
    for i_ch, ch_label in enumerate(channel_labels):
        new_data[:, i_ch] = ssig.decimate(run_xrts[ch_label], decimation_obj.decimation.factor)

    xr_da = xr.DataArray(
        new_data,
        dims=["time", "channel"],
        coords={"time": downsampled_time_axis, "channel": channel_labels},
    )

    xr_ds = xr_da.to_dataset("channel")
    result = {"run": run_obj, "mvts": xr_ds}

    return result

def make_stft_objects(config, local, remote, units):
    """
    2022-02-08: Factor this out of process_mth5_decimation_level in prep for merging
    runs.   
    Parameters
    ----------
    config
    local
    remote

    Returns
    -------

    """
    local_run_obj = local["run"]
    local_run_xrts = local["mvts"]

    remote_run_obj = remote["run"]
    remote_run_xrts = remote["mvts"]
    # <CHECK DATA COVERAGE IS THE SAME IN BOTH LOCAL AND RR>
    # This should be pushed into a previous validator before pipeline starts
    # if config.reference_station_id:
    #    local_run_xrts = local_run_xrts.where(local_run_xrts.time <=
    #                                          remote_run_xrts.time[-1]).dropna(
    #                                          dim="time")
    # </CHECK DATA COVERAGE IS THE SAME IN BOTH LOCAL AND RR>

    local_stft_obj = run_ts_to_stft(config, local_run_xrts)
    local_scale_factors = config.station_scale_factors(config.local_station_id)
    # local_stft_obj = run_ts_to_stft_scipy(config, local_run_xrts)
    local_stft_obj = calibrate_stft_obj(
        local_stft_obj,
        local_run_obj,
        units=units,
        channel_scale_factors=local_scale_factors,
    )
    if config.reference_station_id:
        remote_stft_obj = run_ts_to_stft(config, remote_run_xrts)
        remote_scale_factors = config.station_scale_factors(config.reference_station_id)
        remote_stft_obj = calibrate_stft_obj(
            remote_stft_obj,
            remote_run_obj,
            units=units,
            channel_scale_factors=remote_scale_factors,
        )
    else:
        remote_stft_obj = None
    return local_stft_obj, remote_stft_obj


def process_mth5_decimation_level(config, local_stft_obj, remote_stft_obj, units="MT"):
    """
    Processing pipeline for a single decimation_level
    TODO: Add a check that the processing config sample rates agree with the
    data sampling rates otherwise raise Exception
    This method can be single station or remote based on the process cfg
    :param processing_cfg:
    :return:
    Parameters
    ----------
    config : aurora.config.decimation_level_config.DecimationLevelConfig
    units

    Returns
    -------
    transfer_function_obj : aurora.transfer_function.TTFZ.TTFZ



    """
    # local_stft_obj, remote_stft_obj = make_stft_objects(config, local, remote, units)
    # local_run_obj = local["run"]
    # local_run_xrts = local["mvts"]
    #
    # remote_run_obj = remote["run"]
    # remote_run_xrts = remote["mvts"]
    #
    # # <CHECK DATA COVERAGE IS THE SAME IN BOTH LOCAL AND RR>
    # # This should be pushed into a previous validator before pipeline starts
    # # if config.reference_station_id:
    # #    local_run_xrts = local_run_xrts.where(local_run_xrts.time <=
    # #                                          remote_run_xrts.time[-1]).dropna(
    # #                                          dim="time")
    # # </CHECK DATA COVERAGE IS THE SAME IN BOTH LOCAL AND RR>
    #
    # local_stft_obj = run_ts_to_stft(config, local_run_xrts)
    # local_scale_factors = config.station_scale_factors(config.local_station_id)
    # # local_stft_obj = run_ts_to_stft_scipy(config, local_run_xrts)
    # local_stft_obj = calibrate_stft_obj(
    #     local_stft_obj,
    #     local_run_obj,
    #     units=units,
    #     channel_scale_factors=local_scale_factors,
    # )
    # if config.reference_station_id:
    #     remote_stft_obj = run_ts_to_stft(config, remote_run_xrts)
    #     remote_scale_factors = config.station_scale_factors(config.reference_station_id)
    #     remote_stft_obj = calibrate_stft_obj(
    #         remote_stft_obj,
    #         remote_run_obj,
    #         units=units,
    #         channel_scale_factors=remote_scale_factors,
    #     )
    # else:
    #     remote_stft_obj = None
    frequency_bands = configure_frequency_bands(config)
    transfer_function_header = transfer_function_header_from_config(config)
    transfer_function_obj = TTFZ(
        transfer_function_header, frequency_bands, processing_config=config
    )

    transfer_function_obj = process_transfer_functions(
        config, local_stft_obj, remote_stft_obj, transfer_function_obj
    )

    transfer_function_obj.apparent_resistivity(units=units)
    return transfer_function_obj


def get_data_from_decimation_level_from_mth5(config, mth5_obj, run_id):
    """

    Parameters
    ----------
    config : decimation_level_config
    mth5_obj

    Returns
    -------

    Somewhat complicated function -- see issue #13.  Ultimately this method could be
    embedded in mth5, where the specific attributes of the config needed for this
    method are passed as explicit arguments.

    Should be able to
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


def export_tf(tf_collection, station_metadata_dict={}, survey_dict={}):
    """
    This method may wind up being embedded in the TF class
    Assign transfer_function, residual_covariance, inverse_signal_power, station, survey

    Returns
    -------

    """
    merged_tf_dict = tf_collection.get_merged_dict()
    tf_cls = TF()
    # Transfer Function
    renamer_dict = {"output_channel": "output", "input_channel": "input"}
    tmp = merged_tf_dict["tf"].rename(renamer_dict)
    tf_cls.transfer_function = tmp

    isp = merged_tf_dict["cov_ss_inv"]
    renamer_dict = {"input_channel_1": "input", "input_channel_2": "output"}
    isp = isp.rename(renamer_dict)
    tf_cls.inverse_signal_power = isp

    res_cov = merged_tf_dict["cov_nn"]
    renamer_dict = {"output_channel_1": "input", "output_channel_2": "output"}
    res_cov = res_cov.rename(renamer_dict)
    tf_cls.residual_covariance = res_cov

    tf_cls.station_metadata._runs = []
    tf_cls.station_metadata.from_dict(station_metadata_dict)
    tf_cls.survey_metadata.from_dict(survey_dict)
    return tf_cls


def process_mth5_run(
    run_cfg,
    run_id,
    units="MT",
    show_plot=False,
    z_file_path=None,
    return_collection=True,
    **kwargs,
):
    """
    2022-02-08: TODO: Need to replace run_id (string) with a list, or, maybe,
    support either a list of strings or a single string.
    Stages here:
    1. Read in the config and figure out how many decimation levels there are

    Parameters
    ----------
    run_cfg: config object or path to config
    run_id: string
    units: string
        "MT" or "SI".  To be deprecated once data have units embedded
    show_plot: boolean
        Only used for dev
    z_file_path: string or pathlib.Path
        Target path for a z_file output if desired
    return_collection : boolean
        return_collection=False will return an mt_metadata TF object
    kwargs

    Returns
    -------

    """
    mth5_path = kwargs.get("mth5_path", None)
    processing_config, mth5_obj = initialize_pipeline(run_cfg, mth5_path)
    print(
        f"config indicates {processing_config.number_of_decimation_levels} "
        f"decimation levels to process: {processing_config.decimation_level_ids}"
    )

    tf_dict = {}

    for dec_level_id in processing_config.decimation_level_ids:

        processing_config = processing_config.decimation_level_configs[dec_level_id]
        processing_config.local_station_id = processing_config.local_station_id
        processing_config.reference_station_id = processing_config.reference_station_id
        processing_config.channel_scale_factors = processing_config.channel_scale_factors


        if dec_level_id == 0:
            local, remote = get_data_from_decimation_level_from_mth5(
                processing_config, mth5_obj, run_id
            )

            # APPLY TIMING CORRECTIONS HERE
        else:
            # This code structure assumes application of cascading decimation,
            # and that the decimated data will be accessed from the previous
            # decimation level.  This should be revisited .. it may make
            # more sense to have a get_decimation_level() interface that provides an
            # option of applying decimation or loading predecimated data.
            local = prototype_decimate(processing_config, local)
            if processing_config.reference_station_id:
                remote = prototype_decimate(processing_config, remote)

        # </GET DATA>
        local_stft_obj, remote_stft_obj = make_stft_objects(processing_config,
                                                            local, remote, units)
        # MERGE STFTS goes here
        tf_obj = process_mth5_decimation_level(
            processing_config, local_stft_obj, remote_stft_obj, units=units
        )
        tf_dict[dec_level_id] = tf_obj

        if show_plot:
            from aurora.sandbox.plot_helpers import plot_tf_obj

            plot_tf_obj(tf_obj, out_filename="out")

    # TODO: Add run_obj to TransferFunctionCollection ? WHY? so it doesn't need header?
    tf_collection = TransferFunctionCollection(header=tf_obj.tf_header, tf_dict=tf_dict)
    local_run_obj = mth5_obj.get_run(processing_config["local_station_id"], run_id)

    if z_file_path:
        tf_collection.write_emtf_z_file(z_file_path, run_obj=local_run_obj)

    if return_collection:
        return tf_collection
    else:
        # intended to be the default in future
        station_metadata = local_run_obj.station_group.metadata
        station_metadata._runs = []
        run_metadata = local_run_obj.metadata
        station_metadata.add_run(run_metadata)
        survey_dict = mth5_obj.survey_group.metadata.to_dict()
        
        print(station_metadata.run_list)
        tf_cls = export_tf(
            tf_collection, 
            station_metadata_dict=station_metadata.to_dict(),
            survey_dict=survey_dict
        )
        return tf_cls