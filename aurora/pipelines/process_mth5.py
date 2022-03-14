"""
Note 1: process_mth5_runs assumes application of cascading decimation, and that the
decimated data will be accessed from the previous decimation level.  This should be
revisited. It may make more sense to have a get_decimation_level() interface that
provides an option of applying decimation or loading predecimated data.

#Note 2: Question: Can we run into cases where some of these runs can be decimated
and others can not?  We need a way to handle this. For example, a short run may not
yield any data from a later decimation level.  Returning an empty xarray may work,
... It is desireable that the empty xarray, or whatever comes back to pass through STFT
and MERGE steps smoothy.
If it is empty, we could just drop the run entirely but this may have adverse
consequences on downstream bookkeeping, like the creation of station_metadata before
packaging the tf for export.  This could be worked around by extracting the metadata
at the start of this method. In fact, it would be a good idea in general to run a
pre-check on the data that identifies which decimation levels are valid for each run.
"""
import pandas as pd
import xarray as xr

from aurora.pipelines.helpers import initialize_config
from aurora.pipelines.time_series_helpers import calibrate_stft_obj
from aurora.pipelines.time_series_helpers import get_data_from_mth5
from aurora.pipelines.time_series_helpers import prototype_decimate
from aurora.pipelines.time_series_helpers import run_ts_to_calibrated_stft
from aurora.pipelines.time_series_helpers import run_ts_to_stft
from aurora.pipelines.transfer_function_helpers import process_transfer_functions
from aurora.pipelines.transfer_function_helpers import (
    transfer_function_header_from_config,
)

# from aurora.pipelines.time_series_helpers import run_ts_to_stft_scipy
from aurora.time_series.frequency_band_helpers import configure_frequency_bands
from aurora.transfer_function.transfer_function_collection import (
    TransferFunctionCollection,
)
from aurora.transfer_function.TTFZ import TTFZ

from aurora.tf_kernel.dataset import DatasetDefinition
from mt_metadata.transfer_functions.core import TF
from mth5.mth5 import MTH5


def initialize_pipeline(run_config, mth5_path=None):
    """
    A place to organize args and kwargs.
    This could be split into initialize_config() and initialize_mth5()

    Parameters
    ----------
    run_config : str, pathlib.Path, or a RunConfig object
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
    config = initialize_config(run_config)

    # <Initialize mth5 for reading>
    if mth5_path:
        if config["mth5_path"] != str(mth5_path):
            print(
                "Warning - the mth5 path supplied to initialize pipeline differs"
                "from the one in the config file"
            )
            print(f"config path changing from \n{config['mth5_path']} to \n{mth5_path}")
            config.mth5_path = str(mth5_path)
    mth5_obj = MTH5(file_version="0.1.0")
    mth5_obj.open_mth5(config["mth5_path"], mode="r")
    # </Initialize mth5 for reading>
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



def make_stft_objects(config, local, remote, units):
    """
    2022-02-08: Factor this out of process_tf_decimation_level in prep for merging
    runs.
    2022-03-13: This method to be deprecated/replaced by make_stft_objects_new (which
    will eventually be renamed back to make_stft_objects after testing)
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


def make_stft_objects_new(config, run_obj, run_xrts, units, station_id):#local, remote,
    """
    Note 1: CHECK DATA COVERAGE IS THE SAME IN BOTH LOCAL AND RR
    This should be pushed into a previous validator before pipeline starts
    # # if config.reference_station_id:
    # #    local_run_xrts = local_run_xrts.where(local_run_xrts.time <=
    # #                                          remote_run_xrts.time[-1]).dropna(
    # #                                          dim="time")

    2022-02-08: Factor this out of process_tf_decimation_level in prep for merging
    runs.
    2022-03-13: This method will supercede make_stft_objects.  This will operate on
    local and remote independently.


    Parameters
    ----------
    config
    local
    remote

    Returns
    -------

    """
    stft_obj = run_ts_to_stft(config, run_xrts)
    scale_factors = config.station_scale_factors(station_id)
    # local_stft_obj = run_ts_to_stft_scipy(config, local_run_xrts)
    stft_obj = calibrate_stft_obj(
        stft_obj,
        run_obj,
        units=units,
        channel_scale_factors=scale_factors,
    )
    return stft_obj


def process_tf_decimation_level(config, local_stft_obj, remote_stft_obj, units="MT"):
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
    run_config, mth5_obj = initialize_pipeline(run_cfg, mth5_path)
    print(
        f"config indicates {run_config.number_of_decimation_levels} "
        f"decimation levels to process: {run_config.decimation_level_ids}"
    )

    tf_dict = {}

    for dec_level_id in run_config.decimation_level_ids:

        processing_config = run_config.decimation_level_configs[dec_level_id]
        processing_config.local_station_id = run_config.local_station_id
        processing_config.reference_station_id = run_config.reference_station_id
        processing_config.channel_scale_factors = run_config.channel_scale_factors


        if dec_level_id == 0:
            local, remote = get_data_from_mth5(processing_config, mth5_obj, run_id)
            # APPLY TIMING CORRECTIONS HERE
        else:
            # See Note#1 at top of module
            local = prototype_decimate(processing_config, local)
            if processing_config.reference_station_id:
                remote = prototype_decimate(processing_config, remote)

        # </GET DATA>
        local_stft_obj, remote_stft_obj = make_stft_objects(processing_config,
                                                            local, remote, units)
        # MERGE STFTS goes here
        tf_obj = process_tf_decimation_level(
            processing_config, local_stft_obj, remote_stft_obj, units=units
        )
        tf_dict[dec_level_id] = tf_obj

        if show_plot:
            from aurora.sandbox.plot_helpers import plot_tf_obj

            plot_tf_obj(tf_obj, out_filename="out")

    # TODO: Add run_obj to TransferFunctionCollection ? WHY? so it doesn't need header?
    tf_collection = TransferFunctionCollection(header=tf_obj.tf_header, tf_dict=tf_dict)
    local_run_obj = mth5_obj.get_run(run_config["local_station_id"], run_id)

    if z_file_path:
        tf_collection.write_emtf_z_file(z_file_path, run_obj=local_run_obj)

    if return_collection:
        return tf_collection
    else:
        # intended to be the default in future

        # N.B. Currently, only the last run makes it into the tf object,
        # but we can simply iterate of the run list here, getting run metadata
        # station_metadata.add_run(run_metadata)
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



def process_mth5_from_dataset_definition(
        run_cfg,
        dataset_definition=None,
        units="MT",
        show_plot=False,
        z_file_path=None,
        return_collection=True,
        **kwargs,
):
    """
    2022-02-08: TODO: Replace run_id (string) with a list, or, maybe,
    support either a list of strings or a single string.
    2022-03-07: TODO: Note that run_lists will in general be different at the local
    and remote stations.  If the run_lists are not provided specifically,
    we can extract them from the mth5s.  I would prefer to handle all that logic outside
    of this method, which expects the decision of what to process to be already made.
    Thus, we need run_lists with time_intervals, thus we are back at the dataset
    defintion
    2022-03-12: We could still support run_ids as a list, and what would happen is we
    would then build the DatasetDefinition from the mth5.channel_summary and the
    run_list.  But this would be best done outside this method.  Basically, we want
    to have a dataset_definition.  If one is not given one will be generated.
    2022-03-13: This method will eventually replace process_mth5_run and
    process_mth5_runs and be named process_mth5.

    Stages here:
    1. Read in the config and figure out how many decimation levels there are
    2. ToDo: Based on the run durations, and sampling rates, determined which runs
    are valid for which decimation levels, or for which effective sample rates.
    Parameters
    ----------
    run_cfg: aurora.config.processing_config.RunConfig object or path to json
    representation of that config
    run_ids: list of strings, supports a single string as well
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
    from aurora.pipelines.time_series_helpers import get_data_from_mth5_new
    mth5_path = kwargs.get("mth5_path", None)
    run_config, mth5_obj = initialize_pipeline(run_cfg, mth5_path)

    #<Move into TFKernel()>
    if dataset_definition is None:
        print("This case (no DatasetDefinition) will probably be deprecated (soon)")
        print("But for now we are supporting a run list as well so that old tests pass")
        print("But a run_list is only appropriate for single station processing, "
              "in general we would need a list of (station_id, run_id) tuples")
        print("basically, I am going to assume that the dataset definition is already "
              "cleanly restricted down to only what I want to process, "
              "local_station_id, and remote station_id are only ids, and all runs "
              "listed are candidates for TF")
        #make run_id a list if it is a string denoting a single run
        if isinstance(run_ids, str):
            run_ids = [run_ids, ]
        else:
            print("This processing is experimentatal being developed Mar 2022")
            print(f"run_ids = {run_ids}")
            if not isinstance(run_ids,list):
                print("Error: Expecting run_ids to be a list")
                raise Exception
        print("NOW MAKE DATASET DEFINITION FROM run_ids")
        summary_df = mth5_obj.channel_summary
        #summary_df = pd.read_csv("channel_summary.csv", parse_dates=["start", "end"])
        dataset_definition = DatasetDefinition()
        dataset_df = dataset_definition.from_mth5_channel_summary(summary_df)
        print("NOW restrict dataset definition to run_ids")
        dataset_df = dataset_df[dataset_df["run_id"].isin(run_ids)]
        print("NOW might be a good time to reindex the defn_df")
    else:
        dataset_df = dataset_definition.df
    #</Move into TFKernel()>

    print(
        f"config indicates {run_config.number_of_decimation_levels} "
        f"decimation levels to process: {run_config.decimation_level_ids}"
    )

    tf_dict = {}

    #MAKE SURE DATASET DEFINITION DF HAS "run", "run_ts" columns

    all_run_objs = len(dataset_df) * [None]
    all_run_ts_objs = len(dataset_df) * [None]
    all_stft_objs = len(dataset_df) * [None]
    dataset_df["run"] = None
    dataset_df["run_dataarray"] = None
    dataset_df["stft"] = None

    for dec_level_id in run_config.decimation_level_ids:
        #<This will be replaced by a 1-line call issue #153>
        processing_config = run_config.decimation_level_configs[dec_level_id]
        processing_config.local_station_id = run_config.local_station_id
        processing_config.reference_station_id = run_config.reference_station_id
        processing_config.channel_scale_factors = run_config.channel_scale_factors
        #</This will be replaced by a 1-line call issue #153>


        # local_dataset_defn_df = defn_df[defn_df[
        #                                     "station_id"]==processing_config.local_station_id]
        # remote_dataset_defn_df = defn_df[defn_df[
        #                                      "station_id"]==processing_config.remote_station_id]
        # local_grouper = local_dataset_defn_df.groupby("run")
        # remote_grouper = remote_dataset_defn_df.groupby("run")
        # #grouper = df.groupby(["station", "run"])

        # <GET TIME SERIES DATA>
        if dec_level_id == 0:
            for i,row in dataset_df.iterrows():
                run_dict = get_data_from_mth5_new(processing_config, mth5_obj,
                                                  row.station_id, row.run_id)
                dataset_df["run"].loc[i] = run_dict["run"]
                dataset_df["run_dataarray"].loc[i] = run_dict["mvts"].to_array("channel")
                #Dataframe doesn't like an xarray Dataset in a cell, need to convert
                # to DataArray

                all_run_objs[i] = run_dict["run"]
                all_run_ts_objs[i] = run_dict["mvts"]
                #careful here, i must run from 0 to len(df), otherwise will get
                # indexing errors.  Maybe reset_index() before this loop?

                # APPLY TIMING CORRECTIONS HERE
        else:
            # See Note 1 top of module
            # See Note 2 top of module
            # 2022-02-27: This method modified to iterate over lists
            print("ENSURE HERE THAT LOCAL RUN IS MODIFIED IN PLACE IN THE LIST!!!")
            for i,row in dataset_df.iterrows():
                print(i)
                run_xrts = row["run_dataarray"].to_dataset("channel")
                input_dict = {"run":row["run"], "mvts":run_xrts}
                run_dict = prototype_decimate(processing_config, input_dict)
                dataset_df["run"].loc[i] = run_dict["run"]
                dataset_df["run_dataarray"].loc[i] = run_dict["mvts"].to_array("channel")

            # for i_run, local_run in enumerate(local_runs):
            #     local_runs[i_run] = prototype_decimate(processing_config, local_run)
            # #                local_decimated.append(local_run)
            # #do we need this if statement?  Maybe better to just do the for loop and
            # # make prototype decimate accept {'run': None, 'mvts': None} and return
            # # {'run': None, 'mvts': None}
            # if processing_config.reference_station_id:
            #     for i_run, remote_run in enumerate(remote_runs):
            #         remote_runs[i_run] = prototype_decimate(processing_config, remote_run)

        # </GET TIME SERIES DATA>

        #ANY MERGING OF RUNS IN TIME DOMAIN WOULD GO HERE

        #<CONVERT TO STFT>
        local_stfts = []
        remote_stfts = []
        for i,row in dataset_df.iterrows():
            run_xrts = row["run_dataarray"].to_dataset("channel")
            stft_obj = make_stft_objects_new(processing_config, row["run"],
                                              run_xrts, units, row.station_id)

            if row.station_id == processing_config.local_station_id:
                local_stfts.append(stft_obj)
            elif row.station_id == processing_config.reference_station_id:
                remote_stfts.append(stft_obj)
            # all_stft_objs[i] = stft_obj
            # dataset_df["stft"].loc[i] = stft_obj.to_array("channel")


        # MERGE STFTS goes here
        print("merge-o-rama")
        local_merged_stft_obj = xr.concat(local_stfts, "time")
        # MUTE BAD FCs HERE - Not implemented yet.
        # RETURN FC_OBJECT

        if processing_config.reference_station_id:
            remote_merged_stft_obj = xr.concat(remote_stfts, "time")
        else:
            remote_merged_stft_obj = None

        tf_obj = process_tf_decimation_level(
            processing_config,
            local_merged_stft_obj,
            remote_merged_stft_obj,
            units=units
        )
        tf_dict[dec_level_id] = tf_obj

        if show_plot:
            from aurora.sandbox.plot_helpers import plot_tf_obj

            plot_tf_obj(tf_obj, out_filename="out")

    # TODO: Add run_obj to TransferFunctionCollection ? WHY? so it doesn't need header?
    tf_collection = TransferFunctionCollection(header=tf_obj.tf_header, tf_dict=tf_dict)

    #
    print("Need to review this info @Jared, review role of local_run_obj in export tf ")

    #local_run_obj = mth5_obj.get_run(run_config["local_station_id"], run_id)
    local_run_obj = dataset_df["run"].iloc[0]
    if z_file_path:
        tf_collection.write_emtf_z_file(z_file_path, run_obj=local_run_obj)

    if return_collection:
        return tf_collection
    else:
        # intended to be the default in future
        #
        # There is a container that can handle storage of multiple runs in xml
        # Anna made something like this.
        # N.B. Currently, only the last run makes it into the tf object,
        # but we can simply iterate of the run list here, getting run metadata
        # station_metadata.add_run(run_metadata)
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
