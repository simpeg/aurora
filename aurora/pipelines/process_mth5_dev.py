"""
Note 1: process_mth5_run assumes application of cascading decimation, and that the
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
from aurora.pipelines.time_series_helpers_new import calibrate_stft_obj
from aurora.pipelines.time_series_helpers_new import get_data_from_mth5_new
from aurora.pipelines.time_series_helpers_new import prototype_decimate
from aurora.pipelines.time_series_helpers_new import run_ts_to_calibrated_stft
from aurora.pipelines.time_series_helpers_new import run_ts_to_stft
from aurora.pipelines.transfer_function_helpers_new import process_transfer_functions
from aurora.pipelines.transfer_function_helpers_new import (
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


def initialize_pipeline(run_config):#, local_mth5_obj=None, remote_mth5_obj=None):
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
    # from aurora.config.processing_config import RunConfig
    # if isinstance(config, RunConfig):
    #     if mth5_path:
    #         if config["mth5_path"] != str(mth5_path):
    #             print(
    #                 "Warning - the mth5 path supplied to initialize pipeline differs"
    #                 "from the one in the config file"
    #             )
    #             print(f"config path changing from \n{config['mth5_path']} to \n{mth5_path}")
    #             config.mth5_path = str(mth5_path)
    #         mth5_obj = MTH5(file_version="0.1.0")
    #         mth5_obj.open_mth5(config["mth5_path"], mode="r")
    #         return config, mth5_obj
    # else:
    print("New mt_metadataa Processing class")
    local_mth5_obj = MTH5(file_version="0.1.0")
    local_mth5_obj.open_mth5(config.stations.local.mth5_path, mode="r")
    if config.stations.remote:
        remote_mth5_obj = MTH5(file_version="0.1.0")
        remote_mth5_obj.open_mth5(config.stations.remote[0].mth5_path, mode="r")
    else:
        remote_mth5_obj = None
    # </Initialize mth5 for reading>
    return config, local_mth5_obj, remote_mth5_obj


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



def make_stft_objects(processing_config, i_dec_level, run_obj, run_xrts, units,
                          station_id):
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
    config: processing config top level
    local
    remote

    Returns
    -------

    """
    print("OK - here we go ... ")
    stft_config = processing_config.get_decimation_level(i_dec_level)
    #stft_config = config.to_stft_config_dict() #another approach
    stft_obj = run_ts_to_stft(stft_config, run_xrts)

    print("OK")
    #still local and remote agnostic, it would be nice to be able to acess
    #p.stations[station_id]

    print("fix this so that it gets from config based on station_id, without caring "
          "if local or remote")
    run_id = run_obj.metadata.id
    if station_id==processing_config.stations.local.id:
        scale_factors = processing_config.stations.local.run_dict[
            run_id].channel_scale_factors
    #Need to add logic here to look through list of remote ids
    elif station_id==processing_config.stations.remote[0].id:
        scale_factors = processing_config.stations.remote[0].run_dict[
            run_id].channel_scale_factors
    # local_stft_obj = run_ts_to_stft_scipy(config, local_run_xrts)
    stft_obj = calibrate_stft_obj(
        stft_obj,
        run_obj,
        units=units,
        channel_scale_factors=scale_factors,
    )
    return stft_obj


def process_tf_decimation_level(config, i_dec_level, local_stft_obj,
                                    remote_stft_obj,
                                    units="MT"):
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
    frequency_bands = config.decimations[i_dec_level].frequency_bands_obj()
    transfer_function_header = transfer_function_header_from_config(config, i_dec_level)
    transfer_function_obj = TTFZ(
        transfer_function_header, frequency_bands, processing_config=config
    )

    transfer_function_obj = process_transfer_functions(
        config, i_dec_level, local_stft_obj, remote_stft_obj, transfer_function_obj
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


def populate_dataset_df(i_dec_level, config, dataset_df):
    """
    Could move this into a method of DatasetDefinition
    Called self.populate_with_data()
    Parameters
    ----------
    i_dec_level
    dataset_df
    config: aurora.config.metadata.decimation_level.DecimationLevel
        decimation level config

    Returns
    -------

    """
    """
    
    Returns
    -------

    """
    #<This will be replaced by a 1-line call issue #153>
    # processing_config = run_config.decimation_level_configs[dec_level_id]
    # processing_config.local_station_id = run_config.local_station_id
    # processing_config.reference_station_id = run_config.reference_station_id
    # processing_config.channel_scale_factors = run_config.channel_scale_factors
    #</This will be replaced by a 1-line call issue #153>


    # local_dataset_defn_df = defn_df[defn_df[
    #                                     "station_id"]==processing_config.local_station_id]
    # remote_dataset_defn_df = defn_df[defn_df[
    #                                      "station_id"]==processing_config.remote_station_id]
    # local_grouper = local_dataset_defn_df.groupby("run")
    # remote_grouper = remote_dataset_defn_df.groupby("run")
    # #grouper = df.groupby(["station", "run"])

    # <GET TIME SERIES DATA>
    #Factor this out into a function.
    #function takes an (mth5_obj_list, local_station_id, remote_station_id,
    # local_run_list, remote_run_list)
    #OR (mth5_obj_list, local_station_id, remote_station_id, dataset_df)
    #Note that the dataset_df should be easy to generate from the local_station_id,
    # remote_station_id, local_run_list, remote_run_list, but allows for
    # specification of time_intervals.  This is important in the case where
    # aquisition_runs are non-overlapping between local and remote.  Although,
    # theoretically, merging on the FCs should make nans in the places where
    # there is no overlapping data, and this should be dropped in the TF portion
    # of the code.  However, time-intervals where the data do not have coverage
    # at both stations can be identified in a method before GET TIME SERIES
    # in a future version.
    #get_time_series_data(dec_level_id, ...)

    all_run_objs = len(dataset_df) * [None]
    all_run_ts_objs = len(dataset_df) * [None]
    all_stft_objs = len(dataset_df) * [None] #need these bc df not taking assingments 


    if i_dec_level == 0:
        for i,row in dataset_df.iterrows():
            run_dict = get_data_from_mth5_new(row.mth5_obj,
                                              row.station_id,
                                              row.run_id,
                                              config.decimation.sample_rate)
            #dataset_df.loc[i]["run"] = run_dict["run"]
            dataset_df["run"].loc[i] = run_dict["run"] #value try to set??
            #dataset_df.loc[i]["run_dataarray"] = run_dict["mvts"].to_array("channel")
            dataset_df["run_dataarray"].loc[i] = run_dict["mvts"].to_array("channel")
            #Dataframe doesn't like an xarray Dataset in a cell, need to convert
            # to DataArray

            all_run_objs[i] = run_dict["run"]
            all_run_ts_objs[i] = run_dict["mvts"]
            #careful here, i must run from 0 to len(df), otherwise will get
            # indexing errors.  Maybe reset_index() before this loop?
            # or push reindexing into TF Kernel, so that this method only gets
            # a cleanly indexed df, restricted to only the runs to be processed for
            # this specific TF

            # APPLY TIMING CORRECTIONS HERE
    else:
        # See Note 1 top of module
        # See Note 2 top of module
        for i,row in dataset_df.iterrows():
            run_xrts = row["run_dataarray"].to_dataset("channel")
            input_dict = {"run":row["run"], "mvts":run_xrts}
            run_dict = prototype_decimate(config.decimation, input_dict)
            dataset_df["run"].loc[i] = run_dict["run"]
            dataset_df["run_dataarray"].loc[i] = run_dict["mvts"].to_array("channel")

    return dataset_df


    # </GET TIME SERIES DATA>

def close_mths_objs(df):
    """
    Loop over all unique mth5_objs in the df and make sure they are closed
    Parameters
    ----------
    df

    Returns
    -------

    """
    mth5_objs = df["mth5_obj"].unique()
    for mth5_obj in mth5_objs:
        mth5_obj.close_mth5()
    return


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
    
    processing_config, local_mth5_obj, remote_mth5_obj = initialize_pipeline(run_cfg)

    dataset_df = dataset_definition.df
    #</Move into TFKernel()>

    #move this into initialize_pipeline?  maybe even into TF Kernel?
    #Make a dict of the active mth5_objects (keyed by station_id? namespace clash?)
    mth5_objs = {processing_config.stations.local.id:local_mth5_obj}
    if processing_config.stations.remote:
        mth5_objs[processing_config.stations.remote[0].id] = remote_mth5_obj
    #flesh out dataset_df, populate the with mth5_objs
    all_mth5_objs = len(dataset_df) * [None]
    for i, station_id in enumerate(dataset_df["station_id"]):
        all_mth5_objs[i] = mth5_objs[station_id]
    dataset_df["mth5_obj"] = all_mth5_objs
    dataset_df["run"] = None
    dataset_df["run_dataarray"] = None
    dataset_df["stft"] = None
    #</MAKE SURE DATASET DEFINITION DF HAS "run", "run_ts" columns>

    print(f"config indicates {len(processing_config.decimations)} decimation levels ")

    tf_dict = {}

    # local_station_id = processing_config.stations.local.id #still need this?
    # if processing_config.stations.remote:
    #     remote_station_id = processing_config.stations.remote.id

    for i_dec_level, dec_level_config in enumerate(processing_config.decimations):
        dataset_df = populate_dataset_df(i_dec_level, dec_level_config, dataset_df)
        #ANY MERGING OF RUNS IN TIME DOMAIN WOULD GO HERE

        #<CONVERT TO STFT>
        local_stfts = []
        remote_stfts = []
        for i,row in dataset_df.iterrows():
            run_xrts = row["run_dataarray"].to_dataset("channel")
            print("The decimation_level_config here does not have the scale factors, "
                  "which are needed in make_stft_objects")
            stft_obj = make_stft_objects(processing_config, i_dec_level, row["run"],
                                              run_xrts, units, row.station_id) #stftconfig

            if row.station_id == processing_config.stations.local.id:#local_station_id:
                local_stfts.append(stft_obj)
            elif row.station_id == \
                    processing_config.stations.remote[0].id:#reference_station_id:
                remote_stfts.append(stft_obj)
            # all_stft_objs[i] = stft_obj
            # dataset_df["stft"].loc[i] = stft_obj.to_array("channel")


        # MERGE STFTS goes here
        print("merge-o-rama")

        local_merged_stft_obj = xr.concat(local_stfts, "time")
        # MUTE BAD FCs HERE - Not implemented yet.
        # RETURN FC_OBJECT

        if processing_config.stations.remote:#reference_station_id:
            remote_merged_stft_obj = xr.concat(remote_stfts, "time")
        else:
            remote_merged_stft_obj = None
        #</CONVERT TO STFT>

        tf_obj = process_tf_decimation_level(
            processing_config,
            i_dec_level,
            local_merged_stft_obj,
            remote_merged_stft_obj,
            units=units
        )

        tf_dict[i_dec_level] = tf_obj

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
        close_mths_objs(dataset_df)
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
        close_mths_objs(dataset_df)
        return tf_cls
