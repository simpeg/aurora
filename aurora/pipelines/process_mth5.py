"""
Process an MTH5 using the metdata config object.

Note 1: process_mth5 assumes application of cascading decimation, and that the
decimated data will be accessed from the previous decimation level.  This should be
revisited. It may make more sense to have a get_decimation_level() interface that
provides an option of applying decimation or loading predecimated data.

#Note 2: Question: Can we run into cases where some of these runs can be decimated
and others can not?  We need a way to handle this. For example, a short run may not
yield any data from a later decimation level. Returning an empty xarray may work.
It is desireable that the empty xarray, or whatever comes back to pass through STFT
and MERGE steps smoothy. If it is empty, we could just drop the run entirely but
this may have adverse consequences on downstream bookkeeping, like the creation of
station_metadata before packaging the tf for export. This could be worked around by
extracting the metadata at the start of this method. It would be a good idea in
general to run a pre-check on the data that identifies which decimation levels are
valid for each run. (see Issue #182)

"""
# =============================================================================
# Imports
# =============================================================================

import xarray as xr

from aurora.pipelines.time_series_helpers import calibrate_stft_obj
from aurora.pipelines.time_series_helpers import prototype_decimate
from aurora.pipelines.time_series_helpers import run_ts_to_stft
from aurora.pipelines.transfer_function_helpers import process_transfer_functions
from aurora.pipelines.transfer_function_helpers import tf_header_from_config
from aurora.pipelines.transfer_function_kernel import TransferFunctionKernel

from aurora.transfer_function.transfer_function_collection import (
    TransferFunctionCollection,
)
from aurora.transfer_function.TTFZ import TTFZ

from mt_metadata.transfer_functions.core import TF


# =============================================================================


def make_stft_objects(
    processing_config, i_dec_level, run_obj, run_xrds, units, station_id
):
    """
    Operates on a "per-run" basis

    This method could be modifed in a multiple station code so that it doesn't care
    if the station is "local" or "remote" but rather uses scale factors keyed by
    station_id

    Parameters
    ----------
    processing_config: aurora.config.metadata.processing.Processing
        Metadata about the processing to be applied
    i_dec_level: int
        The decimation level to process
    run_obj: mth5.groups.master_station_run_channel.RunGroup
        The run to transform to stft
    run_xrds: xarray.core.dataset.Dataset
        The data time series from the run to transform
    units: str
        expects "MT".  May change so that this is the only accepted set of units
    station_id: str
        To be deprecated, this information is contained in the run_obj as
        run_obj.station_group.metadata.id

    Returns
    -------
    stft_obj: xarray.core.dataset.Dataset
        Time series of calibrated Fourier coefficients per each channel in the run
    """
    stft_config = processing_config.get_decimation_level(i_dec_level)
    stft_obj = run_ts_to_stft(stft_config, run_xrds)
    # if stft_obj is None:
    #    # not enough data to FFT
    #    return stft_obj
    # stft_obj = run_ts_to_stft_scipy(stft_config, run_xrds)
    run_id = run_obj.metadata.id
    if station_id == processing_config.stations.local.id:
        scale_factors = processing_config.stations.local.run_dict[
            run_id
        ].channel_scale_factors
    elif station_id == processing_config.stations.remote[0].id:
        scale_factors = (
            processing_config.stations.remote[0].run_dict[run_id].channel_scale_factors
        )

    stft_obj = calibrate_stft_obj(
        stft_obj,
        run_obj,
        units=units,
        channel_scale_factors=scale_factors,
    )
    return stft_obj


def process_tf_decimation_level(
    config, i_dec_level, local_stft_obj, remote_stft_obj, units="MT"
):
    """
    Processing pipeline for a single decimation_level

    TODO: Add a check that the processing config sample rates agree with the data
    sampling rates otherwise raise Exception
    This method can be single station or remote based on the process cfg

    Parameters
    ----------
    config: aurora.config.metadata.decimation_level.DecimationLevel
        Config for a single decimation level
    i_dec_level: int
        decimation level_id
        ?could we pack this into the decimation level as an attr?
    local_stft_obj: xarray.core.dataset.Dataset
        The time series of Fourier coefficients from the local station
    remote_stft_obj: xarray.core.dataset.Dataset or None
        The time series of Fourier coefficients from the remote station
    units: str
        one of ["MT","SI"]

    Returns
    -------
    transfer_function_obj : aurora.transfer_function.TTFZ.TTFZ
        The transfer function values packed into an object
    """
    frequency_bands = config.decimations[i_dec_level].frequency_bands_obj()
    tf_header = tf_header_from_config(config, i_dec_level)
    transfer_function_obj = TTFZ(tf_header, frequency_bands, processing_config=config)

    transfer_function_obj = process_transfer_functions(
        config, i_dec_level, local_stft_obj, remote_stft_obj, transfer_function_obj
    )

    return transfer_function_obj


def export_tf(
    tf_collection,
    channel_nomenclature,
    station_metadata_dict={},
    survey_dict={},
):
    """
    This method may wind up being embedded in the TF class
    Assign transfer_function, residual_covariance, inverse_signal_power, station, survey

    Parameters
    ----------
    tf_collection: aurora.transfer_function.transfer_function_collection
    .TransferFunctionCollection
    station_metadata_dict: dict
    survey_dict: dict

    Returns
    -------
    tf_cls: mt_metadata.transfer_functions.core.TF
        Transfer function container
    """
    merged_tf_dict = tf_collection.get_merged_dict(channel_nomenclature)
    channel_nomenclature_dict = channel_nomenclature.to_dict()["channel_nomenclature"]
    tf_cls = TF(channel_nomenclature=channel_nomenclature_dict)
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


def update_dataset_df(i_dec_level, config, dataset_df, tfk=None):
    """
    This function has two different modes.  The first mode, initializes values in the
    array, and could be placed into TFKDataset.initialize_time_series_data()
    The second mode, decimates. Becasue it calls time series operations, I prefer
    to keep it in pipelines.  Maybe name it update_dataset_df().
    Although, there are other ways to approach this:
        Say we had a dictionary of decimation_methods (kind of like the dictionary of
        regression_engines), then one could call TFKernel.update_dataset_df(), where
        the argument was the method to do the decimation.
        In this way, the workflow management could stay inside TFKernel, but the math
        could be done via the data container ...

    Notes:
    1. When iterating over dataframe, (i)ndex must run from 0 to len(df), otherwise
    get indexing errors.  Maybe reset_index() before main loop? or push reindexing
    into TF Kernel, so that this method only gets a cleanly indexed df, restricted to
    only the runs to be processed for this specific TF?
    2. When assigning xarrays to dataframe cells, df dislikes xr.Dataset,
    so we convert to DataArray before assignment


    Parameters
    ----------
    i_dec_level: int
        decimation level id, indexed from zero
    config: aurora.config.metadata.decimation_level.DecimationLevel
        decimation level config
    dataset_df: pd.DataFrame

    Returns
    -------
    dataset_df: pd.DataFrame
        Same df that was input to the function but now has columns:


    """
    if i_dec_level == 0:
        pass
        # replaced with kernel_dataset.initialize_dataframe_for_processing()

        # APPLY TIMING CORRECTIONS HERE
    else:
        print(f"DECIMATION LEVEL {i_dec_level}")
        # See Note 1 top of module
        # See Note 2 top of module
        for i, row in dataset_df.iterrows():
            run_xrds = row["run_dataarray"].to_dataset("channel")
            decimated_run_xrds = prototype_decimate(config.decimation, run_xrds)
            dataset_df["run_dataarray"].at[i] = decimated_run_xrds.to_array("channel")
    print("DATASET DF UPDATED")
    return dataset_df


def process_mth5(
    config,
    tfk_dataset=None,
    units="MT",
    show_plot=False,
    z_file_path=None,
    return_collection=False,
):
    """
    1. Read in the config and figure out how many decimation levels there are
    2. ToDo TFK: Based on the run durations, and sampling rates, determined which runs
    are valid for which decimation levels, or for which effective sample rates.  This
    action should be taken before we get here.  The tfk_dataset should already
    be trimmed to exactly what will be processed.


    Parameters
    ----------
    config: aurora.config.metadata.processing.Processing or path to json
        All processing parameters
    tfk_dataset: aurora.tf_kernel.dataset.Dataset or None
        Specifies what datasets to process according to config
    units: string
        "MT" or "SI".  To be deprecated once data have units embedded
    show_plot: boolean
        Only used for dev
    z_file_path: string or pathlib.Path
        Target path for a z_file output if desired
    return_collection : boolean
        return_collection=False will return an mt_metadata TF object
        return_collection=True will return
        aurora.transfer_function.transfer_function_collection.TransferFunctionCollection

    Returns
    -------
    tf: TransferFunctionCollection or mt_metadata TF
        The transfer funtion object
    tf_cls: mt_metadata.transfer_functions.TF
        TF object
    """
    # Initialize config and mth5s
    tfk = TransferFunctionKernel(dataset=tfk_dataset, config=config)
    tfk.make_processing_summary()
    tfk.validate()
    mth5_objs = tfk.config.initialize_mth5s()

    # Assign additional columns to dataset_df, populate with mth5_objs and xr_ts
    # ANY MERGING OF RUNS IN TIME DOMAIN WOULD GO HERE
    tfk_dataset.initialize_dataframe_for_processing(mth5_objs)
    dataset_df = tfk_dataset.df

    # Place checks that would be done by TF Kernel here
    # see notes labelled with ToDo TFK above

    print(
        f"Processing config indicates {len(tfk.config.decimations)} "
        f"decimation levels "
    )

    tf_dict = {}

    for i_dec_level, dec_level_config in enumerate(tfk.valid_decimations()):
        dataset_df = update_dataset_df(i_dec_level, dec_level_config, dataset_df)

        # TFK 1: get clock-zero from data if needed
        if dec_level_config.window.clock_zero_type == "data start":
            dec_level_config.window.clock_zero = str(dataset_df.start.min())

        # Apply STFT to all runs
        local_stfts = []
        remote_stfts = []
        for i, row in dataset_df.iterrows():
            run_xrds = row["run_dataarray"].to_dataset("channel")
            run_obj = row.mth5_obj.from_reference(row.run_reference)
            stft_obj = make_stft_objects(
                tfk.config, i_dec_level, run_obj, run_xrds, units, row.station_id
            )

            if row.station_id == tfk.config.stations.local.id:
                local_stfts.append(stft_obj)
            elif row.station_id == tfk.config.stations.remote[0].id:
                remote_stfts.append(stft_obj)

        # Merge STFTs
        local_merged_stft_obj = xr.concat(local_stfts, "time")

        if tfk.config.stations.remote:
            remote_merged_stft_obj = xr.concat(remote_stfts, "time")
        else:
            remote_merged_stft_obj = None

        # Could mute bad FCs here - Not implemented yet.

        tf_obj = process_tf_decimation_level(
            tfk.config,
            i_dec_level,
            local_merged_stft_obj,
            remote_merged_stft_obj,
        )
        tf_obj.apparent_resistivity(tfk.config.channel_nomenclature, units=units)
        tf_dict[i_dec_level] = tf_obj

        if show_plot:
            from aurora.sandbox.plot_helpers import plot_tf_obj

            plot_tf_obj(tf_obj, out_filename="out")

    # TODO: Add run_obj to TransferFunctionCollection so it doesn't need header?
    tf_collection = TransferFunctionCollection(header=tf_obj.tf_header, tf_dict=tf_dict)

    # local_run_obj = mth5_obj.get_run(run_config["local_station_id"], run_id)
    local_run_obj = tfk_dataset.get_run_object(0)

    if z_file_path:
        tf_collection.write_emtf_z_file(z_file_path, run_obj=local_run_obj)

    if return_collection:
        # this is now really only to be used for debugging and may be deprecated soon
        tfk_dataset.close_mths_objs()
        return tf_collection
    else:
        local_station_id = tfk.config.stations.local.id
        station_metadata = tfk_dataset.get_station_metadata(local_station_id)
        local_mth5_obj = mth5_objs[local_station_id]

        if local_mth5_obj.file_version == "0.1.0":
            survey_dict = local_mth5_obj.survey_group.metadata.to_dict()
        elif local_mth5_obj.file_version == "0.2.0":
            # this could be a method of tf_kernel.get_survey_dict()
            survey_id = dataset_df[
                dataset_df["station_id"] == local_station_id
            ].survey.unique()[0]
            survey_obj = local_mth5_obj.get_survey(survey_id)
            survey_dict = survey_obj.metadata.to_dict()

        tf_cls = export_tf(
            tf_collection,
            tfk.config.channel_nomenclature,
            station_metadata_dict=station_metadata.to_dict(),
            survey_dict=survey_dict,
        )
        tfk_dataset.close_mths_objs()
        return tf_cls
