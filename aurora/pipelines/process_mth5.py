"""
Process an MTH5 using the metdata config object.

Note 1: process_mth5 assumes application of cascading decimation, and that the
decimated data will be accessed from the previous decimation level.  This should be
revisited. It may make more sense to have a get_decimation_level() interface that
provides an option of applying decimation or loading predecimated data.
This will be addressed via creation of the FC layer inside mth5.

Note 2: We can encounter cases where some runs can be decimated and others can not.
We need a way to handle this. For example, a short run may not yield any data from a
later decimation level. An attempt to handle this has been made in TF Kernel by
adding a is_valid_dataset column, associated with each run-decimation level pair.


Note 3: This point in the loop marks the interface between _generation_ of the FCs and
 their _usage_. In future the code above this comment would be pushed into
 create_fourier_coefficients() and the code below this would access those FCs and
 execute compute_transfer_function()

"""
# =============================================================================
# Imports
# =============================================================================

import xarray as xr

from aurora.pipelines.time_series_helpers import calibrate_stft_obj
from aurora.pipelines.time_series_helpers import run_ts_to_stft
from aurora.pipelines.transfer_function_helpers import process_transfer_functions
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
    processing_config: mt_metadata.transfer_functions.processing.aurora.Processing
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

def station_obj_from_row(row):
    """
    Access the station object
    Note if/else could avoidable if replacing text string "none" with a None object in survey column

    Parameters
    ----------
    row: pd.Series
        A row of tfk.dataset_df

    Returns
    -------
    station_obj:

    """
    if row.survey == "none":
        station_obj = row.mth5_obj.stations_group.get_station(row.station_id)
    else:
        station_obj = row.mth5_obj.stations_group.get_station(
            row.station_id, survey=row.survey
        )
    return station_obj

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
    config: mt_metadata.transfer_functions.processing.aurora.decimation_level.DecimationLevel
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
    transfer_function_obj = TTFZ(i_dec_level, frequency_bands, processing_config=config)

    transfer_function_obj = process_transfer_functions(
        config, i_dec_level, local_stft_obj, remote_stft_obj, transfer_function_obj
    )

    return transfer_function_obj


def enrich_row(row):
    pass


def triage_issue_289(local_stfts, remote_stfts):
    """
    Timing Error Workaround See Aurora Issue #289: seems associated with getting one fewer sample than expected
    from the edge of a run.
    """
    n_chunks = len(local_stfts)
    for i_chunk in range(n_chunks):
        ok = local_stfts[i_chunk].time.shape == remote_stfts[i_chunk].time.shape
        if not ok:
            print("Mismatch in FC array lengths detected -- Issue #289")
            glb = max(
                local_stfts[i_chunk].time.min(),
                remote_stfts[i_chunk].time.min(),
            )
            lub = min(
                local_stfts[i_chunk].time.max(),
                remote_stfts[i_chunk].time.max(),
            )
            cond1 = local_stfts[i_chunk].time >= glb
            cond2 = local_stfts[i_chunk].time <= lub
            local_stfts[i_chunk] = local_stfts[i_chunk].where(cond1 & cond2, drop=True)
            cond1 = remote_stfts[i_chunk].time >= glb
            cond2 = remote_stfts[i_chunk].time <= lub
            remote_stfts[i_chunk] = remote_stfts[i_chunk].where(
                cond1 & cond2, drop=True
            )
            assert local_stfts[i_chunk].time.shape == remote_stfts[i_chunk].time.shape
    return local_stfts, remote_stfts


def merge_stfts(stfts, tfk):
    # Timing Error Workaround See Aurora Issue #289
    local_stfts = stfts["local"]
    remote_stfts = stfts["remote"]
    if tfk.config.stations.remote:
        local_stfts, remote_stfts = triage_issue_289(local_stfts, remote_stfts)
        remote_merged_stft_obj = xr.concat(remote_stfts, "time")
    else:
        remote_merged_stft_obj = None

    local_merged_stft_obj = xr.concat(local_stfts, "time")

    return local_merged_stft_obj, remote_merged_stft_obj


def append_chunk_to_stfts(stfts, chunk, remote):
    if remote:
        stfts["remote"].append(chunk)
    else:
        stfts["local"].append(chunk)
    return stfts


def load_stft_obj_from_mth5(i_dec_level, row, run_obj):
    # Load stft_obj from mth5 (instead of compute)
    station_obj = station_obj_from_row(row)
    fc_group = station_obj.fourier_coefficients_group.add_fc_group(
        run_obj.metadata.id
    )
    fc_decimation_level = fc_group.get_decimation_level(f"{i_dec_level}")
    stft_obj = fc_decimation_level.to_xarray()
    return stft_obj


def save_fourier_coefficients(dec_level_config, i_dec_level, row, run_obj, stft_obj):
    if not dec_level_config.save_fcs:
        msg = "Skip saving FCs. dec_level_config.save_fc = "
        print(f"{msg} {dec_level_config.save_fcs}")
        return
    if dec_level_config.save_fcs_type == "csv":
        msg = "Unless debugging or testing, saving FCs to csv unexpected"
        print(f"WARNING: {msg}")
        csv_name = f"{row.station_id}_dec_level_{i_dec_level}.csv"
        stft_df = stft_obj.to_dataframe()
        stft_df.to_csv(csv_name)
    elif dec_level_config.save_fcs_type == "h5":
        station_obj = station_obj_from_row(row)

        # See Note #1 at top this method (not module)
        if not row.mth5_obj.h5_is_write():
            raise NotImplementedError("See Note #1 at top this method")
        fc_group = station_obj.fourier_coefficients_group.add_fc_group(
            run_obj.metadata.id
        )
        decimation_level_metadata = dec_level_config.to_fc_decimation()
        fc_decimation_level = fc_group.add_decimation_level(
            f"{i_dec_level}",
            decimation_level_metadata=decimation_level_metadata,
        )
        fc_decimation_level.from_xarray(stft_obj)
        fc_decimation_level.update_metadata()
        fc_group.update_metadata()
    return

def process_mth5(
    config,
    tfk_dataset=None,
    units="MT",
    show_plot=False,
    z_file_path=None,
    return_collection=False,
):
    """
    This is the main method used to transform a processing_config,
    and a kernel_dataset into a transfer function estimate.

    Note 1: Logic for building FC layers:
    If the processing config decimation_level.save_fcs_type = "h5" and fc_levels_already_exist is False, then open
    in append mode, else open in read mode.  We should support a flag: force_rebuild_fcs, normally False.  This flag
    is only needed when save_fcs_type=="h5".  If True, then we open in append mode, regarless of fc_levels_already_exist
    The task of setting mode="a", mode="r" can be handled by tfk (maybe in tfk.validate())


    Parameters
    ----------
    config: mt_metadata.transfer_functions.processing.aurora.Processing or path to json
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
    # See Note #1
    tfk.initialize_mth5s(mode="a")
    tfk.check_if_fc_levels_already_exist()  # populate the "fc" column of dataset_df
    print(f"fc_levels_already_exist = {tfk.dataset_df['fc']}")
    print(
        f"Processing config indicates {len(tfk.config.decimations)} "
        f"decimation levels "
    )

    tf_dict = {}

    for i_dec_level, dec_level_config in enumerate(tfk.valid_decimations()):
        #i_dec_level = dec_level_config.decimation.level
        tfk.update_dataset_df(i_dec_level)
        dec_level_config = tfk.apply_clock_zero(dec_level_config)

        stfts = {}
        stfts["local"] = []
        stfts["remote"] = []

        # Check first if TS processing or accessing FC Levels
        for i, row in tfk.dataset_df.iterrows():
            # This iterator could be updated to iterate over row-pairs if remote is True, corresponding to simultaneous data

            if not tfk.is_valid_dataset(row, i_dec_level):
                continue

            run_obj = row.mth5_obj.from_reference(row.run_reference)
            if row.fc:
                stft_obj = load_stft_obj_from_mth5(i_dec_level, row, run_obj)
                stfts = append_chunk_to_stfts(stfts, stft_obj, row.remote)
                continue

            run_xrds = row["run_dataarray"].to_dataset("channel")
            run_obj = row.mth5_obj.from_reference(row.run_reference)
            stft_obj = make_stft_objects(
                tfk.config, i_dec_level, run_obj, run_xrds, units, row.station_id
            )
            # Pack FCs into h5
            save_fourier_coefficients(dec_level_config, i_dec_level, row, run_obj, stft_obj)

            stfts = append_chunk_to_stfts(stfts, stft_obj, row.remote)

        local_merged_stft_obj, remote_merged_stft_obj = merge_stfts(stfts, tfk)

        # FC TF Interface here (see Note #3)
        # Could downweight bad FCs here

        ttfz_obj = process_tf_decimation_level(
            tfk.config,
            i_dec_level,
            local_merged_stft_obj,
            remote_merged_stft_obj,
        )
        ttfz_obj.apparent_resistivity(tfk.config.channel_nomenclature, units=units)
        tf_dict[i_dec_level] = ttfz_obj

        if show_plot:
            from aurora.sandbox.plot_helpers import plot_tf_obj

            plot_tf_obj(ttfz_obj, out_filename="out")

    tf_collection = TransferFunctionCollection(
        tf_dict=tf_dict, processing_config=tfk.config
    )

    tf_cls = tfk.export_tf_collection(tf_collection)

    if z_file_path:
        tf_cls.write(z_file_path)

    tfk.dataset.close_mths_objs()
    if return_collection:
        # this is now really only to be used for debugging and may be deprecated soon
        return tf_collection
    else:
        return tf_cls
