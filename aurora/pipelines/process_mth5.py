"""
This module contains the main methods used in processing mth5 objects to transfer functions.

The main function is called process_mth5.
This function was recently changed to process_mth5_legacy, os that process_mth5
can be repurposed for other TF estimation schemes.  The "legacy" version
corresponds to aurora default processing.


Notes on process_mth5_legacy:
Note 1: process_mth5 assumes application of cascading decimation, and that the
decimated data will be accessed from the previous decimation level.  This should be
revisited. It may make more sense to have a get_decimation_level() interface that
provides an option of applying decimation or loading pre-decimated data.
This will be addressed via creation of the FC layer inside mth5.

Note 2: We can encounter cases where some runs can be decimated and others can not.
We need a way to handle this. For example, a short run may not yield any data from a
later decimation level. An attempt to handle this has been made in TF Kernel by
adding a is_valid_dataset column, associated with each run-decimation level pair.


Note 3: This point in the loop marks the interface between _generation_ of the FCs and
 their _usage_. In future the code above this comment would be pushed into
 create_fourier_coefficients() and the code below this would access those FCs and
 execute compute_transfer_function().
 This would also be an appropriate place to place a feature extraction layer, and
 compute weights for the FCs.

"""

import mth5.groups

# =============================================================================
# Imports
# =============================================================================

from aurora.pipelines.time_series_helpers import calibrate_stft_obj
from aurora.pipelines.time_series_helpers import run_ts_to_stft
from aurora.pipelines.transfer_function_helpers import (
    process_transfer_functions,
)
from aurora.pipelines.transfer_function_kernel import TransferFunctionKernel
from aurora.pipelines.transfer_function_kernel import station_obj_from_row
from aurora.sandbox.triage_metadata import triage_run_id
from aurora.transfer_function.transfer_function_collection import (
    TransferFunctionCollection,
)
from aurora.transfer_function.TTFZ import TTFZ
from loguru import logger
from mth5.helpers import close_open_files
from typing import Optional, Union

import aurora.config.metadata.processing
import pandas as pd
import xarray as xr


SUPPORTED_PROCESSINGS = [
    "legacy",
]

# =============================================================================


def make_stft_objects(
    processing_config, i_dec_level, run_obj, run_xrds, units="MT"
):
    """
    Operates on a "per-run" basis.  Applies STFT to all time series in the input run.

    This method could be modified in a multiple station code so that it doesn't care
    if the station is "local" or "remote" but rather uses scale factors keyed by
    station_id (WIP - issue #329)

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
    if run_obj.station_metadata.id == processing_config.stations.local.id:
        scale_factors = processing_config.stations.local.run_dict[
            run_id
        ].channel_scale_factors
    elif run_obj.station_metadata.id == processing_config.stations.remote[0].id:
        scale_factors = (
            processing_config.stations.remote[0]
            .run_dict[run_id]
            .channel_scale_factors
        )

    stft_obj = calibrate_stft_obj(
        stft_obj,
        run_obj,
        units=units,
        channel_scale_factors=scale_factors,
    )
    return stft_obj


def process_tf_decimation_level(
    config: aurora.config.metadata.processing.Processing,
    i_dec_level: int,
    local_stft_obj: xr.core.dataset.Dataset,
    remote_stft_obj: Union[xr.core.dataset.Dataset, None],
    units="MT",
):
    """
    Processing pipeline for a single decimation_level

    TODO: Add a check that the processing config sample rates agree with the data
    TODO: Add units to local_stft_obj, remote_stft_obj
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
    transfer_function_obj = TTFZ(
        i_dec_level, frequency_bands, processing_config=config
    )
    dec_level_config = config.decimations[i_dec_level]
    # segment_weights = coherence_weights(dec_level_config, local_stft_obj, remote_stft_obj)
    transfer_function_obj = process_transfer_functions(
        dec_level_config, local_stft_obj, remote_stft_obj, transfer_function_obj
    )

    return transfer_function_obj


# def enrich_row(row):
#     pass


def triage_issue_289(local_stfts: list, remote_stfts: list):
    """
    Takes STFT objects in and returns them after shape-checking and making sure they are same.
    WIP:  Timing Error Workaround See Aurora Issue #289.
    Seems associated with getting one fewer sample than expected from the edge of a run.

    returns: Tuple
        (local_stfts, remote_stfts)
        The original input arguments, shape-matched

    """
    n_chunks = len(local_stfts)
    for i_chunk in range(n_chunks):
        ok = local_stfts[i_chunk].time.shape == remote_stfts[i_chunk].time.shape
        if not ok:
            logger.warning(
                "Mismatch in FC array lengths detected -- Issue #289"
            )
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
            local_stfts[i_chunk] = local_stfts[i_chunk].where(
                cond1 & cond2, drop=True
            )
            cond1 = remote_stfts[i_chunk].time >= glb
            cond2 = remote_stfts[i_chunk].time <= lub
            remote_stfts[i_chunk] = remote_stfts[i_chunk].where(
                cond1 & cond2, drop=True
            )
            assert (
                local_stfts[i_chunk].time.shape
                == remote_stfts[i_chunk].time.shape
            )
    return local_stfts, remote_stfts


def merge_stfts(stfts: dict, tfk: TransferFunctionKernel):
    """

    Applies concatenation along the time axis to multiple arrays of STFTs from different runs.
    At the TF estimation level we treat all the FCs in one array.
    This builds the array for both the local and the remote STFTs.

    Parameters
    ----------
    stfts: dict
        The dict is keyed by "local" and "remote".
        Each value is a list of STFTs (one list for local and one for remote)
    tfk: TransferFunctionKernel
        Just here to let us know if there is a remote reference to merge or not.

    Returns
    -------
    local_merged_stft_obj, remote_merged_stft_obj: Tuple
        Both are xr.Datasets
    """
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


def append_chunk_to_stfts(stfts: dict, chunk: xr.Dataset, remote: bool) -> dict:
    """
    Aggregate one STFT into a larger dictionary that tracks all the STFTs

    Parameters
    ----------
    stfts: dict
        has keys "local" and "remote".
    chunk: xr.Dataset
        The data to append to the dictionary
    remote: bool
        If True, append the chunk to stfts["remote"], else append to stfts["local"]

    Returns
    -------
    stfts: dict
        Same as input but now has new chunk appended to it.
    """
    if remote:
        stfts["remote"].append(chunk)
    else:
        stfts["local"].append(chunk)
    return stfts


# def load_spectrogram_from_station_object(station_obj, fc_group_id, fc_decimation_id):
#     """
#     Placeholder.  This could also be a method in mth5
#     Returns
#     -------
#
#     """
#     return station_obj.fourier_coefficients_group.get_fc_group(fc_group_id).get_decimation_level(fc_decimation_id)


def load_stft_obj_from_mth5(
    i_dec_level: int,
    row: pd.Series,
    run_obj: mth5.groups.RunGroup,
    channels: Optional[list] = None,
) -> xr.Dataset:
    """
    Load stft_obj from mth5 (instead of compute)

    Note #1: See note #1 in time_series.frequency_band_helpers.extract_band

    Parameters
    ----------
    i_dec_level: int
        The decimation level where the data are stored within the Fourier Coefficient group
    row: pandas.core.series.Series
        A row of the TFK.dataset_df
    run_obj: mth5.groups.run.RunGroup
        The original time-domain run associated with the data to load

    Returns
    -------
    stft_chunk: xr.Dataset
        An STFT from mth5.
    """
    station_obj = station_obj_from_row(row)
    fc_group = station_obj.fourier_coefficients_group.get_fc_group(
        run_obj.metadata.id
    )
    fc_decimation_level = fc_group.get_decimation_level(f"{i_dec_level}")
    stft_obj = fc_decimation_level.to_xarray(channels=channels)

    cond1 = stft_obj.time >= row.start.tz_localize(None)
    cond2 = stft_obj.time <= row.end.tz_localize(None)
    try:
        stft_chunk = stft_obj.where(cond1 & cond2, drop=True)
    except TypeError:  # see Note #1
        tmp = stft_obj.to_array()
        tmp = tmp.where(cond1 & cond2, drop=True)
        stft_chunk = tmp.to_dataset("variable")
    return stft_chunk


def save_fourier_coefficients(dec_level_config, row, run_obj, stft_obj) -> None:
    """
    Optionally saves the stft object into the MTH5.
    Note that the dec_level_config must have its save_fcs attr set to True to actually save the data.
    WIP

    Note #1: Logic for building FC layers:
    If the processing config decimation_level.save_fcs_type = "h5" and fc_levels_already_exist is False, then open
    in append mode, else open in read mode.  We should support a flag: force_rebuild_fcs, normally False.  This flag
    is only needed when save_fcs_type=="h5".  If True, then we open in append mode, regarless of fc_levels_already_exist
    The task of setting mode="a", mode="r" can be handled by tfk (maybe in tfk.validate())

    Parameters
    ----------
    dec_level_config: mt_metadata.transfer_functions.processing.aurora.decimation_level.DecimationLevel
        The information about decimation level associated with row, run, stft_obj
    row: pd.Series
         A row of the TFK.dataset_df
    run_obj: mth5.groups.run.RunGroup
        The run object associated with the STFTs.
    stft_obj: xr.Dataset
        The data to pack intp the mth5.

    """

    if not dec_level_config.save_fcs:
        msg = "Skip saving FCs. dec_level_config.save_fc = "
        msg = f"{msg} {dec_level_config.save_fcs}"
        logger.info(f"{msg}")
        return
    i_dec_level = dec_level_config.decimation.level
    if dec_level_config.save_fcs_type == "csv":
        msg = "Unless debugging or testing, saving FCs to csv unexpected"
        logger.warning(msg)
        csv_name = f"{row.station}_dec_level_{i_dec_level}.csv"
        stft_df = stft_obj.to_dataframe()
        stft_df.to_csv(csv_name)
    elif dec_level_config.save_fcs_type == "h5":
        logger.info(("Saving FC level"))
        station_obj = station_obj_from_row(row)

        if not row.mth5_obj.h5_is_write():
            msg = "See Note #1: Logic for building FC layers"
            raise NotImplementedError(msg)

        # Get FC group (create if needed)
        if (
            run_obj.metadata.id
            in station_obj.fourier_coefficients_group.groups_list
        ):
            fc_group = station_obj.fourier_coefficients_group.get_fc_group(
                run_obj.metadata.id
            )
        else:
            fc_group = station_obj.fourier_coefficients_group.add_fc_group(
                run_obj.metadata.id
            )

        decimation_level_metadata = dec_level_config.to_fc_decimation()

        # Get FC Decimation Level (create if needed)
        dec_level_name = f"{i_dec_level}"
        if dec_level_name in fc_group.groups_list:
            fc_decimation_level = fc_group.get_decimation_level(dec_level_name)
            fc_decimation_level.metadata = decimation_level_metadata
        else:
            fc_decimation_level = fc_group.add_decimation_level(
                dec_level_name,
                decimation_level_metadata=decimation_level_metadata,
            )
        fc_decimation_level.from_xarray(
            stft_obj, decimation_level_metadata.sample_rate
        )
        fc_decimation_level.update_metadata()
        fc_group.update_metadata()
    else:
        msg = f"dec_level_config.save_fcs_type {dec_level_config.save_fcs_type} not recognized"
        msg = f"{msg} \n Skipping save fcs"
        logger.warning(msg)
    return


def get_spectrogams(tfk, i_dec_level, units="MT"):
    """
    Given a decimation level id, loads a dictianary of all spectragrams from information in tfk.
    TODO: Make this a method of TFK

    Parameters
    ----------
    tfk: TransferFunctionKernel

    i_dec_level: integer
        The decimation level of the spectrograms.

    units: str
        "MT" or "SI", likely to be deprecated

    Returns
    -------
    stfts: dict
        The short time fourier transforms for the decimation level as a dictionary.
        Keys are "local" and "remote".  Values are lists, one (element) xr.Dataset per run
    """

    stfts = {}
    stfts["local"] = []
    stfts["remote"] = []

    # Check first if TS processing or accessing FC Levels
    for i, row in tfk.dataset_df.iterrows():
        # This iterator could be updated to iterate over row-pairs if remote is True,
        # corresponding to simultaneous data

        if not tfk.is_valid_dataset(row, i_dec_level):
            continue

        run_obj = row.mth5_obj.from_reference(row.run_hdf5_reference)
        if row.fc:
            stft_obj = load_stft_obj_from_mth5(i_dec_level, row, run_obj)
            stfts = append_chunk_to_stfts(stfts, stft_obj, row.remote)
            continue

        run_xrds = row["run_dataarray"].to_dataset("channel")

        # Musgraves workaround for old MT data
        triage_run_id(row.run, run_obj)

        stft_obj = make_stft_objects(
            tfk.config,
            i_dec_level,
            run_obj,
            run_xrds,
            units,
        )

        # Pack FCs into h5
        dec_level_config = tfk.config.decimations[i_dec_level]
        save_fourier_coefficients(dec_level_config, row, run_obj, stft_obj)
        stfts = append_chunk_to_stfts(stfts, stft_obj, row.remote)

    return stfts


def process_mth5_legacy(
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
    tf_collection: TransferFunctionCollection or mt_metadata TF
        The transfer function object
    tf_cls: mt_metadata.transfer_functions.TF
        TF object
    """
    # Initialize config and mth5s
    tfk = TransferFunctionKernel(dataset=tfk_dataset, config=config)
    tfk.make_processing_summary()
    tfk.show_processing_summary()
    tfk.validate()

    tfk.initialize_mth5s()

    msg = (
        f"Processing config indicates {len(tfk.config.decimations)} "
        f"decimation levels"
    )
    logger.info(msg)
    tf_dict = {}

    for i_dec_level, dec_level_config in enumerate(tfk.valid_decimations()):
        # if not tfk.all_fcs_already_exist():
        tfk.update_dataset_df(i_dec_level)
        tfk.apply_clock_zero(dec_level_config)

        stfts = get_spectrogams(tfk, i_dec_level, units=units)

        local_merged_stft_obj, remote_merged_stft_obj = merge_stfts(stfts, tfk)

        # FC TF Interface here (see Note #3)
        # Feature Extraction, Selection of weights

        ttfz_obj = process_tf_decimation_level(
            tfk.config,
            i_dec_level,
            local_merged_stft_obj,
            remote_merged_stft_obj,
        )
        ttfz_obj.apparent_resistivity(
            tfk.config.channel_nomenclature, units=units
        )
        tf_dict[i_dec_level] = ttfz_obj

        if show_plot:
            from aurora.sandbox.plot_helpers import plot_tf_obj

            plot_tf_obj(ttfz_obj, out_filename="")

    tf_collection = TransferFunctionCollection(
        tf_dict=tf_dict, processing_config=tfk.config
    )

    tf_cls = tfk.export_tf_collection(tf_collection)

    if z_file_path:
        tf_cls.write(z_file_path)

    tfk.dataset.close_mth5s()
    if return_collection:
        # this is now really only to be used for debugging and may be deprecated soon
        return tf_collection
    else:
        return tf_cls


def process_mth5(
    config,
    tfk_dataset=None,
    units="MT",
    show_plot=False,
    z_file_path=None,
    return_collection=False,
    processing_type="legacy",
):
    """
    This is a pass-through method that routes the config and tfk_dataset to MT data processing.
    It currently only supports legacy aurora processing.

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
    processing_type: string
        Controlled vocabulary, must be one of ["legacy",]
        This is not really supported now, but the idea is that in future, the config and tfk_dataset can be passed to
        another processing method if desired.

    Returns
    -------
    tf_obj: TransferFunctionCollection or mt_metadata.transfer_functions.TF
        The transfer function object
    """
    if processing_type not in SUPPORTED_PROCESSINGS:
        raise NotImplementedError(
            f"Processing type {processing_type} not supported"
        )

    if processing_type == "legacy":
        try:
            return process_mth5_legacy(
                config,
                tfk_dataset=tfk_dataset,
                units=units,
                show_plot=show_plot,
                z_file_path=z_file_path,
                return_collection=return_collection,
            )
        except Exception as e:
            close_open_files()
            msg = "Failed to run legacy processing\n"
            msg += "closing all open mth5 files and exiting"
            msg += f"The encountered exception was {e}"
            logger.error(msg)
            logger.exception(msg)
            return
