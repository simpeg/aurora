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
 the creation of the spectrograms and the code below this would access those FCs and
 execute compute_transfer_function().
 This would also be an appropriate place to place a feature extraction layer, and
 compute weights for the FCs.

"""

import mth5.groups

# =============================================================================
# Imports
# =============================================================================
from aurora.pipelines.feature_weights import calculate_weights
from aurora.pipelines.feature_weights import extract_features
from aurora.pipelines.transfer_function_helpers import (
    process_transfer_functions,
    process_transfer_functions_with_weights,
)
from aurora.pipelines.transfer_function_kernel import TransferFunctionKernel
from aurora.time_series.spectrogram_helpers import get_spectrograms
from aurora.time_series.spectrogram_helpers import merge_stfts
from aurora.transfer_function.transfer_function_collection import (
    TransferFunctionCollection,
)
from aurora.transfer_function.TTFZ import TTFZ

from loguru import logger
from mth5.helpers import close_open_files
from mth5.processing import KernelDataset
from typing import Literal, Optional, Tuple, Union

import aurora.config.metadata.processing
import pandas as pd
import xarray as xr

SUPPORTED_PROCESSINGS = [
    "legacy",
]

# =============================================================================


def process_tf_decimation_level(
    config: aurora.config.metadata.processing.Processing,
    i_dec_level: int,
    local_stft_obj: xr.core.dataset.Dataset,
    remote_stft_obj: Union[xr.core.dataset.Dataset, None],
    weights: Optional[
        Tuple[str]
    ] = None,  # TODO: Make this a Literal of SupportedWeights
    units="MT",
):
    """
    Processing pipeline for a single decimation_level

    TODO: Add a check that the processing config sample rates agree with the data
     sampling rates otherwise raise Exception
    TODO: Add units to local_stft_obj, remote_stft_obj
    TODO: This is the method that should be accessing weights
    This method can be single station or remote based on the process cfg

    Parameters
    ----------
    config: aurora.config.metadata.processing.Processing,
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
    dec_level_config = config.decimations[i_dec_level]

    try:
        transfer_function_obj = process_transfer_functions_with_weights(
            dec_level_config=dec_level_config,
            local_stft_obj=local_stft_obj,
            remote_stft_obj=remote_stft_obj,
            transfer_function_obj=transfer_function_obj,
        )
    except Exception as e:
        msg = (
            f"Processing transfer functions with weights failed for decimation level {i_dec_level} "
            f"with exception: {e}"
        )
        logger.warning(msg)
        transfer_function_obj = process_transfer_functions(
            dec_level_config=dec_level_config,
            local_stft_obj=local_stft_obj,
            remote_stft_obj=remote_stft_obj,
            transfer_function_obj=transfer_function_obj,
        )
    return transfer_function_obj


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
    tfk.update_processing_summary()
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
        tfk.update_dataset_df(i_dec_level)  # TODO: could add noise here
        tfk.apply_clock_zero(dec_level_config)

        stfts = get_spectrograms(tfk, i_dec_level, units=units)

        local_merged_stft_obj, remote_merged_stft_obj = merge_stfts(stfts, tfk)

        # FC TF Interface here (see Note #3)
        try:
            # Feature Extraction, Selection of weights
            extract_features(dec_level_config, tfk_dataset)
            calculate_weights(dec_level_config, tfk_dataset)
        except Exception as e:
            msg = f"Feature weights calculation Failed -- procesing without weights -- {e}"
            logger.warning(msg)

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

            plot_tf_obj(ttfz_obj, out_filename="")

    tf_collection = TransferFunctionCollection(
        tf_dict=tf_dict, processing_config=tfk.config
    )

    try:
        tf_cls = tfk.export_tf_collection(tf_collection)
        logger.info(f"type(tf_cls): {type(tf_cls)}")
        if z_file_path:
            tf_cls.write(z_file_path)
            logger.info(f"Transfer function object written to {z_file_path}")
    except Exception as e:
        msg = "TF collection could not export to mt_metadata TransferFunction\n"
        msg += f"Failed with exception {e}\n"
        msg += "Perhaps an unconventional mixture of input/output channels was used\n"
        msg += f"Input channels were {tfk.config.decimations[0].input_channels}\n"
        msg += f"Output channels were {tfk.config.decimations[0].output_channels}\n"
        msg += "No z-file will be written in this case\n"
        msg += "Will return a legacy TransferFunctionCollection object, not mt_metadata object."
        logger.error(msg)
        return_collection = True

    tfk.dataset.close_mth5s()
    if return_collection:
        return tf_collection  # mostly used for debugging and may be deprecated.
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
        raise NotImplementedError(f"Processing type {processing_type} not supported")

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
