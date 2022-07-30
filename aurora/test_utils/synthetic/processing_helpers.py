from pathlib import Path

from aurora.config.metadata.processing import Processing
from aurora.pipelines.helpers import initialize_config
from aurora.pipelines.process_mth5 import process_mth5


def process_sythetic_data(
    processing_config, tfk_dataset, units="MT", z_file_path="", return_collection=True
):
    """

    Parameters
    ----------
    processing_config: str or Path, or a Processing() object
        where the processing configuration file is found
    tfk_dataset: aurora.tf_kernel.dataset.Dataset
        class that has a df that describes the runs to be processed.
    z_file_path: str or Path
        Optional, a place to store the output TF in EMTF z-file format.

    Returns
    -------
    tf_collection:
    aurora.transfer_function.transfer_function_collection.TransferFunctionCollection
        Container for TF.  TransferFunctionCollection will probably be deprecated.

    """
    cond1 = isinstance(processing_config, str)
    cond2 = isinstance(processing_config, Path)
    if cond1 or cond2:
        # load from a json path or string
        print("Not tested since implementation of new mt_metadata Processing object")
        config = initialize_config(processing_config)
    elif isinstance(processing_config, Processing):
        config = processing_config
    else:
        print(f"processing_config has unexpected type {type(processing_config)}")
        raise Exception

    tf_collection = process_mth5(
        config,
        tfk_dataset,
        units=units,
        z_file_path=z_file_path,
        return_collection=return_collection,
    )
    return tf_collection
