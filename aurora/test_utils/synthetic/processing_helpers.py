from pathlib import Path

from aurora.config.metadata.processing import Processing
from aurora.pipelines.helpers import initialize_config
from aurora.pipelines.process_mth5_new import process_mth5

def process_sythetic_data(processing_config, dataset_definition, units="MT",
                          z_file_path=""):
    """

    Parameters
    ----------
    processing_config: str or Path, or a Processing() object
        where the processing configuration file is found
    dataset_definition: aurora.tf_kernel.dataset.DatasetDefinition
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
    if (cond1 or cond2):
        print("This needs to be updated to work with new mt_metadata Processing object")
        #load from a json path or string
        config = initialize_config(processing_config)
    elif isinstance(processing_config, Processing):
        config = processing_config
    else:
        print(f"processing_config has unexpected type {type(processing_config)}")
        raise Exception

    tf_collection = process_mth5(config, dataset_definition, units=units, 
                                 z_file_path=z_file_path)
    return tf_collection
