from pathlib import Path

from mt_metadata.transfer_functions.processing.aurora import Processing
from aurora.pipelines.helpers import initialize_config
from aurora.pipelines.process_mth5 import process_mth5


def process_synthetic_data(
    processing_config, tfk_dataset, units="MT", z_file_path="", return_collection=False
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


def tf_obj_from_synthetic_data(mth5_path):
    """Helper function for test_issue_139"""
    from aurora.config.config_creator import ConfigCreator
    from aurora.pipelines.run_summary import RunSummary
    from aurora.transfer_function.kernel_dataset import KernelDataset

    run_summary = RunSummary()
    run_summary.from_mth5s(list((mth5_path,)))

    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, "test1", "test2")

    # Define the processing Configuration
    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(kernel_dataset)

    tf_cls = process_mth5(
        config,
        kernel_dataset,
        units="MT",
        z_file_path="zzz.zz",
    )
    return tf_cls
