from aurora.pipelines.process_mth5 import process_mth5


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
        z_file_path="test1_RRtest2.zrr",
    )
    return tf_cls
