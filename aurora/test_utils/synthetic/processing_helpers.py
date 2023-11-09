from aurora.pipelines.process_mth5 import process_mth5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test1_h5


def get_example_kernel_dataset():
    """
    Some tests could benefit from having a ready-made kernel dataset object.
    This creates one from the synthetic data.

    Returns
    -------
    kernel_dataset: aurora.transfer_function.kernel_dataset.KernelDataset
        The kernel dataset from a synthetic, single station mth5
    """

    from aurora.pipelines.run_summary import RunSummary
    from aurora.transfer_function.kernel_dataset import KernelDataset

    mth5_path = create_test1_h5(force_make_mth5=False)

    run_summary = RunSummary()
    run_summary.from_mth5s(
        [
            mth5_path,
        ]
    )

    kernel_dataset = KernelDataset()
    station_id = run_summary.df.station_id.iloc[0]
    kernel_dataset.from_run_summary(run_summary, station_id)
    return kernel_dataset


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


# def main():
#     kd = get_example_kernel_dataset()
#     return
#
# if __name__ == "__main__":
#     main()
