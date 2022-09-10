from aurora.config import BANDS_DEFAULT_FILE
from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test3_h5
from aurora.test_utils.synthetic.paths import AURORA_RESULTS_PATH
from aurora.transfer_function.kernel_dataset import KernelDataset


def test_each_run_individually():
    mth5_path = create_test3_h5()
    run_summary = RunSummary()
    run_summary.from_mth5s(
        [
            mth5_path,
        ]
    )
    print(run_summary.df)

    for run_id in run_summary.df.run_id.unique():
        kernel_dataset = KernelDataset()
        kernel_dataset.from_run_summary(run_summary, "test3")
        station_runs_dict = {}
        station_runs_dict["test3"] = [
            run_id,
        ]
        keep_or_drop = "keep"
        kernel_dataset.select_station_runs(station_runs_dict, keep_or_drop)
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(kernel_dataset)

        for decimation in config.decimations:
            decimation.estimator.engine = "RME"
        show_plot = False  # True
        z_file_path = AURORA_RESULTS_PATH.joinpath(f"syn3_{run_id}.zss")
        tf_cls = process_mth5(
            config,
            kernel_dataset,
            units="MT",
            show_plot=show_plot,
            z_file_path=z_file_path,
        )
        xml_file_base = f"syn3_{run_id}.xml"
        xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
        tf_cls.write_tf_file(fn=xml_file_name, file_type="emtfxml")


def test_all_runs():
    mth5_path = create_test3_h5()
    run_summary = RunSummary()
    run_summary.from_mth5s(
        [
            mth5_path,
        ]
    )
    print(run_summary.df)

    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, "test3")
    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(kernel_dataset, estimator={"engine": "RME"})

    show_plot = False  # True
    z_file_path = AURORA_RESULTS_PATH.joinpath("syn3_all.zss")
    tf_cls = process_mth5(
        config, kernel_dataset, units="MT", show_plot=show_plot, z_file_path=z_file_path
    )
    xml_file_name = AURORA_RESULTS_PATH.joinpath("syn3_all.xml")
    tf_cls.write_tf_file(fn=xml_file_name, file_type="emtfxml")


def test():
    test_each_run_individually()
    test_all_runs()


if __name__ == "__main__":
    test()
