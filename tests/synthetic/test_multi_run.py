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
    run_summary.from_mth5s([mth5_path,])
    print(run_summary.df)

    for run_id in run_summary.df.run_id.unique():
        kernel_dataset = KernelDataset()
        kernel_dataset.from_run_summary(run_summary, "test3")
        station_runs_dict = {}
        station_runs_dict["test3"] = [run_id,]
        keep_or_drop = "keep"
        kernel_dataset.select_station_runs(station_runs_dict, keep_or_drop)
        print(kernel_dataset.df)
        cc = ConfigCreator()
        sample_rate = kernel_dataset.df.sample_rate.iloc[0]
        config = cc.create_run_processing_object(emtf_band_file=BANDS_DEFAULT_FILE,
                                                 sample_rate=sample_rate
                                                 )
        config.stations.from_dataset_dataframe(kernel_dataset.df)
        for decimation in config.decimations:
            decimation.estimator.engine = "RME"
        show_plot = False #True
        z_file_path = AURORA_RESULTS_PATH.joinpath(f"syn3_{run_id}.zss")
        tf_cls = process_mth5(config,
                          kernel_dataset,
                          units="MT",
                          show_plot=show_plot,
                          z_file_path=z_file_path,
                          return_collection=False
                          )




def test_all_runs():
    mth5_path = create_test3_h5()
    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_path,])
    print(run_summary.df)

    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, "test3")
    cc = ConfigCreator()
    sample_rate = kernel_dataset.df.sample_rate.iloc[0]
    config = cc.create_run_processing_object(emtf_band_file=BANDS_DEFAULT_FILE,
                                             sample_rate=sample_rate
                                             )
    config.stations.from_dataset_dataframe(kernel_dataset.df)
    for decimation in config.decimations:
        decimation.estimator.engine = "RME"
    show_plot = False#True
    z_file_path = AURORA_RESULTS_PATH.joinpath("syn3_all.zss")
    tf_cls = process_mth5(config,
                          kernel_dataset,
                          units="MT",
                          show_plot=show_plot,
                          z_file_path=z_file_path,
                          return_collection=False
                          )


def test():
    test_all_runs()
    #test_each_run_individually()

if __name__ == "__main__":
    test()