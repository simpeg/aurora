import logging
import unittest
from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from mth5.data.make_mth5_from_asc import create_test3_h5
from mth5.helpers import close_open_files

# from mtpy-v2
from mtpy.processing import RunSummary, KernelDataset

synthetic_test_paths = SyntheticTestPaths()
synthetic_test_paths.mkdirs()
AURORA_RESULTS_PATH = synthetic_test_paths.aurora_results_path


class TestMultiRunProcessing(unittest.TestCase):
    """
    Runs several synthetic multi-run processing tests from config creation to
    tf_collection.

    """

    remake_mth5_for_each_test = False

    def setUp(self):
        close_open_files()
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.getLogger("matplotlib.ticker").disabled = True

    @classmethod
    def setUpClass(cls) -> None:
        """Add a fresh h5 to start the test, sowe don't have FCs in there from other tests"""
        create_test3_h5(force_make_mth5=True)

    def make_mth5(self):
        close_open_files()
        mth5_path = create_test3_h5(force_make_mth5=self.remake_mth5_for_each_test)
        return mth5_path

    def make_run_summary(self):
        mth5_path = self.make_mth5()
        mth5s = [
            mth5_path,
        ]
        run_summary = RunSummary()
        run_summary.from_mth5s(mth5s)
        return run_summary

    def test_each_run_individually(self):
        close_open_files()
        run_summary = self.make_run_summary()
        for run_id in run_summary.df.run.unique():
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
            tf_cls.write(fn=xml_file_name, file_type="emtfxml")

    def test_all_runs(self):
        close_open_files()
        run_summary = self.make_run_summary()
        kernel_dataset = KernelDataset()
        kernel_dataset.from_run_summary(run_summary, "test3")
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(
            kernel_dataset, estimator={"engine": "RME"}
        )

        show_plot = False  # True
        z_file_path = AURORA_RESULTS_PATH.joinpath("syn3_all.zss")
        tf_cls = process_mth5(
            config,
            kernel_dataset,
            units="MT",
            show_plot=show_plot,
            z_file_path=z_file_path,
        )
        xml_file_name = AURORA_RESULTS_PATH.joinpath("syn3_all.xml")
        tf_cls.write(fn=xml_file_name, file_type="emtfxml")

    def test_works_with_truncated_run(self):
        """
        Synthetic runs are 40000s long.  By truncating one of the runs to 10000s,
        we make the 4th decimation invalid for that run invalid.  By truncating to
        2000s long we make the 3rd and 4th decimation levels invalid.
        Returns
        -------

        """
        import pandas as pd

        run_summary = self.make_run_summary()
        delta = pd.Timedelta(seconds=38000)
        run_summary.df.end.iloc[1] -= delta
        kernel_dataset = KernelDataset()
        kernel_dataset.from_run_summary(run_summary, "test3")
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(
            kernel_dataset, estimator={"engine": "RME"}
        )

        show_plot = False  # True
        z_file_path = AURORA_RESULTS_PATH.joinpath("syn3_all_truncated_run.zss")
        tf_cls = process_mth5(
            config,
            kernel_dataset,
            units="MT",
            show_plot=show_plot,
            z_file_path=z_file_path,
        )
        xml_file_name = AURORA_RESULTS_PATH.joinpath("syn3_all_truncated_run.xml")
        tf_cls.write(fn=xml_file_name, file_type="emtfxml")


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
