"""Pytest translation of test_multi_run.py

Tests multi-run processing scenarios including individual runs, combined runs,
and runs with truncated data.
"""

import logging

import pandas as pd
import pytest
from mth5.helpers import close_open_files
from mth5.processing import KernelDataset, RunSummary

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5


@pytest.fixture(autouse=True)
def setup_logging():
    """Disable noisy matplotlib loggers."""
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.ticker").disabled = True


@pytest.fixture(scope="module")
def run_summary_test3(worker_safe_test3_h5):
    """Create a RunSummary from test3.h5 MTH5 file."""
    close_open_files()
    mth5_paths = [worker_safe_test3_h5]
    run_summary = RunSummary()
    run_summary.from_mth5s(mth5_paths)
    return run_summary


class TestMultiRunProcessing:
    """Tests for multi-run processing scenarios - cache expensive process_mth5 calls."""

    @pytest.fixture(scope="class")
    def kernel_dataset_test3(self, run_summary_test3):
        """Create kernel dataset for test3."""
        kernel_dataset = KernelDataset()
        kernel_dataset.from_run_summary(run_summary_test3, "test3")
        return kernel_dataset

    @pytest.fixture(scope="class")
    def config_test3(self, kernel_dataset_test3):
        """Create config for test3 with RME estimator."""
        cc = ConfigCreator()
        return cc.create_from_kernel_dataset(
            kernel_dataset_test3, estimator={"engine": "RME"}
        )

    @pytest.fixture(scope="class")
    def processed_tf_all_runs(
        self, kernel_dataset_test3, config_test3, synthetic_test_paths
    ):
        """Process all runs together - expensive operation, done once."""
        close_open_files()
        z_file_path = synthetic_test_paths.aurora_results_path.joinpath("syn3_all.zss")
        return process_mth5(
            config_test3,
            kernel_dataset_test3,
            units="MT",
            show_plot=False,
            z_file_path=z_file_path,
        )

    def test_all_runs(self, processed_tf_all_runs, synthetic_test_paths):
        """Test processing all runs together."""
        xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
            "syn3_all.xml"
        )
        processed_tf_all_runs.write(fn=xml_file_name, file_type="emtfxml")


def test_each_run_individually(run_summary_test3, synthetic_test_paths, subtests):
    """Test processing each run individually.

    Note: This test must process each run separately, so it cannot use class fixtures.
    It processes 4 runs individually which is inherently expensive.
    """
    close_open_files()

    for run_id in run_summary_test3.df.run.unique():
        with subtests.test(run=run_id):
            kernel_dataset = KernelDataset()
            kernel_dataset.from_run_summary(run_summary_test3, "test3")
            station_runs_dict = {}
            station_runs_dict["test3"] = [run_id]
            keep_or_drop = "keep"
            kernel_dataset.select_station_runs(station_runs_dict, keep_or_drop)

            cc = ConfigCreator()
            config = cc.create_from_kernel_dataset(kernel_dataset)

            for decimation in config.decimations:
                decimation.estimator.engine = "RME"

            show_plot = False
            z_file_path = synthetic_test_paths.aurora_results_path.joinpath(
                f"syn3_{run_id}.zss"
            )
            tf_cls = process_mth5(
                config,
                kernel_dataset,
                units="MT",
                show_plot=show_plot,
                z_file_path=z_file_path,
            )

            xml_file_base = f"syn3_{run_id}.xml"
            xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
                xml_file_base
            )
            tf_cls.write(fn=xml_file_name, file_type="emtfxml")


def test_works_with_truncated_run(run_summary_test3, synthetic_test_paths):
    """Test processing with a truncated run.

    Synthetic runs are 40000s long. By truncating one of the runs to 10000s,
    we make the 4th decimation invalid for that run. By truncating to 2000s
    long we make the 3rd and 4th decimation levels invalid.

    Note: This test modifies run_summary, so it cannot use class fixtures.
    """
    # Make a copy of the run summary to avoid modifying the fixture
    import copy

    run_summary = copy.deepcopy(run_summary_test3)

    delta = pd.Timedelta(seconds=38000)
    run_summary.df.loc[1, "end"] -= delta

    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, "test3")

    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(kernel_dataset, estimator={"engine": "RME"})

    show_plot = False
    z_file_path = synthetic_test_paths.aurora_results_path.joinpath(
        "syn3_all_truncated_run.zss"
    )
    tf_cls = process_mth5(
        config,
        kernel_dataset,
        units="MT",
        show_plot=show_plot,
        z_file_path=z_file_path,
    )

    # process_mth5 may return None if insufficient data after truncation
    if tf_cls is not None:
        xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
            "syn3_all_truncated_run.xml"
        )
        tf_cls.write(fn=xml_file_name, file_type="emtfxml")
