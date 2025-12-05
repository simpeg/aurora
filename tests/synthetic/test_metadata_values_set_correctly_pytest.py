"""
TODO: Deprecate -- This now basically duplicates a test in MTH5 (issue #191)

Tests setting of start time as per aurora issue #188
"""

import logging

import pandas as pd
import pytest
from loguru import logger
from mth5.data.station_config import make_station_03
from mth5.helpers import close_open_files
from mth5.processing import RunSummary


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


def test_start_times_correct(run_summary_test3, subtests):
    """Test that start times in run summary match station configuration."""
    station_03 = make_station_03()

    for run in station_03.runs:
        with subtests.test(run=run.run_metadata.id):
            summary_row = run_summary_test3.df[
                run_summary_test3.df.run == run.run_metadata.id
            ].iloc[0]
            logger.info(summary_row.start)
            logger.info(run.run_metadata.time_period.start)
            assert summary_row.start == pd.Timestamp(
                str(run.run_metadata.time_period.start)
            )
