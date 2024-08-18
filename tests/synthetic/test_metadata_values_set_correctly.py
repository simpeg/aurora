"""
TODO: Deprecate -- This now basically duplicates a test in MTH5 (issue #191)
"""

from loguru import logger
import logging
import pandas as pd
import unittest

# from mtpy-v2
from mtpy.processing import RunSummary
from mth5.data.make_mth5_from_asc import create_test3_h5
from mth5.data.station_config import make_station_03
from mth5.helpers import close_open_files


class TestMetadataValuesSetCorrect(unittest.TestCase):
    """
    Tests setting of start time as per aurora issue #188
    """

    remake_mth5_for_each_test = False

    def setUp(self):
        close_open_files()
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.getLogger("matplotlib.ticker").disabled = True

    def make_mth5(self):
        close_open_files()
        mth5_path = create_test3_h5(
            force_make_mth5=self.remake_mth5_for_each_test
        )
        return mth5_path

    def make_run_summary(self):
        mth5_path = self.make_mth5()
        mth5s = [
            mth5_path,
        ]
        run_summary = RunSummary()
        run_summary.from_mth5s(mth5s)
        return run_summary

    def test_start_times_correct(self):
        run_summary = self.make_run_summary()
        run_summary
        station_03 = make_station_03()
        for run in station_03.runs:
            summary_row = run_summary.df[
                run_summary.df.run == run.run_metadata.id
            ].iloc[0]
            logger.info(summary_row.start)
            logger.info(run.start)
            assert summary_row.start == pd.Timestamp(run.start)

    def tearDown(self):
        close_open_files()


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
