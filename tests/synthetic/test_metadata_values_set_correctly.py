import logging
import pandas as pd
import unittest

from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test3_h5
from aurora.test_utils.synthetic.station_config import make_station_03
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

    def test_start_times_correct(self):
        run_summary = self.make_run_summary()
        run_summary
        station_03 = make_station_03()
        for run in station_03.runs:
            summary_row = run_summary.df[run_summary.df.run_id == run.id].iloc[0]
            assert summary_row.start == pd.Timestamp(run.start)


def main():
    # tmp = TestMetadataValuesSetCorrect()
    # tmp.setUp()
    # tmp.test_start_times_correct()
    unittest.main()


if __name__ == "__main__":
    main()
