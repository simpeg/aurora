from loguru import logger

import datetime
import unittest

from mth5.data.make_mth5_from_asc import create_test1_h5
from mth5.data.paths import SyntheticTestPaths
from mth5.helpers import close_open_files
from mth5.utils.helpers import initialize_mth5

synthetic_test_paths = SyntheticTestPaths()
MTH5_PATH = synthetic_test_paths.mth5_path


class TestSlicingRunTS(unittest.TestCase):
    """
    This will get moved into MTH5
    """

    @classmethod
    def setUpClass(self):
        close_open_files()
        self.mth5_path = MTH5_PATH.joinpath("test1.h5")
        if not self.mth5_path.exists():
            create_test1_h5(file_version="0.1.0")

    def setUp(self):
        pass

    def test_can_slice_a_run_ts_using_timestamp(self):
        mth5_obj = initialize_mth5(self.mth5_path, "r")
        df = mth5_obj.channel_summary.to_dataframe()
        try:
            run_001 = mth5_obj.get_run(station_name="test1", run_name="001")
        except ValueError:
            # this can happen on local machine
            run_001 = mth5_obj.get_run(
                station_name="test1",
                run_name="001",
                survey=mth5_obj.surveys_group.groups_list[0],
            )
        run_ts_01 = run_001.to_runts()
        start = df.iloc[0].start
        end = df.iloc[0].end
        run_ts_02 = run_001.to_runts(start=start, end=end)
        run_ts_03 = run_001.to_runts(
            start=start, end=end + datetime.timedelta(microseconds=499999)
        )

        run_ts_04 = run_001.to_runts(
            start=start, end=end + datetime.timedelta(microseconds=500000)
        )
        logger.info(f"run_ts_01 has {len(run_ts_01.dataset.ex.data)} samples")
        logger.info(f"run_ts_02 has {len(run_ts_02.dataset.ex.data)} samples")
        logger.info(f"run_ts_03 has {len(run_ts_03.dataset.ex.data)} samples")
        logger.info(f"run_ts_04 has {len(run_ts_04.dataset.ex.data)} samples")


def main():
    unittest.main()
    # test_can_slice_a_run_ts_using_timestamp()


if __name__ == "__main__":
    main()
