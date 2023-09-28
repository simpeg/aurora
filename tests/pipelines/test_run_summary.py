#import logging
import unittest

from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
from aurora.test_utils.synthetic.paths import DATA_PATH


class TestRunSummary(unittest.TestCase):
    """ """
    @classmethod
    def setUpClass(self):
        self._mth5_path = DATA_PATH.joinpath("test12rr.h5")
        if not self._mth5_path.exists():
            self._mth5_path = create_test12rr_h5()
        self._rs = RunSummary()
        self._rs.from_mth5s([self._mth5_path,])
        print("OK")

    def setUp(self):
        rs = self._rs.clone()

    def test_add_duration(self):
        rs = self._rs.clone()
        rs.add_duration()
        assert("duration" in rs.df.columns)




def main():
    # tmp = TestRunSummary()
    # tmp.setUpClass()
    # tmp.test_add_duration()
    unittest.main()


if __name__ == "__main__":
    main()
