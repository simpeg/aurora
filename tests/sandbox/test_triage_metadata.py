import pathlib
import unittest

from aurora.sandbox.triage_metadata import triage_run_id
from mth5.mth5 import MTH5
from mth5.utils.helpers import initialize_mth5


class TestTriageMetadata(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(cls):
        cls.fn = pathlib.Path("tmp.h5")
        cls.station_id = "MT1"
        cls.expected_run_id = "001"
        cls.erroneous_run_id = "123"
        m = initialize_mth5(cls.fn)
        m.add_station(cls.station_id)
        m.add_run(cls.station_id, cls.erroneous_run_id)
        m.close_mth5()

    def test_triage_run_id(self):
        m = MTH5()
        m.open_mth5(self.fn)
        run_obj = m.get_run(self.station_id, self.erroneous_run_id)
        assert run_obj.metadata.id == self.erroneous_run_id
        triage_run_id(self.expected_run_id, run_obj)
        assert run_obj.metadata.id == self.expected_run_id
        return

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fn.unlink()


# def main():
#     unittest.main()
#
#
# if __name__ == "__main__":
#     main()
